// How much does ONE cusparseSpMV call actually cost, and how much of that
// is the algorithm choice?
//
// This exists because of a specific unexplained measurement.
// benchmarks/bench_gpu_latency.cpp showed a GPU pricing iteration spending
// 124-156 us inside its event wait, against a 28 us round trip for an empty
// kernel -- roughly 100 us unaccounted for, and flat in problem size.
// Cutting the number of queued operations from 11 to 8 moved it by ~5 us,
// which refuted the per-operation explanation. The only heavyweight item
// left in the queue is cusparseSpMV.
//
// The suspicion is specific and consequential: this project uses
// CUSPARSE_SPMV_CSR_ALG2, chosen deliberately for determinism
// (docs/architecture/NUMERICS.md 1 -- ALG1's row-splitting makes the
// summation order depend on scheduling, so it is not bit-reproducible).
// ALG2 buys that reproducibility with load-balancing analysis. If that
// analysis is being redone on every call, the determinism guarantee is
// costing far more than it should, and cusparseSpMV_preprocess exists
// precisely to hoist it out.
//
// That matters well beyond pricing. Any first-order LP method is SpMV in a
// loop, so a slow per-call SpMV would sink that approach before it started.
// This is the measurement that says whether the GPU path has a future.
//
// Reported per algorithm:
//   - per-call wall time with a synchronize (what a latency-bound caller pays)
//   - per-call throughput when many calls are queued back to back (what a
//     throughput-bound caller pays)
//   - whether repeated runs are bit-identical, which is the property ALG2
//     is being paid for in the first place

#include "cuda/CudaDevice.hpp"
#include "cuda/CudaStream.hpp"
#include "cuda/CusparseHandle.hpp"
#include "cuda/GpuSpMV.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Convert.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace sihps;

namespace {

struct AlgCase {
    const char* name;
    cusparseSpMVAlg_t alg;
};

double now_us(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6;
}

// Builds the CSR of A^T the same way the solver does -- CSC's arrays
// reinterpreted -- so the matrix measured here is exactly the one pricing
// multiplies against.
CSRMatrix transpose_as_csr(const CSRMatrix& a) {
    const CSCMatrix csc = csr_to_csc(a);
    std::vector<std::int32_t> row_ptr(csc.col_ptr(), csc.col_ptr() + a.cols() + 1);
    std::vector<std::int32_t> col_idx(csc.row_idx(), csc.row_idx() + csc.nnz());
    std::vector<double> values(csc.values(), csc.values() + csc.nnz());
    return CSRMatrix(a.cols(), a.rows(), std::move(row_ptr), std::move(col_idx),
                      std::move(values));
}

void measure(const char* stem, const CSRMatrix& at) {
    std::printf("\n%s -- A^T is %d x %d, %d nnz\n", stem, at.rows(), at.cols(), at.nnz());
    std::printf("  %-14s %14s %16s %12s\n", "algorithm", "per call+sync", "per call queued",
                "reproducible");
    std::printf("  %s\n", std::string(62, '-').c_str());

    const AlgCase cases[] = {
        {"CSR_ALG1", CUSPARSE_SPMV_CSR_ALG1},
        {"CSR_ALG2", CUSPARSE_SPMV_CSR_ALG2},
        {"ALG_DEFAULT", CUSPARSE_SPMV_ALG_DEFAULT},
    };

    std::vector<double> x(static_cast<std::size_t>(at.cols()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 1.0 / (1.0 + static_cast<double>(i % 89));
    }
    std::vector<double> y(static_cast<std::size_t>(at.rows()), 0.0);
    std::vector<double> reference;

    for (const AlgCase& c : cases) {
        CusparseHandle handle;
        CudaStream stream;
        GpuCsrSpMV spmv(at, handle, stream, c.alg);

        spmv.multiply(x.data(), y.data());
        if (reference.empty()) reference = y;

        constexpr int kWarmup = 100;
        constexpr int kReps = 1000;

        // Latency: what a caller pays when it must have the answer now.
        // multiply() leaves x device-resident, so every timed call below
        // reuses it and measures the SpMV alone rather than the transfer.
        for (int i = 0; i < kWarmup; ++i) spmv.multiply_device_resident();
        stream.synchronize();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) {
            spmv.multiply_device_resident();
            stream.synchronize();
        }
        const double sync_us = now_us(t0) / kReps;

        // Throughput: what it costs when many calls are in flight, which is
        // the regime a first-order method runs in.
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) spmv.multiply_device_resident();
        stream.synchronize();
        const double queued_us = now_us(t0) / kReps;

        // Bit-reproducibility across repeated runs of the SAME algorithm.
        // This is the property ALG2 is chosen for; ALG1 is expected to fail
        // it on some matrices and pass by luck on others, which is exactly
        // why it cannot be relied on.
        bool identical = true;
        std::vector<double> y2(static_cast<std::size_t>(at.rows()), 0.0);
        for (int rep = 0; rep < 5 && identical; ++rep) {
            spmv.multiply(x.data(), y2.data());
            identical = std::equal(y.begin(), y.end(), y2.begin());
        }

        std::printf("  %-14s %11.2f us %13.2f us %12s\n", c.name, sync_us, queued_us,
                    identical ? "yes" : "NO");
    }
}

} // namespace

int main(int argc, char** argv) {
    const DeviceInfo info = CudaDevice::select(0);
    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", info.name.c_str(), info.compute_capability_major,
                info.compute_capability_minor, info.multiprocessor_count);
    std::printf("cuSPARSE %d\n", CUSPARSE_VERSION);

    std::vector<std::string> stems;
    for (int i = 1; i < argc; ++i) stems.push_back(argv[i]);
    if (stems.empty()) stems = {"sctap1", "wood1p", "fit2d", "d6cube"};

    for (const std::string& stem : stems) {
        try {
            const auto model = read_mps_file("data/netlib_lp/feasible/" + stem + ".mps");
            const LpProblem p = lp_problem_from_mps(model);
            measure(stem.c_str(), transpose_as_csr(p.A));
        } catch (const std::exception& e) {
            std::printf("\n%s -- unavailable: %s\n", stem.c_str(), e.what());
        }
    }

    std::printf("\nRead 'per call+sync' against bench_gpu_latency's ~124-156 us pricing wait:\n");
    std::printf("if SpMV alone accounts for most of it, the algorithm choice is the bottleneck,\n");
    std::printf("not the number of queued operations.\n");
    return 0;
}
