// CPU/GPU SpMV equivalence tests -- prompt.md \S3.7, validation hierarchy
// Level 4. Every test here computes y = A*x once on the CPU
// (CSRMatrix::multiply, already validated independently in
// tests/sparse/test_csr_csc.cpp) and once on the GPU (GpuCsrSpMV, via
// cuSPARSE), then checks both absolute and relative residual between the
// two -- never just one or the other, per prompt.md's explicit instruction.

#include "../test_framework.hpp"
#include "cuda/CudaDevice.hpp"
#include "cuda/CudaStream.hpp"
#include "cuda/CusparseHandle.hpp"
#include "cuda/GpuSpMV.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <random>
#include <vector>

using sihps::CSRMatrix;
using sihps::CudaDevice;
using sihps::CudaStream;
using sihps::CusparseHandle;
using sihps::GpuCsrSpMV;
using sihps::Triplet;

namespace {

struct Residuals {
    double max_abs;
    double max_rel;
};

Residuals compare(const std::vector<double>& cpu, const std::vector<double>& gpu) {
    double max_abs = 0.0, max_rel = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        double abs_diff = std::fabs(cpu[i] - gpu[i]);
        double denom = 1.0 + std::fabs(cpu[i]);
        double rel_diff = abs_diff / denom;
        max_abs = std::max(max_abs, abs_diff);
        max_rel = std::max(max_rel, rel_diff);
    }
    return {max_abs, max_rel};
}

std::vector<Triplet> random_triplets(std::int32_t rows, std::int32_t cols, double density,
                                      unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);
    std::vector<Triplet> t;
    for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t j = 0; j < cols; ++j) {
            if (unit(rng) < density) t.push_back({i, j, val(rng)});
        }
    }
    return t;
}

// FP64 relative tolerance: loose enough to absorb legitimate
// summation-order differences between CPU row-major accumulation and
// cuSPARSE's internal reduction strategy (which is not required to match
// any particular order), tight enough to catch a real correctness bug.
// docs/architecture/NUMERICS.md \S3 uses 1e-8 as the LP-level residual
// tolerance; SpMV itself, being a single un-iterated computation, is held
// to a substantially tighter bound here.
constexpr double kAbsTol = 1e-9;
constexpr double kRelTol = 1e-9;

void check_equivalence(const CSRMatrix& a, const std::vector<double>& x) {
    std::vector<double> y_cpu(static_cast<std::size_t>(a.rows()));
    a.multiply(x.data(), y_cpu.data());

    CudaDevice::select(0);
    CusparseHandle handle;
    CudaStream stream;
    GpuCsrSpMV gpu(a, handle, stream);

    std::vector<double> y_gpu(static_cast<std::size_t>(a.rows()));
    gpu.multiply(x.data(), y_gpu.data());

    auto res = compare(y_cpu, y_gpu);
    SIHPS_ASSERT_TRUE(res.max_abs < kAbsTol);
    SIHPS_ASSERT_TRUE(res.max_rel < kRelTol);
}

} // namespace

SIHPS_TEST(gpu_spmv_matches_cpu_on_diagonal_matrix) {
    std::vector<Triplet> t = {{0, 0, 2.0}, {1, 1, 3.0}, {2, 2, 4.0}};
    CSRMatrix a = CSRMatrix::from_triplets(3, 3, t);
    check_equivalence(a, {1.0, 1.0, 1.0});
}

SIHPS_TEST(gpu_spmv_matches_cpu_on_dense_matrix) {
    std::int32_t n = 16;
    std::vector<Triplet> t;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> val(-5.0, 5.0);
    for (std::int32_t i = 0; i < n; ++i) {
        for (std::int32_t j = 0; j < n; ++j) t.push_back({i, j, val(rng)});
    }
    CSRMatrix a = CSRMatrix::from_triplets(n, n, t);
    std::vector<double> x(static_cast<std::size_t>(n));
    for (auto& v : x) v = val(rng);
    check_equivalence(a, x);
}

SIHPS_TEST(gpu_spmv_matches_cpu_on_rectangular_matrix) {
    auto t = random_triplets(50, 30, 0.1, 11);
    CSRMatrix a = CSRMatrix::from_triplets(50, 30, t);
    std::mt19937 rng(12);
    std::uniform_real_distribution<double> val(-4.0, 4.0);
    std::vector<double> x(30);
    for (auto& v : x) v = val(rng);
    check_equivalence(a, x);
}

SIHPS_TEST(gpu_spmv_matches_cpu_on_pathologically_sparse_matrix) {
    // ~0.01% density on a 500x500 matrix -- most rows are empty.
    auto t = random_triplets(500, 500, 0.0001, 21);
    CSRMatrix a = CSRMatrix::from_triplets(500, 500, t);
    std::vector<double> x(500, 1.0);
    check_equivalence(a, x);
}

SIHPS_TEST(gpu_spmv_matches_cpu_on_dense_ish_matrix) {
    auto t = random_triplets(80, 80, 0.6, 31);
    CSRMatrix a = CSRMatrix::from_triplets(80, 80, t);
    std::mt19937 rng(32);
    std::uniform_real_distribution<double> val(-2.0, 2.0);
    std::vector<double> x(80);
    for (auto& v : x) v = val(rng);
    check_equivalence(a, x);
}

SIHPS_TEST(gpu_spmv_matches_cpu_on_large_random_matrix) {
    auto t = random_triplets(3000, 3000, 0.002, 41);
    CSRMatrix a = CSRMatrix::from_triplets(3000, 3000, t);
    std::vector<double> x(3000, 1.0);
    check_equivalence(a, x);
}

SIHPS_TEST(gpu_spmv_repeated_execution_is_stable) {
    // Same GpuCsrSpMV instance, multiple different x vectors -- verifies
    // no state leaks between calls (e.g. a stale device buffer, or a
    // workspace sized/reused incorrectly).
    auto t = random_triplets(100, 100, 0.05, 51);
    CSRMatrix a = CSRMatrix::from_triplets(100, 100, t);

    CudaDevice::select(0);
    CusparseHandle handle;
    CudaStream stream;
    GpuCsrSpMV gpu(a, handle, stream);

    std::mt19937 rng(52);
    std::uniform_real_distribution<double> val(-3.0, 3.0);
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<double> x(100);
        for (auto& v : x) v = val(rng);

        std::vector<double> y_cpu(100);
        a.multiply(x.data(), y_cpu.data());

        std::vector<double> y_gpu(100);
        gpu.multiply(x.data(), y_gpu.data());

        auto res = compare(y_cpu, y_gpu);
        SIHPS_ASSERT_TRUE(res.max_abs < kAbsTol);
        SIHPS_ASSERT_TRUE(res.max_rel < kRelTol);
    }
}
