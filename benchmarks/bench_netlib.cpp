// SpMV benchmark on real Netlib LP constraint matrices (not synthetic
// random ones) -- prompt.md \S3.8 and the problem statement's own
// "benchmark against MIPLIB/Netlib/Mittelmann" requirement. Structurally
// realistic sparsity (banded/block structure from real LP models) is a
// different test of H1 than uniform-random matrices: real matrices often
// have much more irregular row-length distributions, which is exactly
// what cuSPARSE's algorithm-selection and load-balancing has to handle.
//
// Deliberately capped to files under a size threshold: this is a
// benchmark run repeatedly during development, not a one-shot batch job,
// and the point here is measurement, not stress-testing every Netlib
// instance up to osa-60's 52MB. Files skipped for being too large are
// printed, not silently dropped.

#include "cuda/CudaDevice.hpp"
#include "cuda/CudaEvent.hpp"
#include "cuda/CudaStream.hpp"
#include "cuda/CusparseHandle.hpp"
#include "cuda/GpuSpMV.hpp"
#include "io/MpsReader.hpp"
#include "sparse/CSRMatrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

constexpr std::uintmax_t kMaxFileBytes = 2 * 1024 * 1024; // 2 MB cap for this pass

double cpu_seconds_per_call(const CSRMatrix& a, const std::vector<double>& x,
                             std::vector<double>& y, int repeats) {
    a.multiply(x.data(), y.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) a.multiply(x.data(), y.data());
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / repeats;
}

double gpu_kernel_seconds_per_call(GpuCsrSpMV& gpu, CudaStream& stream, int repeats) {
    gpu.multiply_device_resident();
    stream.synchronize();
    CudaEvent start, end;
    start.record(stream.handle());
    for (int r = 0; r < repeats; ++r) gpu.multiply_device_resident();
    end.record(stream.handle());
    end.synchronize();
    return (static_cast<double>(CudaEvent::elapsed_ms(start, end)) / 1000.0) / repeats;
}

} // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";

    auto info = CudaDevice::select(0);
    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", info.name.c_str(), info.compute_capability_major,
                info.compute_capability_minor, info.multiprocessor_count);
    std::printf("Scanning: %s (cap: %.1f MB per file)\n\n", dir.c_str(),
                kMaxFileBytes / (1024.0 * 1024.0));

    std::vector<fs::path> files;
    if (!fs::exists(dir)) {
        std::printf("Directory does not exist: %s\n", dir.c_str());
        return 1;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    int skipped = 0;
    std::printf("%16s %8s %8s %10s %12s %12s %12s %9s\n", "instance", "rows", "cols", "nnz",
                "CPU us", "GPU-kern us", "GPU-rtrip us", "speedup");
    std::printf("%s\n", std::string(96, '-').c_str());

    for (const auto& path : files) {
        auto size = fs::file_size(path);
        if (size > kMaxFileBytes) {
            ++skipped;
            continue;
        }

        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception& e) {
            std::printf("%16s  PARSE ERROR: %s\n", path.stem().string().c_str(), e.what());
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.constraint_triplets.empty()) {
            std::printf("%16s  skipped (degenerate dimensions)\n", path.stem().string().c_str());
            continue;
        }

        CSRMatrix a = CSRMatrix::from_triplets(model.n_rows, model.n_cols,
                                                model.constraint_triplets);
        std::vector<double> x(static_cast<std::size_t>(model.n_cols), 1.0);
        std::vector<double> y(static_cast<std::size_t>(model.n_rows));

        const int repeats = 30;
        double cpu_s = cpu_seconds_per_call(a, x, y, repeats);

        CusparseHandle handle;
        CudaStream stream;
        GpuCsrSpMV gpu(a, handle, stream);
        std::vector<double> y_gpu(static_cast<std::size_t>(model.n_rows));
        gpu.multiply(x.data(), y_gpu.data());
        double gpu_kernel_s = gpu_kernel_seconds_per_call(gpu, stream, repeats);

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) gpu.multiply(x.data(), y_gpu.data());
        auto t1 = std::chrono::steady_clock::now();
        double gpu_roundtrip_s = std::chrono::duration<double>(t1 - t0).count() / repeats;

        // Cross-check GPU result against CPU result while we're here --
        // free correctness evidence on real-world matrices, not just the
        // synthetic ones in tests/cuda/test_gpu_spmv.cpp.
        double max_abs_diff = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(y[i] - y_gpu[i]));
        }

        std::printf("%16s %8d %8d %10d %12.2f %12.2f %12.2f %8.2fx%s\n",
                    path.stem().string().c_str(), model.n_rows, model.n_cols, a.nnz(),
                    cpu_s * 1e6, gpu_kernel_s * 1e6, gpu_roundtrip_s * 1e6,
                    cpu_s / gpu_kernel_s, max_abs_diff > 1e-6 ? "  [MISMATCH!]" : "");
    }

    std::printf("\n%d file(s) skipped for exceeding the %.1f MB cap for this pass.\n", skipped,
                kMaxFileBytes / (1024.0 * 1024.0));
    return 0;
}
