// SpMV benchmark -- prompt.md \S3.8. Measures, does not assume:
//   - CPU SpMV throughput
//   - GPU SpMV throughput (kernel-only, via CudaEvent timing on
//     device-resident vectors -- isolates the cuSPARSE kernel from PCIe
//     transfer cost)
//   - GPU SpMV throughput including the full host<->device round trip
//     (the realistic cost if this were called once per B&B node)
//   - scaling with nnz and with matrix dimension
//
// This is the direct empirical test of H1 (docs/research/SOTA.md \S5):
// "GPU-accelerated sparse linear algebra can materially accelerate
// repeated LP operations for refinery-scale sparse models" -- on this
// specific RTX 3050 Laptop (4.29GB, 14 SMs), not on the datacenter
// hardware the cited literature used. No number below is fabricated or
// extrapolated; every row is an actual measurement.

#include "cuda/CudaDevice.hpp"
#include "cuda/CudaEvent.hpp"
#include "cuda/CudaStream.hpp"
#include "cuda/CusparseHandle.hpp"
#include "cuda/GpuSpMV.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Triplet.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace sihps;

namespace {

CSRMatrix make_random_csr(std::int32_t n, double density, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(n) * n * density * 1.2);
    for (std::int32_t i = 0; i < n; ++i) {
        for (std::int32_t j = 0; j < n; ++j) {
            if (unit(rng) < density) triplets.push_back({i, j, val(rng)});
        }
    }
    return CSRMatrix::from_triplets(n, n, triplets);
}

double cpu_seconds_per_call(const CSRMatrix& a, const std::vector<double>& x,
                             std::vector<double>& y, int repeats) {
    // Warm-up (first-touch page faults, cache warming) excluded from the
    // timed region.
    a.multiply(x.data(), y.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        a.multiply(x.data(), y.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / repeats;
}

double gpu_kernel_seconds_per_call(GpuCsrSpMV& gpu, CudaStream& stream, int repeats) {
    gpu.multiply_device_resident(); // warm-up: first launch pays one-time driver overhead
    stream.synchronize();

    CudaEvent start, end;
    start.record(stream.handle());
    for (int r = 0; r < repeats; ++r) {
        gpu.multiply_device_resident();
    }
    end.record(stream.handle());
    end.synchronize();
    float ms = CudaEvent::elapsed_ms(start, end);
    return (static_cast<double>(ms) / 1000.0) / repeats;
}

double gpu_roundtrip_seconds_per_call(GpuCsrSpMV& gpu, const std::vector<double>& x,
                                       std::vector<double>& y, int repeats) {
    gpu.multiply(x.data(), y.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        gpu.multiply(x.data(), y.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / repeats;
}

} // namespace

int main() {
    auto info = CudaDevice::select(0);
    std::size_t free_bytes = 0, total_bytes = 0;
    CudaDevice::memory_info(free_bytes, total_bytes);

    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", info.name.c_str(), info.compute_capability_major,
                info.compute_capability_minor, info.multiprocessor_count);
    std::printf("VRAM: %.2f GB total, %.2f GB free at benchmark start\n",
                total_bytes / 1e9, free_bytes / 1e9);
    std::printf("\n%10s %10s %14s %14s %14s %14s %10s\n", "n", "nnz", "CPU us/call",
                "GPU-kernel us", "GPU-rtrip us", "GFLOP/s(CPU)", "speedup");
    std::printf("%s\n", std::string(96, '-').c_str());

    CusparseHandle handle;
    CudaStream stream;

    struct Case { std::int32_t n; double density; };
    const std::vector<Case> cases = {
        {200, 0.05},    {1000, 0.01},  {5000, 0.002}, {10000, 0.001},
        {20000, 0.0005}, {40000, 0.00025},
    };

    for (const auto& c : cases) {
        CSRMatrix a = make_random_csr(c.n, c.density, 1000 + static_cast<unsigned>(c.n));
        std::vector<double> x(static_cast<std::size_t>(c.n), 1.0);
        std::vector<double> y(static_cast<std::size_t>(c.n));

        const int repeats = 20;
        double cpu_s = cpu_seconds_per_call(a, x, y, repeats);

        GpuCsrSpMV gpu(a, handle, stream);
        // Populate device x once for the kernel-only timing (no per-call
        // transfer -- this isolates cuSPARSE's own kernel cost).
        std::vector<double> y_gpu(static_cast<std::size_t>(c.n));
        gpu.multiply(x.data(), y_gpu.data()); // also primes device_x() with x
        double gpu_kernel_s = gpu_kernel_seconds_per_call(gpu, stream, repeats);
        double gpu_roundtrip_s = gpu_roundtrip_seconds_per_call(gpu, x, y_gpu, repeats);

        double gflops_cpu = (2.0 * a.nnz()) / cpu_s / 1e9;
        double speedup_kernel_only = cpu_s / gpu_kernel_s;

        std::printf("%10d %10d %14.2f %14.2f %14.2f %14.3f %9.2fx\n", c.n, a.nnz(), cpu_s * 1e6,
                    gpu_kernel_s * 1e6, gpu_roundtrip_s * 1e6, gflops_cpu, speedup_kernel_only);
    }

    std::printf("\nNotes:\n");
    std::printf("  - 'GPU-kernel' isolates the cuSPARSE SpMV call itself (device-resident x/y,\n");
    std::printf("    no PCIe transfer per call) -- the H1-relevant number.\n");
    std::printf("  - 'GPU-rtrip' includes host<->device transfer each call -- the realistic\n");
    std::printf("    cost if this were invoked once per B&B node without keeping x device-\n");
    std::printf("    resident across calls.\n");
    std::printf("  - 'speedup' = CPU time / GPU-kernel time. <1.0x means CPU was faster.\n");
    std::printf("  - These are measurements on this specific RTX 3050 Laptop (4.29GB, 14 SMs),\n");
    std::printf("    not a claim about datacenter-GPU behavior (docs/research/SOTA.md \\S0).\n");
    return 0;
}
