// What does a GPU pricing iteration actually SPEND its time on?
//
// docs/architecture/CPU_GPU.md §4 established that GPU pricing loses to CPU
// pricing on this workload and that the cause is per-iteration overhead
// rather than bandwidth. "Overhead" is not an actionable diagnosis, so this
// benchmark decomposes it into the two things it could actually be, and
// measures the two available fixes against each other:
//
//   (a) HOST DESCHEDULING ON EVERY SYNC. The CUDA runtime's default host
//       wait policy is cudaDeviceScheduleAuto, which picks yield/blocking
//       when there are many more CPU cores than GPUs -- exactly this
//       machine. Yielding means an OS context switch on every pricing
//       iteration, and a context switch costs more than the wait itself
//       when the wait is tens of microseconds. cudaDeviceScheduleSpin
//       burns a core instead. That is a bad trade for a background job and
//       potentially an excellent one for a latency-bound solver inner loop.
//
//   (b) SUBMISSION COST. A pricing iteration queues on the order of a
//       dozen operations (two H2D copies, two SpMVs, three kernels, a D2H,
//       plus the Devex chain). Each submission costs host time whether or
//       not the GPU is busy. CUDA graphs collapse a fixed sequence into a
//       single submission -- which is precisely the case here, because the
//       sequence is identical every iteration and only the BUFFER CONTENTS
//       change.
//
// Both are measured directly rather than argued for. The chain lengths
// below match what GpuPricer actually issues, so the numbers transfer.
//
// The wait policy must be set before the CUDA context exists, so it comes
// from the environment rather than a command-line flag parsed after main
// has already touched the runtime:
//
//   SIHPS_CUDA_WAIT=spin|yield|blocking|auto  ./bench_gpu_latency

#include "cuda/CudaDevice.hpp"
#include "cuda/CudaStream.hpp"
#include "cuda/GpuPricer.hpp"
#include "cuda/PricingKernels.cuh"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "sparse/Convert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sihps;

namespace {

double now_us(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6;
}

// A stand-in for one pricing iteration's queued work: `kernels` empty
// kernels plus `copies` host-to-device transfers of `bytes` each. Empty
// kernels isolate submission cost from execution cost, which is the point
// -- the real kernels' execution time is measured separately by
// bench_pricing_backend and is not what this file is about.
void submit_chain(cudaStream_t stream, int kernels, int copies, void* d_dst, const void* h_src,
                  std::size_t bytes) {
    for (int i = 0; i < copies; ++i) {
        cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream);
    }
    for (int i = 0; i < kernels; ++i) gpu::launch_noop(stream);
}

} // namespace

int main() {
    // ---- (a) host wait policy, chosen before the context is created ----
    const char* want = std::getenv("SIHPS_CUDA_WAIT");
    const std::string mode = want ? want : "default";
    unsigned flag = 0;
    if (mode == "spin") {
        flag = cudaDeviceScheduleSpin;
    } else if (mode == "yield") {
        flag = cudaDeviceScheduleYield;
    } else if (mode == "blocking") {
        flag = cudaDeviceScheduleBlockingSync;
    } else if (mode == "auto") {
        flag = cudaDeviceScheduleAuto;
    }
    if (flag != 0) {
        const cudaError_t e = cudaSetDeviceFlags(flag);
        if (e != cudaSuccess) {
            std::printf("cudaSetDeviceFlags(%s) failed: %s\n", mode.c_str(),
                        cudaGetErrorString(e));
            return 1;
        }
    }

    const DeviceInfo info = CudaDevice::select(0);
    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", info.name.c_str(), info.compute_capability_major,
                info.compute_capability_minor, info.multiprocessor_count);
    std::printf("host wait policy: %s\n\n", mode.c_str());

    CudaStream stream;

    constexpr int kWarmup = 500;
    constexpr int kReps = 3000;

    // ---------------- baseline: one empty kernel ----------------
    for (int i = 0; i < kWarmup; ++i) gpu::launch_noop(stream.handle());
    stream.synchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) gpu::launch_noop(stream.handle());
    stream.synchronize();
    const double submit_us = now_us(t0) / kReps;

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
        gpu::launch_noop(stream.handle());
        stream.synchronize();
    }
    const double roundtrip_us = now_us(t0) / kReps;

    std::printf("single empty kernel:\n");
    std::printf("  submit only        : %7.2f us\n", submit_us);
    std::printf("  submit + sync      : %7.2f us   (sync alone ~ %.2f us)\n\n", roundtrip_us,
                roundtrip_us - submit_us);

    // ---------------- a realistic chain, launched op by op ----------------
    // 7 kernels + 4 copies is what GpuPricer issues per Devex iteration:
    // status H2D, x H2D, SpMV, fused price, D2H result  (pricing)
    // status H2D, x H2D, SpMV, devex fused, devex finalize  (weights).
    constexpr int kChainKernels = 7;
    constexpr int kChainCopies = 4;
    constexpr std::size_t kCopyBytes = 16384; // ~a mid-size model's status array

    void* d_dst = nullptr;
    cudaMalloc(&d_dst, kCopyBytes);
    void* h_src = nullptr;
    cudaHostAlloc(&h_src, kCopyBytes, cudaHostAllocDefault);
    std::memset(h_src, 0, kCopyBytes);

    for (int i = 0; i < kWarmup; ++i) {
        submit_chain(stream.handle(), kChainKernels, kChainCopies, d_dst, h_src, kCopyBytes);
    }
    stream.synchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
        submit_chain(stream.handle(), kChainKernels, kChainCopies, d_dst, h_src, kCopyBytes);
        stream.synchronize();
    }
    const double chain_us = now_us(t0) / kReps;

    // ---------------- the same chain as a CUDA graph ----------------
    // Captured once, replayed with a single submission. The buffers are the
    // same every time -- only their CONTENTS change between iterations,
    // which is exactly the situation graphs are for.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    double graph_us = -1.0;
    double capture_us = -1.0;

    t0 = std::chrono::steady_clock::now();
    cudaError_t cap = cudaStreamBeginCapture(stream.handle(), cudaStreamCaptureModeRelaxed);
    if (cap == cudaSuccess) {
        submit_chain(stream.handle(), kChainKernels, kChainCopies, d_dst, h_src, kCopyBytes);
        cap = cudaStreamEndCapture(stream.handle(), &graph);
    }
    if (cap == cudaSuccess) {
        cap = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    }
    capture_us = now_us(t0);

    if (cap == cudaSuccess) {
        for (int i = 0; i < kWarmup; ++i) cudaGraphLaunch(graph_exec, stream.handle());
        stream.synchronize();

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) {
            cudaGraphLaunch(graph_exec, stream.handle());
            stream.synchronize();
        }
        graph_us = now_us(t0) / kReps;
    } else {
        std::printf("CUDA graph capture failed: %s\n", cudaGetErrorString(cap));
    }

    std::printf("chain of %d kernels + %d H2D copies (one GpuPricer Devex iteration):\n",
                kChainKernels, kChainCopies);
    std::printf("  op-by-op + sync    : %7.2f us\n", chain_us);
    if (graph_us > 0.0) {
        std::printf("  CUDA graph + sync  : %7.2f us   (%.2fx faster, saves %.1f us/iteration)\n",
                    graph_us, chain_us / graph_us, chain_us - graph_us);
        std::printf("  graph build cost   : %7.2f us   (once, at solve start)\n", capture_us);
    }

    std::printf("\nRead this against bench_pricing_backend's per-iteration pricing times:\n");
    std::printf("  the chain figure is the FLOOR that pricing pays before doing any work.\n\n");

    // ---------------- the real thing, broken down ----------------
    // The synthetic chain above accounts only for submission and waiting.
    // If a real pricing call costs substantially more, the remainder is
    // cuSPARSE's own host-side cost or genuine kernel execution -- and
    // which of those it is decides whether CUDA graphs are worth building.
    // So drive an actual GpuPricer over a real Netlib matrix and let its
    // own counters answer.
    for (const char* stem : {"sctap1", "fit2d", "wood1p"}) {
        MpsModel model;
        try {
            model = read_mps_file(std::string(SIHPS_DATA_ROOT) + "/netlib_lp/feasible/" + stem +
                                   ".mps");
        } catch (const std::exception& e) {
            std::printf("  (%s unavailable: %s)\n", stem, e.what());
            continue;
        }
        const LpProblem p = lp_problem_from_mps(model);
        const CSCMatrix a_csc = csr_to_csc(p.A);
        const std::int32_t m = p.n_rows(), ns = p.n_cols();
        const std::int32_t n_total = ns + 2 * m;

        GpuPricer pricer(a_csc, m, ns, m, m);
        std::vector<double> cost(static_cast<std::size_t>(n_total), 1.0);
        std::vector<double> lo(static_cast<std::size_t>(n_total), 0.0);
        std::vector<double> hi(static_cast<std::size_t>(n_total), 1e30);
        std::vector<double> art(static_cast<std::size_t>(m), 1.0);
        pricer.sync_phase(cost.data(), lo.data(), hi.data(), art.data());

        std::vector<double> y(static_cast<std::size_t>(m), 0.5);
        std::vector<double> binv(static_cast<std::size_t>(m), 0.25);
        std::vector<std::uint8_t> status(static_cast<std::size_t>(n_total), 0);

        constexpr int kIters = 400;
        for (int i = 0; i < 50; ++i) { // warm up
            pricer.price_and_select(y.data(), status.data(), true, 1e-7);
            pricer.devex_update(binv.data(), 0, 1, 1.5);
        }
        pricer.reset_profile();

        const auto t_real = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) {
            pricer.price_and_select(y.data(), status.data(), true, 1e-7);
            pricer.devex_update(binv.data(), 0, 1, 1.5);
        }
        const double wall_us = now_us(t_real) / kIters;

        const GpuPricerProfile& pr = pricer.profile();
        const double per = 1e6 / static_cast<double>(kIters);
        std::printf("%s  (m=%d, n_struct=%d, n_total=%d, nnz=%d) -- per Devex iteration:\n", stem,
                    m, ns, n_total, p.A.nnz());
        std::printf("    stage (host memcpy -> pinned) : %7.2f us\n", pr.stage_seconds * per);
        std::printf("    submit  (H2D + SpMV + kernel) : %7.2f us\n", pr.submit_seconds * per);
        std::printf("    wait    (event synchronize)   : %7.2f us\n", pr.wait_seconds * per);
        std::printf("    devex stage                   : %7.2f us\n",
                    pr.devex_stage_seconds * per);
        std::printf("    devex submit                  : %7.2f us\n",
                    pr.devex_submit_seconds * per);
        std::printf("    ------------------------------------------\n");
        std::printf("    accounted                     : %7.2f us   of %7.2f us wall\n\n",
                    pr.total_seconds() * per, wall_us);
    }

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    cudaFreeHost(h_src);
    cudaFree(d_dst);
    return 0;
}
