// CPU pricing vs GPU pricing, measured head to head on real Netlib LP
// instances through the full solve_lp pipeline.
//
// WHAT THIS EXISTS TO SETTLE
// --------------------------
// docs/research/SOTA.md's H1 ("a hybrid CPU-GPU architecture outperforms a
// CPU-only one on refinery-scale LP") and H5 ("GPU-accelerated pricing is
// faster than CPU pricing on this workload") have been UNEVIDENCED for the
// entire life of this project. The GPU code was correct -- CPU/GPU SpMV
// equivalence has been tested since Phase 1 -- but correct is not fast,
// and prompt.md is explicit that a performance claim without a measurement
// is not a claim at all. Every validated result this engine has reported
// so far ran on PricingBackend::CPU. This benchmark is what makes the
// other backend's cost a number rather than an assumption.
//
// WHY IT REPORTS PRICING TIME SEPARATELY FROM SOLVE TIME
// -----------------------------------------------------
// Pricing is the ONLY stage that differs between the two backends. The
// basis factorization, both ratio-test passes, the pivot update and
// presolve are byte-identical. A speedup quoted on end-to-end solve time
// would therefore be diluted by work the backend choice cannot affect, and
// a slowdown would be masked by it. Both are reported: pricing time
// isolates the effect, and solve time says whether it matters.
//
// The pricing timer covers reduced costs, the entering-variable search,
// and (under Devex) the pivot row and weight update -- on the GPU backend
// the first two are one fused kernel sequence with no host-visible
// boundary, so timing them together is the only way to compare like with
// like.
//
// WHAT IS BEING COMPARED, PRECISELY
// ---------------------------------
// Same algorithm, same pricing rule, same tolerances, same scaling, same
// presolve, same iteration path. The two backends compute the same reduced
// costs and apply the same tie-break, so they should follow the same
// pivots -- but cuSPARSE's summation order differs from the CPU's
// sequential dot product, so on a near-tie the two can diverge onto
// different (equally valid) paths. Iteration counts are therefore reported
// rather than asserted equal; the OBJECTIVE is asserted equal, because the
// path may differ but the optimum may not.

#include "cuda/CudaStream.hpp"
#include "cuda/PricingKernels.cuh"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

struct RunStats {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective = 0.0;
    int iterations = 0;
    double solve_seconds = 0.0;
    double pricing_seconds = 0.0;
    bool threw = false;
};

RunStats run_one(const LpProblem& p, PricingBackend backend, PricingRule rule) {
    RunStats s;
    LpSolverOptions opts;
    opts.backend = backend;
    opts.pricing_rule = rule;
    try {
        const LpSolution r = solve_lp(p, opts);
        s.status = r.status;
        s.objective = r.objective_value;
        s.iterations = r.iterations;
        s.solve_seconds = r.solve_seconds;
        s.pricing_seconds = r.pricing_seconds;
    } catch (const std::exception&) {
        s.threw = true;
    }
    return s;
}

// Per-iteration, GPU pricing costs one round trip: queue a few operations,
// then block until a 24-byte result comes back. If that round trip costs
// more than the CPU's entire pricing loop, no kernel improvement can
// rescue the backend at that size -- so the round trip is measured
// directly rather than inferred from the totals, and printed next to them.
//
// Two numbers, because they answer different questions:
//   queue-only  -- what one extra kernel in the chain costs. Measured by
//                  launching N kernels and synchronizing once, so the
//                  submission cost is what is being divided.
//   round trip  -- what ONE pricing call costs before it does any work.
//                  Launch, then immediately wait. This is the floor under
//                  every GPU pricing iteration.
void probe_launch_latency() {
    constexpr int kWarmup = 200;
    constexpr int kReps = 2000;
    CudaStream stream;

    for (int i = 0; i < kWarmup; ++i) gpu::launch_noop(stream.handle());
    stream.synchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) gpu::launch_noop(stream.handle());
    stream.synchronize();
    const double queue_us =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / kReps;

    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
        gpu::launch_noop(stream.handle());
        stream.synchronize();
    }
    const double roundtrip_us =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() * 1e6 / kReps;

    std::printf("empty-kernel launch latency on this machine:\n");
    std::printf("  queued (submission only) : %7.2f us/launch\n", queue_us);
    std::printf("  launch + synchronize     : %7.2f us/round trip\n", roundtrip_us);
    std::printf("  -> GPU pricing cannot beat CPU pricing on any model whose entire CPU\n"
                "     pricing loop costs less than %.2f us per iteration.\n\n",
                roundtrip_us);
}

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPT";
        case LpStatus::INFEASIBLE: return "INFEAS";
        case LpStatus::UNBOUNDED: return "UNBND";
        case LpStatus::ITERATION_LIMIT: return "ITERLIM";
        case LpStatus::NUMERICAL_FAILURE: return "NUMFAIL";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::int32_t max_rows = (argc > 2) ? std::atoi(argv[2]) : 600;
    const PricingRule rule =
        (argc > 3 && std::string(argv[3]) == "dantzig") ? PricingRule::DANTZIG : PricingRule::DEVEX;

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    probe_launch_latency();
    std::printf("pricing rule: %s\n\n", rule == PricingRule::DEVEX ? "DEVEX" : "DANTZIG");
    std::printf("%14s %6s %6s %8s | %8s %8s %8s | %8s %8s %8s | %8s %8s %s\n", "instance", "rows",
                "cols", "nnz", "CPU iter", "CPU pric", "CPU solv", "GPU iter", "GPU pric",
                "GPU solv", "pric x", "solv x", "obj");
    std::printf("%s\n", std::string(148, '-').c_str());

    double tot_cpu_pricing = 0.0, tot_gpu_pricing = 0.0;
    double tot_cpu_solve = 0.0, tot_gpu_solve = 0.0;
    long tot_cpu_iter = 0, tot_gpu_iter = 0;
    long tot_nnz = 0;
    int compared = 0, obj_mismatch = 0, status_mismatch = 0, skipped = 0;
    int gpu_pricing_wins = 0;

    for (const auto& path : files) {
        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) continue;

        const LpProblem p = lp_problem_from_mps(model);
        const std::int32_t nnz = p.A.nnz();

        const RunStats cpu = run_one(p, PricingBackend::CPU, rule);
        const RunStats gpu = run_one(p, PricingBackend::GPU, rule);

        if (cpu.threw || gpu.threw) {
            ++skipped;
            std::printf("%14s %6d %6d %8d |  (threw on %s backend -- excluded)\n",
                        path.stem().string().c_str(), model.n_rows, model.n_cols, nnz,
                        cpu.threw ? "CPU" : "GPU");
            continue;
        }

        const char* obj_note = "match";
        if (cpu.status != gpu.status) {
            obj_note = "STATUS DIFF";
            ++status_mismatch;
        } else if (cpu.status == LpStatus::OPTIMAL) {
            const double rel =
                std::fabs(cpu.objective - gpu.objective) / (1.0 + std::fabs(cpu.objective));
            if (rel >= 1e-6) {
                obj_note = "MISMATCH";
                ++obj_mismatch;
            }
            ++compared;
            tot_cpu_pricing += cpu.pricing_seconds;
            tot_gpu_pricing += gpu.pricing_seconds;
            tot_cpu_solve += cpu.solve_seconds;
            tot_gpu_solve += gpu.solve_seconds;
            tot_cpu_iter += cpu.iterations;
            tot_gpu_iter += gpu.iterations;
            tot_nnz += nnz;
            if (gpu.pricing_seconds < cpu.pricing_seconds) ++gpu_pricing_wins;
        }

        const double pric_x =
            (gpu.pricing_seconds > 1e-9) ? cpu.pricing_seconds / gpu.pricing_seconds : 0.0;
        const double solv_x =
            (gpu.solve_seconds > 1e-9) ? cpu.solve_seconds / gpu.solve_seconds : 0.0;

        std::printf("%14s %6d %6d %8d | %8d %8.3f %8.3f | %8d %8.3f %8.3f | %7.2fx %7.2fx %s%s%s\n",
                    path.stem().string().c_str(), model.n_rows, model.n_cols, nnz, cpu.iterations,
                    cpu.pricing_seconds, cpu.solve_seconds, gpu.iterations, gpu.pricing_seconds,
                    gpu.solve_seconds, pric_x, solv_x, obj_note,
                    cpu.status == gpu.status ? "" : " ", cpu.status == gpu.status
                        ? ""
                        : (std::string(status_str(cpu.status)) + "/" + status_str(gpu.status))
                              .c_str());
    }

    std::printf("\n%s\n", std::string(148, '=').c_str());
    std::printf("Instances compared (both OPTIMAL): %d   excluded (threw): %d\n", compared,
                skipped);
    std::printf("Total nnz across compared instances: %ld\n", tot_nnz);
    std::printf("Iterations   CPU %ld   GPU %ld\n", tot_cpu_iter, tot_gpu_iter);
    std::printf("PRICING time CPU %.3f s   GPU %.3f s", tot_cpu_pricing, tot_gpu_pricing);
    if (tot_gpu_pricing > 1e-9) {
        std::printf("   (GPU %.2fx the CPU's pricing time)", tot_gpu_pricing / tot_cpu_pricing);
    }
    std::printf("\nSOLVE   time CPU %.3f s   GPU %.3f s", tot_cpu_solve, tot_gpu_solve);
    if (tot_gpu_solve > 1e-9) {
        std::printf("   (GPU %.2fx the CPU's solve time)", tot_gpu_solve / tot_cpu_solve);
    }
    if (tot_cpu_iter > 0) {
        std::printf("\nPer-iteration pricing  CPU %.1f us   GPU %.1f us",
                    tot_cpu_pricing * 1e6 / static_cast<double>(tot_cpu_iter),
                    tot_gpu_pricing * 1e6 / static_cast<double>(tot_gpu_iter));
    }
    std::printf("\nInstances where GPU pricing was faster: %d / %d\n", gpu_pricing_wins,
                compared);
    std::printf("Objective mismatches: %d   status mismatches: %d   (both must be 0 -- the "
                "backend changes where pricing runs, not what the optimum is)\n",
                obj_mismatch, status_mismatch);

    return (obj_mismatch == 0 && status_mismatch == 0) ? 0 : 1;
}
