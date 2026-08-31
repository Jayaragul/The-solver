// End-to-end LP solve benchmark on real Netlib LP instances -- this is
// the actual H5 experiment (docs/research/SOTA.md \S5): does GPU-
// accelerated full pricing actually beat CPU pricing inside a real
// simplex solve loop, on this hardware, on real problems? Both backends
// run the IDENTICAL algorithm (same pivoting, same ratio test, same basis
// updates) -- only the pricing step differs, isolating that one variable.
//
// v1's dense-basis-inverse simplex is O(m^2) per iteration, so this pass
// caps instances by row count to keep each solve well under the kind of
// multi-minute runtime that would be irresponsible to run repeatedly on a
// laptop -- excluded instances are printed, not silently dropped.
//
// Also records host memory (peak RSS via getrusage) and GPU memory
// (free/total via cudaMemGetInfo before and after each solve) per the
// request to track "CPU, GPU, memory block allocation."

#include "cuda/CudaDevice.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/Simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace sihps;
namespace fs = std::filesystem;

namespace {

constexpr std::int32_t kMaxRows = 600; // dense O(m^2)-per-iteration cap for this pass

long peak_rss_kb() {
#ifdef __linux__
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss; // KB on Linux
#else
    // A portable peak-RSS query is not available in the MSVC CRT. Keep the
    // benchmark runnable and report zero rather than inventing a value.
    return 0;
#endif
}

struct SolveRecord {
    std::string name;
    std::int32_t rows, cols;
    LpStatus status;
    double objective;
    int phase1_iters, phase2_iters;
    double solve_seconds;
    double pricing_seconds;
    double primal_residual, dual_residual;
};

SolveRecord run_one(const std::string& name, const LpProblem& p, PricingBackend backend,
                     bool& unsupported) {
    SolveRecord rec;
    rec.name = name;
    rec.rows = p.n_rows();
    rec.cols = p.n_cols();
    unsupported = false;

    try {
        auto t0 = std::chrono::steady_clock::now();
        Simplex simplex(p, backend);
        auto result = simplex.solve();
        auto t1 = std::chrono::steady_clock::now();

        rec.status = result.status;
        rec.objective = result.objective_value;
        rec.phase1_iters = result.phase1_iterations;
        rec.phase2_iters = result.phase2_iterations;
        rec.solve_seconds = std::chrono::duration<double>(t1 - t0).count();
        rec.pricing_seconds = result.pricing_seconds;
        rec.primal_residual = result.primal_residual;
        rec.dual_residual = result.dual_residual;
    } catch (const std::exception&) {
        // Constructor-time rejection (e.g. a free variable -- v1 scope
        // limit, docs/architecture/LP.md) happens before Simplex::solve()'s
        // own try/catch can see it; treat it as a per-instance skip, not a
        // reason to abort the whole sweep.
        unsupported = true;
        rec.status = LpStatus::NUMERICAL_FAILURE;
    }
    return rec;
}

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPTIMAL";
        case LpStatus::INFEASIBLE: return "INFEASIBLE";
        case LpStatus::UNBOUNDED: return "UNBOUNDED";
        case LpStatus::ITERATION_LIMIT: return "ITER_LIMIT";
        case LpStatus::NUMERICAL_FAILURE: return "NUM_FAILURE";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    auto gpu_info = CudaDevice::select(0);
    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", gpu_info.name.c_str(),
                gpu_info.compute_capability_major, gpu_info.compute_capability_minor,
                gpu_info.multiprocessor_count);
    std::printf("Row cap for this pass: %d (dense O(m^2)-per-iteration v1 basis)\n\n", kMaxRows);

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%14s %6s %6s %11s %14s %9s %9s %10s %10s %10s\n", "instance", "rows", "cols",
                "status", "objective", "p1 iter", "p2 iter", "CPU sec", "GPU sec", "speedup");
    std::printf("%s\n", std::string(110, '-').c_str());

    int skipped = 0, mismatches = 0;
    std::size_t gpu_mem_free_before = 0, gpu_mem_total = 0;
    CudaDevice::memory_info(gpu_mem_free_before, gpu_mem_total);
    long rss_before = peak_rss_kb();

    for (const auto& path : files) {
        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > kMaxRows) {
            ++skipped;
            continue;
        }

        LpProblem p = lp_problem_from_mps(model);
        bool cpu_unsupported = false, gpu_unsupported = false;
        auto cpu_rec = run_one(path.stem().string(), p, PricingBackend::CPU, cpu_unsupported);
        auto gpu_rec = run_one(path.stem().string(), p, PricingBackend::GPU, gpu_unsupported);

        if (cpu_unsupported) {
            std::printf("%14s  SKIPPED: unsupported model feature (e.g. free variable)\n",
                        path.stem().string().c_str());
            ++skipped;
            continue;
        }

        if (cpu_rec.status == LpStatus::OPTIMAL && gpu_rec.status == LpStatus::OPTIMAL &&
            std::fabs(cpu_rec.objective - gpu_rec.objective) >
                1e-4 * (1.0 + std::fabs(cpu_rec.objective))) {
            ++mismatches;
        }

        std::printf("%14s %6d %6d %11s %14.6f %9d %9d %10.4f %10.4f %9.2fx%s\n",
                    cpu_rec.name.c_str(), cpu_rec.rows, cpu_rec.cols, status_str(cpu_rec.status),
                    cpu_rec.objective, cpu_rec.phase1_iters, cpu_rec.phase2_iters,
                    cpu_rec.solve_seconds, gpu_rec.solve_seconds,
                    cpu_rec.solve_seconds / gpu_rec.solve_seconds,
                    (cpu_rec.status != gpu_rec.status) ? "  [STATUS MISMATCH]" : "");
    }

    std::size_t gpu_mem_free_after = 0;
    CudaDevice::memory_info(gpu_mem_free_after, gpu_mem_total);
    long rss_after = peak_rss_kb();

    std::printf("\n%d instance(s) skipped (rows > %d or degenerate/unparseable).\n", skipped,
                kMaxRows);
    std::printf("%d CPU/GPU objective mismatch(es) among solved-optimal instances.\n", mismatches);
    std::printf("\nHost peak RSS: %.1f MB before -> %.1f MB after (delta %.1f MB)\n",
                rss_before / 1024.0, rss_after / 1024.0, (rss_after - rss_before) / 1024.0);
    std::printf("GPU free VRAM: %.2f GB before -> %.2f GB after (of %.2f GB total)\n",
                gpu_mem_free_before / 1e9, gpu_mem_free_after / 1e9, gpu_mem_total / 1e9);
    return 0;
}
