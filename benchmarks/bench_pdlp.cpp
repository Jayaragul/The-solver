// GPU PDLP against the CPU simplex, on real Netlib LP instances.
//
// WHAT IS BEING TESTED
// --------------------
// docs/architecture/CPU_GPU.md 4 measured GPU pricing losing to CPU pricing
// by 3-5x and identified the cause exactly: a simplex iteration cannot
// proceed until the host knows which column entered, so every iteration
// pays a device synchronize, and an SpMV behind a sync costs 33-52 us
// against 15-26 us queued (benchmarks/bench_spmv_algorithm.cpp). That is a
// property of the ALGORITHM, so the response is a different algorithm
// rather than better kernels.
//
// PDHG needs no host decision per iteration, so an entire restart window is
// queued and synchronized once. This benchmark checks whether that
// structural argument actually shows up in wall-clock time, and at what
// accuracy -- because a first-order method that is fast and wrong is worth
// nothing.
//
// HOW IT IS JUDGED
// ----------------
// Against the PUBLISHED Netlib optimum, not against our own simplex: two
// implementations agreeing tells you they share a bug just as easily as it
// tells you they are right. Reported per instance:
//
//   - objective error vs the published reference (relative)
//   - the verified KKT triple PDLP terminated on
//   - host synchronizations (the quantity the design exists to minimize)
//   - wall clock against the CPU simplex through the same pipeline
//
// A first-order method converges quickly to moderate accuracy and slowly to
// high accuracy. Both regimes are therefore run -- 1e-4 and 1e-8 -- because
// quoting only the loose one would misrepresent what the method costs to be
// trustworthy, and quoting only the tight one would hide where it is
// genuinely fast.

#include "cuda/CudaDevice.hpp"
#include "cuda/GpuPdlp.hpp"
#include "io/MpsReader.hpp"
#include "io/NetlibReference.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "lp/Scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

struct PdlpRun {
    PdlpStats stats;
    double objective = 0.0; // in ORIGINAL units
    bool threw = false;
};

// Runs PDLP on the Ruiz-scaled problem and unscales the objective back.
//
// Scaling is not optional for a first-order method: its convergence rate
// depends on the conditioning of A, and an unscaled refinery-style model
// can be many orders of magnitude out of balance. The same Ruiz
// equilibration the simplex uses (docs/architecture/NUMERICS.md 2) is
// applied here, so the two solvers see equally conditioned data and the
// comparison stays honest.
PdlpRun run_pdlp(const LpProblem& p, double eps, std::int32_t restart_period,
                 double time_limit, bool adaptive) {
    PdlpRun run;
    try {
        // Ruiz first, then Pock-Chambolle on what Ruiz produced. The two
        // target different quantities -- see Scaling.hpp -- and PDLP wants
        // both. Composed into a single (R, C) so unscaling happens once.
        const ScaleFactors ruiz = compute_ruiz_scaling(p.A);
        const CSRMatrix a_ruiz = apply_ruiz_scaling(p.A, ruiz);
        const ScaleFactors pc = compute_pock_chambolle_scaling(a_ruiz);
        const ScaleFactors scale = compose_scaling(ruiz, pc);
        const CSRMatrix a_scaled = apply_ruiz_scaling(p.A, scale);

        const auto n = static_cast<std::size_t>(p.n_cols());
        const auto m = static_cast<std::size_t>(p.n_rows());

        // x = C x', so bounds divide by the column scale and costs multiply
        // by it; the objective value itself is invariant.
        std::vector<double> cost(n), lower(n), upper(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double c = scale.col_scale[j];
            cost[j] = p.obj[j] * c;
            lower[j] = p.lower[j] / c;
            upper[j] = p.upper[j] / c;
        }

        // A x + s = rhs with slack_lower <= s <= slack_upper is exactly
        // rhs - slack_upper <= A x <= rhs - slack_lower.
        std::vector<double> row_lower(m), row_upper(m);
        for (std::size_t i = 0; i < m; ++i) {
            const double r = scale.row_scale[i];
            row_lower[i] = r * (p.rhs[i] - p.slack_upper[i]);
            row_upper[i] = r * (p.rhs[i] - p.slack_lower[i]);
        }

        GpuPdlp pdlp(a_scaled, cost, lower, upper, row_lower, row_upper);
        PdlpParams params;
        params.eps_optimal = eps;
        params.restart_period = restart_period;
        params.time_limit_seconds = time_limit;
        params.adaptive_step = adaptive;

        std::vector<double> x_scaled, y_scaled;
        run.stats = pdlp.solve(params, x_scaled, y_scaled);

        double obj = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            obj += p.obj[j] * (x_scaled[j] * scale.col_scale[j]);
        }
        run.objective = obj;
    } catch (const std::exception&) {
        run.threw = true;
    }
    return run;
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::int32_t max_rows = (argc > 2) ? std::atoi(argv[2]) : 600;
    const double eps = (argc > 3) ? std::atof(argv[3]) : 1e-8;
    const std::int32_t restart_period = (argc > 4) ? std::atoi(argv[4]) : 64;
    const double time_limit = (argc > 5) ? std::atof(argv[5]) : 20.0;
    // Lower bound on size, so a sweep can target the large end without
    // spending its time limit on models the CPU solves in microseconds.
    const std::int32_t min_rows = (argc > 6) ? std::atoi(argv[6]) : 0;
    // Above this many rows the CPU simplex comparison is skipped. The
    // simplex takes no time limit, and on a 100k-row model it would run for
    // hours before this benchmark could print anything. A skipped CPU run
    // is reported as "skipped" and excluded from the totals -- NOT recorded
    // as an infinite CPU time, which would hand PDLP a free win on exactly
    // the instances the comparison is supposed to be about.
    const std::int32_t cpu_row_cap = (argc > 7) ? std::atoi(argv[7]) : 3000;
    // Adaptive step size on/off, so the A/B that justifies it can be run
    // from the command line rather than by editing a default and rebuilding.
    const bool adaptive = (argc > 8) ? (std::atoi(argv[8]) != 0) : true;

    const DeviceInfo info = CudaDevice::select(0);
    std::printf("GPU: %s (CC %d.%d, %d SMs)\n", info.name.c_str(), info.compute_capability_major,
                info.compute_capability_minor, info.multiprocessor_count);
    std::printf("eps = %g   restart period = %d   time limit = %.1f s   adaptive step = %s\n\n",
                eps, restart_period, time_limit, adaptive ? "on" : "off");

    const NetlibReference refs = NetlibReference::load("data/netlib_readme.txt");

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%12s %6s %6s %8s | %9s %7s %6s %9s | %10s %9s | %8s %s\n", "instance", "rows",
                "cols", "nnz", "PDLP iter", "syncs", "rstrt", "PDLP sec", "KKT worst", "obj err",
                "CPU sec", "verdict");
    std::printf("%s\n", std::string(132, '-').c_str());

    int solved = 0, attempted = 0, accurate = 0;
    double total_pdlp = 0.0, total_cpu = 0.0, total_pdlp_vs_cpu = 0.0;
    long total_iters = 0, total_syncs = 0;

    for (const auto& path : files) {
        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) continue;
        if (model.n_rows < min_rows) continue;

        const std::string stem = path.stem().string();
        const std::vector<ReferenceValue>* reference = refs.find(stem);
        if (reference == nullptr || reference->empty()) {
            continue; // no published optimum: nothing to judge against
        }

        const LpProblem p = lp_problem_from_mps(model);
        ++attempted;

        const PdlpRun pdlp = run_pdlp(p, eps, restart_period, time_limit, adaptive);

        // CPU simplex through the shipping pipeline, for the wall-clock
        // comparison. Same machine, same model, same moment.
        double cpu_sec = -1.0;
        if (model.n_rows <= cpu_row_cap) {
            const auto t_cpu = std::chrono::steady_clock::now();
            try {
                (void)solve_lp(p, LpSolverOptions{});
            } catch (const std::exception&) {
            }
            cpu_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cpu).count();
        }

        if (pdlp.threw) {
            std::printf("%12s %6d %6d %8d |  (PDLP threw)\n", stem.c_str(), model.n_rows,
                        model.n_cols, p.A.nnz());
            continue;
        }

        // The readme lists several published values per instance (summary,
        // CPLEX, MINOS) which disagree in their last digits. Agreeing with
        // ANY of them is agreement -- scoring against one arbitrarily
        // chosen column would fail a correct solver for matching a
        // different reference implementation.
        double obj_err = std::numeric_limits<double>::infinity();
        for (const ReferenceValue& rv : *reference) {
            obj_err = std::min(obj_err,
                                std::fabs(pdlp.objective - rv.value) / (1.0 + std::fabs(rv.value)));
        }
        const double kkt = std::max({pdlp.stats.relative_primal_residual,
                                      pdlp.stats.relative_dual_residual, pdlp.stats.relative_gap});

        const char* verdict = "iter/time limit";
        if (pdlp.stats.converged) {
            ++solved;
            // Converging on the KKT triple and MATCHING THE PUBLISHED
            // OPTIMUM are different claims. Both are required before this
            // reports success, because a first-order method can satisfy a
            // relative KKT test on a badly scaled model and still be wrong
            // about the objective.
            if (obj_err < 1e-6) {
                verdict = "OPTIMAL";
                ++accurate;
            } else {
                verdict = "CONVERGED BUT OFF";
            }
            total_pdlp += pdlp.stats.seconds;
            if (cpu_sec >= 0.0) {
                total_cpu += cpu_sec;
                total_pdlp_vs_cpu += pdlp.stats.seconds;
            }
            total_iters += pdlp.stats.iterations;
            total_syncs += pdlp.stats.host_syncs;
        }

        char cpu_cell[16];
        if (cpu_sec >= 0.0) {
            std::snprintf(cpu_cell, sizeof(cpu_cell), "%8.3f", cpu_sec);
        } else {
            std::snprintf(cpu_cell, sizeof(cpu_cell), "%8s", "skipped");
        }
        std::printf("%12s %6d %6d %8d | %9d %7d %6d %9.3f | %10.2e %9.2e | %s %s\n",
                    stem.c_str(), model.n_rows, model.n_cols, p.A.nnz(), pdlp.stats.iterations,
                    pdlp.stats.host_syncs, pdlp.stats.restarts, pdlp.stats.seconds, kkt, obj_err,
                    cpu_cell, verdict);
        std::fflush(stdout);
    }

    std::printf("\n%s\n", std::string(132, '=').c_str());
    std::printf("attempted %d   converged %d   converged AND within 1e-6 of the published "
                "optimum %d\n",
                attempted, solved, accurate);
    if (solved > 0) {
        std::printf("on converged instances: PDLP %.3f s total\n", total_pdlp);
        std::printf("where the CPU simplex also ran: PDLP %.3f s   CPU %.3f s",
                    total_pdlp_vs_cpu, total_cpu);
        if (total_pdlp_vs_cpu > 1e-9) {
            std::printf("   (PDLP %.2fx)", total_cpu / total_pdlp_vs_cpu);
        }
        std::printf("\niterations %ld across %ld host syncs -- %.0f iterations per sync\n",
                    total_iters, total_syncs,
                    total_syncs > 0 ? static_cast<double>(total_iters) / total_syncs : 0.0);
    }
    return 0;
}
