// Primal two-phase vs dual simplex, measured head to head on real Netlib
// LP instances -- the evidence behind LP.md §2's cold-start selection rule.
//
// This exists for the same reason bench_pricing_rule.cpp does: prompt.md
// forbids asserting an algorithmic win without measuring it. The
// literature's case for dual simplex as a solver default (SOTA.md §1.1,
// §1.2) is a case about WARM STARTS -- re-solving a branch-and-bound child
// from its parent's basis, where only primal feasibility was disturbed.
// Every solve here is a COLD start, so this benchmark measures the one
// situation that argument does not cover, and its result is what LP.md §2
// cites for resolving AUTO to the primal path today.
//
// Both sides run the FULL pipeline (solve_lp: presolve -> scale -> simplex
// -> postsolve -> verify), not Simplex in isolation. That matters and is
// not a detail of convenience: whether a model admits a dual-feasible
// all-slack start is a property of the model the simplex actually receives,
// and presolve changes it -- pilot87 has no dual-feasible start unreduced
// but does once presolve has fixed columns and tightened bounds. Measuring
// the bare Simplex would therefore classify instances differently from the
// configuration that actually ships.
//
// Classification comes from the engine's own report rather than a
// duplicate of its test:
//   dual_iterations == 0        -> no dual-feasible start; the DUAL column
//                                  IS the primal path, so the row is
//                                  excluded from the totals (comparing the
//                                  primal path against itself measures
//                                  nothing).
//   used_dual_simplex           -> the dual path's result was verified and
//                                  reported.
//   iterations > 0, !used_dual  -> the dual path ran, its result failed
//                                  verification, and the primal fallback
//                                  produced the answer. Its cost is real
//                                  and stays in the reported time.
//
// Objective agreement is asserted between the two: the algorithm changes
// the path to the optimum, never the optimum.

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
    int dual_iterations = 0;
    bool used_dual = false;
    double seconds = 0.0;
};

RunStats run_one(const LpProblem& p, LpAlgorithm algorithm) {
    RunStats s;
    LpSolverOptions opts;
    opts.algorithm = algorithm;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        LpSolution r = solve_lp(p, opts);
        s.status = r.status;
        s.objective = r.objective_value;
        s.iterations = r.iterations;
        s.dual_iterations = r.dual_iterations;
        s.used_dual = r.used_dual_simplex;
    } catch (const std::exception&) {
        s.status = LpStatus::NUMERICAL_FAILURE;
    }
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return s;
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

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%14s %6s %6s | %9s %9s %8s | %9s %9s %8s %10s | %8s %8s %s\n", "instance", "rows",
                "cols", "PRI iter", "PRI sec", "PRI st", "DUAL iter", "DUAL sec", "DUAL st",
                "dual path", "iter x", "time x", "obj");
    std::printf("%s\n", std::string(152, '-').c_str());

    long total_pri_iter = 0, total_dual_iter = 0;
    double total_pri_sec = 0.0, total_dual_sec = 0.0;
    int compared = 0, obj_mismatch = 0, no_dual_start = 0, fell_back = 0;

    for (const auto& path : files) {
        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) continue;

        const LpProblem p = lp_problem_from_mps(model);
        const RunStats pri = run_one(p, LpAlgorithm::PRIMAL);
        const RunStats dual = run_one(p, LpAlgorithm::DUAL);

        const bool entered_dual = dual.dual_iterations > 0;
        const char* path_note = !entered_dual ? "no-start" : (dual.used_dual ? "dual" : "fell-back");
        if (!entered_dual) ++no_dual_start;
        else if (!dual.used_dual) ++fell_back;

        bool obj_ok = true;
        if (entered_dual && pri.status == LpStatus::OPTIMAL && dual.status == LpStatus::OPTIMAL) {
            ++compared;
            total_pri_iter += pri.iterations;
            total_dual_iter += dual.iterations;
            total_pri_sec += pri.seconds;
            total_dual_sec += dual.seconds;
            const double rel =
                std::fabs(pri.objective - dual.objective) / (1.0 + std::fabs(pri.objective));
            obj_ok = rel < 1e-6;
            if (!obj_ok) ++obj_mismatch;
        }

        const double iter_x =
            (dual.iterations > 0) ? static_cast<double>(pri.iterations) / dual.iterations : 0.0;
        const double time_x = (dual.seconds > 1e-9) ? pri.seconds / dual.seconds : 0.0;

        std::printf("%14s %6d %6d | %9d %9.3f %8s | %9d %9.3f %8s %10s | %7.2fx %7.2fx %s\n",
                    path.stem().string().c_str(), model.n_rows, model.n_cols, pri.iterations,
                    pri.seconds, status_str(pri.status), dual.iterations, dual.seconds,
                    status_str(dual.status), path_note, iter_x, time_x,
                    obj_ok ? "match" : "MISMATCH");
    }

    std::printf("\n%s\n", std::string(152, '=').c_str());
    std::printf("Instances compared (dual path entered AND both OPTIMAL): %d\n", compared);
    std::printf("Excluded: %d with no dual-feasible start", no_dual_start);
    std::printf(", %d where the dual path ran but its result failed verification and the "
                "primal fallback produced the answer\n",
                fell_back);
    std::printf("Total iterations  primal %ld  vs  dual %ld", total_pri_iter, total_dual_iter);
    if (total_pri_iter > 0) {
        std::printf("   (dual %.2fx)",
                     static_cast<double>(total_dual_iter) / static_cast<double>(total_pri_iter));
    }
    std::printf("\nTotal wall-clock  primal %.3f s  vs  dual %.3f s", total_pri_sec,
                total_dual_sec);
    if (total_pri_sec > 1e-9) {
        std::printf("   (dual %.2fx)", total_dual_sec / total_pri_sec);
    }
    std::printf("\nObjective mismatches between algorithms: %d (must be 0 -- the algorithm "
                "changes the path, not the optimum)\n",
                obj_mismatch);

    // A cold-start comparison only. It says nothing about the warm-start
    // case dual simplex is actually chosen for in LP.md §2, because no
    // warm-start entry point exists yet to measure.
    return obj_mismatch == 0 ? 0 : 1;
}
