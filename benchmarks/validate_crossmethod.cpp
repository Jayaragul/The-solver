// Cross-method validation for instances with no published reference value.
//
// WHY THIS EXISTS
// ---------------
// `validate_netlib` scores an objective against a published optimum. 21 of
// the 114 models in this repository have no such value in
// `data/netlib_readme.txt` -- the Kennington families (cre-*, ken-*, osa-*,
// pds-*), the QAP relaxations, `standgub` and `truss` -- so the validator
// skips them BEFORE solving. They have therefore never been attempted, and
// they are among the largest and most industrially structured models
// available here. `ken-18` alone is roughly six times larger than anything
// in the validated set.
//
// Skipping the hardest models because they are hard to score is exactly the
// kind of silent coverage gap CLAUDE_OPUS_SOLVER_ROADMAP.md Phase 1 exists
// to close, and its acceptance criteria ask for an independent objective
// checker rather than only a lookup table.
//
// WHAT THIS CHECKS, AND WHAT IT DOES NOT
// --------------------------------------
// Two solvers that share no code path beyond parsing and presolve:
//
//     SIMPLEX      -- exact vertex, CPU, basis factorization
//     FIRST_ORDER  -- approximate interior-ish point, GPU, PDHG
//
// Agreement between them is genuine evidence: they would have to be wrong
// in the same direction by the same amount to agree spuriously, and they
// share no arithmetic that could make that likely. Both results also pass
// the same original-space residual gate (NUMERICS.md 6) independently.
//
// This is nonetheless WEAKER than a published optimum, and is reported as
// AGREE rather than PASS to keep the distinction visible. Agreement shows
// the two methods found the same point; it cannot prove that point optimal.
// A shared presolve bug would deceive both, which is why `--nopresolve` is
// worth running alongside.
//
// Budgets are mandatory here rather than optional: these models are large
// enough that an ungoverned simplex can run for hours, and a benchmark that
// cannot finish produces no information at all.

#include "bench/RunMetadata.hpp"
#include "io/MpsReader.hpp"
#include "io/NetlibReference.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sihps;

namespace {

// Relative objective agreement required between the two methods. Looser
// than the 1e-6 used against published optima because a first-order method
// is being compared against an exact vertex, and PDLP is stopped at a
// finite tolerance by design.
constexpr double kAgreeTol = 1e-5;

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

struct Outcome {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective = 0.0;
    double seconds = 0.0;
    std::int32_t iterations = 0;
};

Outcome solve_with(const LpProblem& p, LpMethod method, bool presolve, double simplex_budget,
                   double pdlp_eps, double pdlp_time_limit) {
    Outcome o;
    LpSolverOptions opts;
    opts.method = method;
    opts.use_presolve = presolve;
    opts.hybrid_simplex_budget_seconds = simplex_budget;
    opts.simplex_time_budget_seconds = simplex_budget;
    opts.pdlp.eps_optimal = pdlp_eps;
    opts.pdlp.time_limit_seconds = pdlp_time_limit;

    const auto t0 = std::chrono::steady_clock::now();
    try {
        const LpSolution s = solve_lp(p, opts);
        o.status = s.status;
        o.objective = s.objective_value;
        o.iterations = s.iterations;
    } catch (const std::exception&) {
        o.status = LpStatus::NUMERICAL_FAILURE;
    }
    o.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return o;
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::string readme = (argc > 2) ? argv[2] : "data/netlib_readme.txt";
    // Wall-clock budget for EACH method on EACH instance.
    const double budget = (argc > 3) ? std::atof(argv[3]) : 120.0;
    const bool presolve = !(argc > 4 && std::string(argv[4]) == "nopresolve");
    const std::string jsonl_path = (argc > 5) ? argv[5] : "";

    const bench::RunMetadata meta = bench::RunMetadata::capture();
    bench::RunConfig cfg;
    cfg.method = "CROSS(SIMPLEX,FIRST_ORDER)";
    cfg.pricing_rule = "DEVEX";
    cfg.algorithm = "AUTO";
    cfg.presolve = presolve;
    cfg.hybrid_simplex_budget_seconds = budget;
    cfg.hybrid_first_order_eps = 1e-8;

    std::printf("build %s (%s)  %s  CUDA %s  %s\n", meta.git_commit.c_str(),
                meta.git_dirty.c_str(), meta.compiler.c_str(), meta.cuda_version.c_str(),
                meta.gpu_name.c_str());
    std::printf("threads=%d  presolve=%s  budget=%.0f s per method  agree tol=%g\n\n",
                meta.thread_count, presolve ? "on" : "off", budget, kAgreeTol);

    const NetlibReference refs = NetlibReference::load(readme);

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".mps") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%14s %7s %7s %9s | %11s %9s %9s | %11s %9s %9s | %10s %s\n", "instance", "rows",
                "cols", "nnz", "simplex", "obj", "sec", "first-order", "obj", "sec", "rel diff",
                "verdict");
    std::printf("%s\n", std::string(150, '-').c_str());

    int agree = 0, disagree = 0, only_simplex = 0, only_pdlp = 0, neither = 0;
    double oracle_total = 0.0, hybrid_total = 0.0, simplex_total = 0.0, first_order_total = 0.0;
    int hybrid_solved = 0;
    std::vector<std::string> problems;
    std::vector<bench::InstanceRecord> records;

    for (const auto& path : files) {
        const std::string name = path.stem().string();
        // Only the instances validate_netlib cannot score.
        const auto* r = refs.find(name);
        if (r && !r->empty()) continue;

        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0) continue;

        const LpProblem p = lp_problem_from_mps(model);

        const Outcome s = solve_with(p, LpMethod::SIMPLEX, presolve, budget, 1e-8, budget);
        const Outcome f = solve_with(p, LpMethod::FIRST_ORDER, presolve, budget, 1e-8, budget);
        // The engine's own choice, measured on the same instance in the
        // same process. Reported against the ORACLE (the better of the two
        // above) because "is the hybrid fast" is a less useful question
        // than "how much does the hybrid's choice cost against a perfect
        // one".
        const Outcome h = solve_with(p, LpMethod::HYBRID, presolve, budget, 1e-8, budget);

        const bool s_ok = s.status == LpStatus::OPTIMAL;
        const bool f_ok = f.status == LpStatus::OPTIMAL;

        double rel = std::numeric_limits<double>::quiet_NaN();
        const char* verdict = "?";
        if (s_ok && f_ok) {
            rel = std::fabs(s.objective - f.objective) / (1.0 + std::fabs(s.objective));
            if (rel < kAgreeTol) {
                verdict = "AGREE";
                ++agree;
            } else {
                verdict = "DISAGREE";
                ++disagree;
                problems.push_back(name + "(disagree)");
            }
        } else if (s_ok) {
            verdict = "simplex only";
            ++only_simplex;
        } else if (f_ok) {
            verdict = "first-order only";
            ++only_pdlp;
        } else {
            verdict = "NEITHER";
            ++neither;
            problems.push_back(name + "(neither)");
        }

        // The oracle can only be formed where at least one method
        // succeeded; a failed method contributes no time to beat.
        const double best = (s_ok && f_ok) ? std::min(s.seconds, f.seconds)
                                            : (s_ok ? s.seconds : (f_ok ? f.seconds : 0.0));
        if (s_ok || f_ok) {
            oracle_total += best;
            simplex_total += s_ok ? s.seconds : budget;
            first_order_total += f_ok ? f.seconds : budget;
        }
        if (h.status == LpStatus::OPTIMAL) {
            ++hybrid_solved;
            hybrid_total += h.seconds;
        } else if (s_ok || f_ok) {
            hybrid_total += h.seconds;
        }

        std::printf("%14s %7d %7d %9d | %11s %9.4g %9.2f | %11s %9.4g %9.2f | %10.2e %-16s | "
                    "hybrid %8.2f %s\n",
                    name.c_str(), model.n_rows, model.n_cols, p.A.nnz(), status_str(s.status),
                    s.objective, s.seconds, status_str(f.status), f.objective, f.seconds, rel,
                    verdict, h.seconds, status_str(h.status));
        std::fflush(stdout);

        bench::InstanceRecord rec;
        rec.instance_path = path.string();
        rec.instance_hash = bench::hash_file(path.string());
        rec.rows = model.n_rows;
        rec.cols = model.n_cols;
        rec.nnz = p.A.nnz();
        rec.status = status_str(s.status);
        rec.objective = s.objective;
        rec.reference_objective = f.objective; // the other method IS the reference here
        rec.relative_objective_error = rel;
        rec.passed = (verdict[0] == 'A');
        rec.reference_source = "cross-method";
        rec.wall_seconds = s.seconds + f.seconds;
        rec.simplex_seconds = s.seconds;
        rec.first_order_seconds = f.seconds;
        rec.status = std::string(status_str(s.status)) + "/" + status_str(f.status);
        rec.iterations = s.iterations;
        rec.pdlp_iterations = f.iterations;
        rec.used_first_order = f_ok;
        records.push_back(rec);
    }

    std::printf("%s\n", std::string(150, '=').c_str());
    std::printf("AGREE %d   DISAGREE %d   simplex-only %d   first-order-only %d   neither %d\n",
                agree, disagree, only_simplex, only_pdlp, neither);
    std::printf("\nsolved: hybrid %d   (of %d instances where either method succeeded)\n",
                hybrid_solved, agree + disagree + only_simplex + only_pdlp);
    std::printf("total time   simplex %.1f s   first-order %.1f s   HYBRID %.1f s   "
                "oracle %.1f s\n",
                simplex_total, first_order_total, hybrid_total, oracle_total);
    if (oracle_total > 1e-9) {
        std::printf("hybrid / oracle = %.2fx\n", hybrid_total / oracle_total);
        std::printf(
            "  NOT like-for-like, and the direction of the bias is known: the two columns\n"
            "  above solve at the eps passed to this benchmark, while HYBRID runs its\n"
            "  first-order path at LpSolverOptions::hybrid_first_order_eps, which is looser.\n"
            "  A ratio below 1.00 therefore does NOT mean the hybrid beat a perfect chooser --\n"
            "  it means it was allowed a cheaper stopping test. Both still clear the same\n"
            "  original-space gate. Read this as an upper bound on how good the method CHOICE\n"
            "  is, not as a speedup.\n");
    }
    if (!problems.empty()) {
        std::printf("NEEDS ATTENTION:");
        for (const auto& q : problems) std::printf(" %s", q.c_str());
        std::printf("\n");
    }
    std::printf("\nNOTE: AGREE is weaker evidence than a published optimum. It shows two\n"
                "independent methods reached the same point and both cleared the original-space\n"
                "residual gate; it does not prove that point optimal.\n");

    if (!jsonl_path.empty()) {
        bench::JsonlWriter w(jsonl_path, meta, cfg);
        if (w.ok()) {
            for (const auto& rec : records) w.write(rec);
            std::printf("records written to %s\n", jsonl_path.c_str());
        }
    }
    // DISAGREE and NEITHER are real findings, not failures of this tool, so
    // the exit code reports only whether the sweep could run at all.
    return 0;
}
