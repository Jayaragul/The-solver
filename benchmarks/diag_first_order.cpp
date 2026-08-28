// Runs ONE instance through solve_lp on the first-order path and prints
// every quantity needed to tell a first-order failure apart from a
// pipeline failure.
//
// WHY THIS EXISTS
// ---------------
// `bench_pdlp` solves the RAW model; `solve_lp` solves the PRESOLVED one.
// When dfl001 converged in 2.9 s under the first and failed to converge at
// all under the second, that difference was the only candidate that the
// aggregate numbers could not distinguish -- and guessing between "presolve
// changed the model" and "the solver is misconfigured" is exactly the move
// that produced two wrong conclusions earlier in this project.
//
// So this prints both paths side by side, with presolve on and off, and
// reports the KKT triple the solver actually reached rather than only
// whether it passed.

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace sihps;

namespace {

const char* status_name(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPTIMAL";
        case LpStatus::INFEASIBLE: return "INFEASIBLE";
        case LpStatus::UNBOUNDED: return "UNBOUNDED";
        case LpStatus::ITERATION_LIMIT: return "ITER_LIMIT";
        case LpStatus::NUMERICAL_FAILURE: return "NUM_FAILURE";
    }
    return "?";
}

void run(const LpProblem& p, const char* label, LpMethod method, bool presolve, double eps) {
    LpSolverOptions opts;
    opts.method = method;
    opts.use_presolve = presolve;
    opts.pdlp.eps_optimal = eps;

    const auto t0 = std::chrono::steady_clock::now();
    LpSolution s;
    try {
        s = solve_lp(p, opts);
    } catch (const std::exception& e) {
        std::printf("%-34s THREW: %s\n", label, e.what());
        return;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("%-34s %-12s obj=%.10g  %.3f s\n", label, status_name(s.status),
                s.objective_value, secs);
    std::printf("%34s   presolve removed: %d rows, %d cols\n", "", s.presolve_removed_rows,
                s.presolve_removed_cols);
    if (s.used_first_order) {
        const PdlpStats& q = s.pdlp;
        std::printf("%34s   pdlp: converged=%d cancelled=%d iters=%d restarts=%d syncs=%d\n", "",
                    static_cast<int>(q.converged), static_cast<int>(q.cancelled), q.iterations,
                    q.restarts, q.host_syncs);
        std::printf("%34s   pdlp KKT: primal=%.3e dual=%.3e gap=%.3e  |A|=%.6g  eta_ratio=%.4g\n",
                    "", q.relative_primal_residual, q.relative_dual_residual, q.relative_gap,
                    q.matrix_norm, q.step_size_ratio);
        std::printf("%34s   pdlp rejected steps: %d\n", "", q.rejected_steps);
    }
    std::printf("%34s   verify: primal_res=%.3e dual_res=%.3e\n", "", s.primal_residual,
                s.dual_residual);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: diag_first_order <instance.mps> [eps]\n");
        return 1;
    }
    const double eps = (argc > 2) ? std::atof(argv[2]) : 1e-8;

    const MpsModel model = read_mps_file(argv[1]);
    const LpProblem p = lp_problem_from_mps(model);
    std::printf("%s: rows=%d cols=%d nnz=%d   eps=%g\n\n", argv[1], p.n_rows(), p.n_cols(),
                p.A.nnz(), eps);

    run(p, "FIRST_ORDER, presolve ON", LpMethod::FIRST_ORDER, true, eps);
    run(p, "FIRST_ORDER, presolve OFF", LpMethod::FIRST_ORDER, false, eps);
    return 0;
}
