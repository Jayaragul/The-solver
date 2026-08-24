#include "LpSolver.hpp"

#include "Scaling.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>

namespace sihps {
namespace {

// Acceptance threshold for the original-space primal check. Matches the
// tolerance Simplex certifies its own result against, so composing the
// pipeline does not silently loosen what "feasible" means.
constexpr double kFinalPrimalTol = 1e-6;

double vector_inf_norm(const std::vector<double>& v) {
    double best = 0.0;
    for (double x : v) best = std::max(best, std::fabs(x));
    return best;
}

// Row and column feasibility of `x` against the ORIGINAL model.
double original_space_primal_residual(const LpProblem& problem, const std::vector<double>& x) {
    const std::int32_t m = problem.n_rows();
    const std::int32_t n = problem.n_cols();

    std::vector<double> ax(static_cast<std::size_t>(m), 0.0);
    if (n > 0 && m > 0) problem.A.multiply(x.data(), ax.data());

    double row_violation = 0.0;
    for (std::int32_t i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double lo = problem.rhs[ii] - problem.slack_upper[ii];
        const double hi = problem.rhs[ii] - problem.slack_lower[ii];
        if (std::isfinite(lo)) row_violation = std::max(row_violation, lo - ax[ii]);
        if (std::isfinite(hi)) row_violation = std::max(row_violation, ax[ii] - hi);
    }
    double bound_violation = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        bound_violation = std::max(bound_violation, problem.lower[jj] - x[jj]);
        bound_violation = std::max(bound_violation, x[jj] - problem.upper[jj]);
    }
    row_violation = std::max(0.0, row_violation);
    bound_violation = std::max(0.0, bound_violation);

    const double rhs_norm = vector_inf_norm(problem.rhs);
    return std::max(row_violation / (1.0 + rhs_norm), bound_violation);
}

// Runs GPU PDLP on `target` and returns its primal point in that problem's
// own column space, or false if it did not converge.
//
// SCALING IS PART OF THE ALGORITHM HERE, not a tidiness measure. A
// first-order method's convergence rate depends directly on the
// conditioning of A, so the matrix is equilibrated with Ruiz and then
// preconditioned with Pock-Chambolle (Scaling.hpp explains why both, and
// why in that order). Skipping either turns models that converge in
// thousands of iterations into models that do not converge at all.
bool run_first_order(const LpProblem& target, const PdlpParams& params, std::vector<double>& x_out,
                     PdlpStats& stats_out) {
    const ScaleFactors ruiz = compute_ruiz_scaling(target.A);
    const CSRMatrix a_ruiz = apply_ruiz_scaling(target.A, ruiz);
    const ScaleFactors pc = compute_pock_chambolle_scaling(a_ruiz);
    const ScaleFactors scale = compose_scaling(ruiz, pc);
    const CSRMatrix a_scaled = apply_ruiz_scaling(target.A, scale);

    const auto n = static_cast<std::size_t>(target.n_cols());
    const auto m = static_cast<std::size_t>(target.n_rows());

    // x = C x', so costs multiply by the column scale and bounds divide by
    // it; the objective VALUE is invariant under this change of variable.
    std::vector<double> cost(n), lower(n), upper(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double c = scale.col_scale[j];
        cost[j] = target.obj[j] * c;
        lower[j] = target.lower[j] / c;
        upper[j] = target.upper[j] / c;
    }

    // A x + s = rhs with slack_lower <= s <= slack_upper is exactly
    // rhs - slack_upper <= A x <= rhs - slack_lower.
    std::vector<double> row_lower(m), row_upper(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double r = scale.row_scale[i];
        row_lower[i] = r * (target.rhs[i] - target.slack_upper[i]);
        row_upper[i] = r * (target.rhs[i] - target.slack_lower[i]);
    }

    GpuPdlp pdlp(a_scaled, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x_scaled, y_scaled;
    stats_out = pdlp.solve(params, x_scaled, y_scaled);

    x_out.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) x_out[j] = x_scaled[j] * scale.col_scale[j];
    return stats_out.converged;
}

} // namespace

LpSolution solve_lp(const LpProblem& problem, const LpSolverOptions& options) {
    LpSolution solution;
    const std::int32_t n = problem.n_cols();

    PresolveResult reduction;
    const LpProblem* target = &problem;

    if (options.use_presolve) {
        const auto t0 = std::chrono::steady_clock::now();
        reduction = presolve(problem);
        solution.presolve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (reduction.status == PresolveStatus::INFEASIBLE) {
            solution.status = LpStatus::INFEASIBLE;
            return solution;
        }
        if (reduction.status == PresolveStatus::UNBOUNDED) {
            solution.status = LpStatus::UNBOUNDED;
            return solution;
        }
        solution.presolve_removed_rows = reduction.removed_rows();
        solution.presolve_removed_cols = reduction.removed_cols();
        target = &reduction.reduced;
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<double> reduced_x;
    if (target->n_cols() == 0) {
        // Presolve fixed every variable. There is nothing left to optimize,
        // but the result still has to clear the same verification gate as
        // any other -- it is not exempt just because no simplex ran.
        solution.status = LpStatus::OPTIMAL;
    } else if (options.method == LpMethod::FIRST_ORDER) {
        solution.used_first_order = true;
        PdlpStats stats;
        bool ok = false;
        try {
            ok = run_first_order(*target, options.pdlp, reduced_x, stats);
        } catch (const std::exception&) {
            ok = false;
        }
        solution.pdlp = stats;
        solution.iterations = stats.iterations;
        // Converging on PDLP's own relative KKT triple is a claim about the
        // scaled reduced problem. It is NOT the claim this function makes,
        // which is about the ORIGINAL model -- so the status is provisional
        // here and only becomes OPTIMAL after the same original-space gate
        // every other result passes through, below.
        solution.status = ok ? LpStatus::OPTIMAL : LpStatus::ITERATION_LIMIT;
    } else if (options.method == LpMethod::HYBRID) {
        // ---------------------------------------------------------------
        // THE RACE
        // ---------------------------------------------------------------
        // The two methods fail on disjoint, unpredictable sets of models,
        // and -- crucially -- they contend for almost no shared hardware:
        // the simplex is CPU-bound and PDLP is GPU-bound. Running them
        // concurrently and taking the first VERIFIED result therefore costs
        // roughly max(0, overhead) rather than the sum of both, and turns
        // the engine's wall clock into min(simplex, pdlp) per instance.
        //
        // Sequential fallback was tried first and rejected: on `dfl001` the
        // simplex grinds for 792 s before conceding, so simplex-then-PDLP
        // costs 794 s where the race costs about 3 s.
        //
        // Neither competitor is trusted on its own say-so. Whoever finishes
        // first only sets the cancel flag; the answer still has to clear
        // the same original-space verification gate below.
        // Only a CONVERGED first-order result may cancel the simplex.
        //
        // The first version of this raced on "the GPU thread finished",
        // which is a different and much weaker condition: PDLP exhausting
        // its 500,000 iterations without converging also finishes, and it
        // was killing simplex solves that were about to succeed. That took
        // Netlib validation from 92/93 down to 86/93 -- `stocfor3`,
        // `pilot87`, `dfl001`, `pilot`, `woodw`, `d2q06c` and `greenbeb`
        // all reported ITERATION_LIMIT after roughly 60 s with only a few
        // thousand iterations and no refactorizations, which is the
        // signature of a cancelled solve rather than a stalled one.
        //
        // A loser must never be able to stop a winner. Each side now
        // cancels the other only on a result it could actually return.
        std::atomic<bool> pdlp_succeeded{false};
        std::atomic<bool> simplex_done{false};

        std::vector<double> pdlp_x;
        PdlpStats pdlp_stats;

        PdlpParams pdlp_params = options.pdlp;
        pdlp_params.cancel = &simplex_done;

        std::thread gpu_thread([&]() {
            bool ok = false;
            try {
                ok = run_first_order(*target, pdlp_params, pdlp_x, pdlp_stats);
            } catch (const std::exception&) {
                ok = false;
            }
            if (ok) pdlp_succeeded.store(true, std::memory_order_release);
        });

        Simplex simplex(*target, options.backend, options.use_ruiz_scaling, options.pricing_rule,
                         options.algorithm);
        simplex.set_cancel_flag(&pdlp_succeeded);
        LpResult lp = simplex.solve();
        const bool simplex_won = lp.status == LpStatus::OPTIMAL ||
                                  lp.status == LpStatus::INFEASIBLE ||
                                  lp.status == LpStatus::UNBOUNDED;
        if (simplex_won) simplex_done.store(true, std::memory_order_release);

        gpu_thread.join();

        if (simplex_won) {
            solution.status = lp.status;
            solution.iterations = lp.phase1_iterations + lp.phase2_iterations + lp.dual_iterations;
            solution.used_dual_simplex = lp.used_dual_simplex;
            solution.dual_iterations = lp.dual_iterations;
            solution.refactorizations = lp.refactorizations;
            solution.pricing_seconds = lp.pricing_seconds;
            solution.simplex_profile = lp.profile;
            solution.dual_residual = lp.dual_residual;
            solution.dual_residual_is_reduced_space = options.use_presolve;
            reduced_x = std::move(lp.x);
            solution.pdlp = pdlp_stats;
        } else if (pdlp_succeeded.load(std::memory_order_acquire)) {
            // The simplex gave up (or was cancelled) and PDLP converged.
            solution.used_first_order = true;
            solution.first_order_fallback_used = true;
            solution.pdlp = pdlp_stats;
            solution.iterations = pdlp_stats.iterations;
            solution.status = LpStatus::OPTIMAL; // provisional; gated below
            reduced_x = std::move(pdlp_x);
        } else {
            // Both failed. Report the simplex's status, which is the more
            // informative of the two.
            solution.status = lp.status;
            solution.iterations = lp.phase1_iterations + lp.phase2_iterations + lp.dual_iterations;
            solution.pdlp = pdlp_stats;
            reduced_x = std::move(lp.x);
        }
    } else {
        Simplex simplex(*target, options.backend, options.use_ruiz_scaling, options.pricing_rule,
                         options.algorithm);
        LpResult lp = simplex.solve();
        solution.status = lp.status;
        solution.iterations = lp.phase1_iterations + lp.phase2_iterations + lp.dual_iterations;
        solution.used_dual_simplex = lp.used_dual_simplex;
        solution.dual_iterations = lp.dual_iterations;
        solution.refactorizations = lp.refactorizations;
        solution.pricing_seconds = lp.pricing_seconds;
        solution.simplex_profile = lp.profile;
        solution.dual_residual = lp.dual_residual;
        solution.dual_residual_is_reduced_space = options.use_presolve;
        reduced_x = std::move(lp.x);

    }
    solution.solve_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    if (solution.status != LpStatus::OPTIMAL) {
        return solution;
    }

    solution.x = options.use_presolve ? postsolve(reduction, reduced_x) : reduced_x;
    if (static_cast<std::int32_t>(solution.x.size()) != n) {
        solution.x.resize(static_cast<std::size_t>(n), 0.0);
    }

    // Objective from the ORIGINAL cost vector and the reconstructed full
    // solution. Computing it this way means presolve needs no separate
    // objective-offset bookkeeping for the columns it fixed: their
    // contribution is already present in x.
    solution.objective_value = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
        solution.objective_value +=
            problem.obj[static_cast<std::size_t>(j)] * solution.x[static_cast<std::size_t>(j)];
    }

    // The hard invariant (docs/architecture/NUMERICS.md \S6) applied to the
    // whole pipeline: OPTIMAL survives only if the reconstructed solution is
    // feasible for the problem the CALLER posed. A presolve reduction that
    // was not exactly invertible fails here rather than being reported as a
    // clean optimum.
    solution.primal_residual = original_space_primal_residual(problem, solution.x);
    if (!(solution.primal_residual <= kFinalPrimalTol)) {
        solution.status = LpStatus::NUMERICAL_FAILURE;
    }
    return solution;
}

} // namespace sihps
