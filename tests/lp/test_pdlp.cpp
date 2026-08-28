// Tests for the GPU PDLP solver (src/cuda/GpuPdlp.hpp).
//
// A first-order method is easy to get subtly wrong in a way that still
// looks like it works: a sign error in the dual step, or a support function
// with the wrong bound, produces an iteration that converges smoothly to
// the wrong point. So these tests check against HAND-COMPUTED optima on
// problems small enough to solve on paper, not against this project's own
// simplex -- two implementations agreeing can just as easily mean they
// share an assumption.
//
// The one exception is the afiro comparison at the end, which is there to
// check the two solvers agree on a real model once each has been
// independently validated.

#include "test_framework.hpp"

#include "cuda/GpuPdlp.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "lp/Scaling.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <string>
#include <vector>

using sihps::CSRMatrix;
using sihps::GpuPdlp;
using sihps::kInfinity;
using sihps::LpProblem;
using sihps::LpSolution;
using sihps::LpSolverOptions;
using sihps::LpStatus;
using sihps::PdlpParams;
using sihps::PdlpStats;
using sihps::Triplet;

namespace {

PdlpParams tight_params() {
    PdlpParams p;
    p.eps_optimal = 1e-9;
    p.restart_period = 64;
    p.max_iterations = 400000;
    p.time_limit_seconds = 20.0;
    return p;
}

} // namespace

SIHPS_TEST(pdlp_solves_a_hand_verified_two_variable_lp) {
    // maximize 3x + 5y  (i.e. minimize -3x - 5y)
    //   x        <= 4
    //        2y  <= 12
    //   3x + 2y  <= 18
    //   x, y >= 0
    //
    // The classic textbook LP. Optimum is at the intersection of the second
    // and third constraints: x = 2, y = 6, objective 36 (so -36 minimized).
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 2.0}, {2, 0, 3.0}, {2, 1, 2.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(3, 2, t);

    const std::vector<double> cost = {-3.0, -5.0};
    const std::vector<double> lower = {0.0, 0.0};
    const std::vector<double> upper = {kInfinity, kInfinity};
    // rl <= Ax <= ru, with rl = -inf for a pure "<=" row.
    const std::vector<double> row_lower = {-kInfinity, -kInfinity, -kInfinity};
    const std::vector<double> row_upper = {4.0, 12.0, 18.0};

    GpuPdlp pdlp(a, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x, y;
    const PdlpStats stats = pdlp.solve(tight_params(), x, y);

    SIHPS_ASSERT_TRUE(stats.converged);
    SIHPS_ASSERT_NEAR(x[0], 2.0, 1e-5);
    SIHPS_ASSERT_NEAR(x[1], 6.0, 1e-5);
    SIHPS_ASSERT_NEAR(cost[0] * x[0] + cost[1] * x[1], -36.0, 1e-5);
}

SIHPS_TEST(pdlp_handles_an_equality_row) {
    // minimize x + y  subject to  x + y == 5,  x,y in [0, 10].
    // Every feasible point is optimal with objective 5; the test is that
    // the equality is honoured, which is the case where row_lower ==
    // row_upper and the projection collapses to a single point.
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(1, 2, t);

    const std::vector<double> cost = {1.0, 1.0};
    const std::vector<double> lower = {0.0, 0.0};
    const std::vector<double> upper = {10.0, 10.0};
    const std::vector<double> row_lower = {5.0};
    const std::vector<double> row_upper = {5.0};

    GpuPdlp pdlp(a, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x, y;
    const PdlpStats stats = pdlp.solve(tight_params(), x, y);

    SIHPS_ASSERT_TRUE(stats.converged);
    SIHPS_ASSERT_NEAR(x[0] + x[1], 5.0, 1e-6);
    SIHPS_ASSERT_NEAR(cost[0] * x[0] + cost[1] * x[1], 5.0, 1e-6);
}

SIHPS_TEST(pdlp_respects_a_two_sided_row_range) {
    // minimize -x  subject to  2 <= x <= 7 (as a ROW, not a bound), with
    // the variable bound itself wide open at [0, 100].
    //
    // This separates the two projections: if the row range were being
    // treated as one-sided, or if the Moreau step used the wrong bound, the
    // answer would run to the variable's bound of 100 instead of stopping
    // at the row's upper limit of 7.
    std::vector<Triplet> t = {{0, 0, 1.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(1, 1, t);

    const std::vector<double> cost = {-1.0};
    const std::vector<double> lower = {0.0};
    const std::vector<double> upper = {100.0};
    const std::vector<double> row_lower = {2.0};
    const std::vector<double> row_upper = {7.0};

    GpuPdlp pdlp(a, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x, y;
    const PdlpStats stats = pdlp.solve(tight_params(), x, y);

    SIHPS_ASSERT_TRUE(stats.converged);
    SIHPS_ASSERT_NEAR(x[0], 7.0, 1e-6);
}

SIHPS_TEST(pdlp_respects_variable_upper_bounds) {
    // minimize -x - y  subject to  x + y <= 100,  x in [0,3], y in [0,4].
    // The row never binds; the answer is entirely determined by the
    // variable bounds, so a projection that ignored `upper` would run away.
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(1, 2, t);

    const std::vector<double> cost = {-1.0, -1.0};
    const std::vector<double> lower = {0.0, 0.0};
    const std::vector<double> upper = {3.0, 4.0};
    const std::vector<double> row_lower = {-kInfinity};
    const std::vector<double> row_upper = {100.0};

    GpuPdlp pdlp(a, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x, y;
    const PdlpStats stats = pdlp.solve(tight_params(), x, y);

    SIHPS_ASSERT_TRUE(stats.converged);
    SIHPS_ASSERT_NEAR(x[0], 3.0, 1e-6);
    SIHPS_ASSERT_NEAR(x[1], 4.0, 1e-6);
}

SIHPS_TEST(pdlp_reports_the_sync_count_it_claims_to_minimize) {
    // The entire architectural case for PDLP on a GPU is that it does NOT
    // synchronize per iteration (GpuPdlp.hpp). If a future change quietly
    // introduced a per-iteration sync, every correctness test here would
    // still pass and only the speed would collapse -- so the invariant is
    // asserted directly: syncs must stay far below iterations.
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 2.0}, {2, 0, 3.0}, {2, 1, 2.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(3, 2, t);
    const std::vector<double> cost = {-3.0, -5.0};
    const std::vector<double> lower = {0.0, 0.0};
    const std::vector<double> upper = {kInfinity, kInfinity};
    const std::vector<double> row_lower = {-kInfinity, -kInfinity, -kInfinity};
    const std::vector<double> row_upper = {4.0, 12.0, 18.0};

    GpuPdlp pdlp(a, cost, lower, upper, row_lower, row_upper);
    PdlpParams params = tight_params();
    params.restart_period = 128;
    std::vector<double> x, y;
    const PdlpStats stats = pdlp.solve(params, x, y);

    SIHPS_ASSERT_TRUE(stats.converged);
    SIHPS_ASSERT_TRUE(stats.iterations > 0);
    // Two evaluations per check window (current iterate and average), plus
    // a handful of fixed syncs for setup and read-back. Anything close to
    // one sync per iteration means the design has been broken.
    SIHPS_ASSERT_TRUE(stats.host_syncs < stats.iterations / 8 + 16);
}

SIHPS_TEST(pdlp_is_reproducible_across_repeated_solves) {
    // The reductions inside the residual kernel are combined by whichever
    // block retires last, which is genuinely nondeterministic. The combine
    // order is not (PdlpKernels.cu), so the iteration count and objective
    // must be identical run to run. Exact equality: a tolerance would pass
    // even if block-order dependence had crept in.
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 2.0}, {2, 0, 3.0}, {2, 1, 2.0}};
    const CSRMatrix a = CSRMatrix::from_triplets(3, 2, t);
    const std::vector<double> cost = {-3.0, -5.0};
    const std::vector<double> lower = {0.0, 0.0};
    const std::vector<double> upper = {kInfinity, kInfinity};
    const std::vector<double> row_lower = {-kInfinity, -kInfinity, -kInfinity};
    const std::vector<double> row_upper = {4.0, 12.0, 18.0};

    GpuPdlp first(a, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x1, y1;
    const PdlpStats s1 = first.solve(tight_params(), x1, y1);

    for (int rep = 0; rep < 3; ++rep) {
        GpuPdlp again(a, cost, lower, upper, row_lower, row_upper);
        std::vector<double> x2, y2;
        const PdlpStats s2 = again.solve(tight_params(), x2, y2);
        SIHPS_ASSERT_EQ(s2.iterations, s1.iterations);
        SIHPS_ASSERT_NEAR(x2[0], x1[0], 0.0);
        SIHPS_ASSERT_NEAR(x2[1], x1[1], 0.0);
    }
}

SIHPS_TEST(pdlp_agrees_with_the_simplex_on_afiro) {
    // Now that both solvers have been checked against hand-computed optima
    // independently, agreement on a real model is meaningful.
    const auto model =
        sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/afiro.mps");
    const LpProblem p = sihps::lp_problem_from_mps(model);

    // Ruiz then Pock-Chambolle, as the first-order path uses.
    const sihps::ScaleFactors ruiz = sihps::compute_ruiz_scaling(p.A);
    const CSRMatrix a_ruiz = sihps::apply_ruiz_scaling(p.A, ruiz);
    const sihps::ScaleFactors pc = sihps::compute_pock_chambolle_scaling(a_ruiz);
    const sihps::ScaleFactors scale = sihps::compose_scaling(ruiz, pc);
    const CSRMatrix a_scaled = sihps::apply_ruiz_scaling(p.A, scale);

    const auto n = static_cast<std::size_t>(p.n_cols());
    const auto m = static_cast<std::size_t>(p.n_rows());
    std::vector<double> cost(n), lower(n), upper(n), row_lower(m), row_upper(m);
    for (std::size_t j = 0; j < n; ++j) {
        const double c = scale.col_scale[j];
        cost[j] = p.obj[j] * c;
        lower[j] = p.lower[j] / c;
        upper[j] = p.upper[j] / c;
    }
    for (std::size_t i = 0; i < m; ++i) {
        const double r = scale.row_scale[i];
        row_lower[i] = r * (p.rhs[i] - p.slack_upper[i]);
        row_upper[i] = r * (p.rhs[i] - p.slack_lower[i]);
    }

    GpuPdlp pdlp(a_scaled, cost, lower, upper, row_lower, row_upper);
    PdlpParams params = tight_params();
    params.eps_optimal = 1e-9;
    std::vector<double> x_scaled, y_scaled;
    const PdlpStats stats = pdlp.solve(params, x_scaled, y_scaled);
    SIHPS_ASSERT_TRUE(stats.converged);

    double pdlp_obj = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        pdlp_obj += p.obj[j] * (x_scaled[j] * scale.col_scale[j]);
    }

    const LpSolution simplex = sihps::solve_lp(p, LpSolverOptions{});
    SIHPS_ASSERT_TRUE(simplex.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(pdlp_obj, simplex.objective_value,
                       1e-5 * (1.0 + std::fabs(simplex.objective_value)));
}
