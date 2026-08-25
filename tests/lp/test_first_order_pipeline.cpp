// End-to-end test of LpMethod::FIRST_ORDER through solve_lp.
//
// tests/lp/test_pdlp.cpp validates the SOLVER (GpuPdlp) against
// hand-computed optima. This file validates the PIPELINE around it, which
// is a separate set of ways to be wrong: presolve reduces the model, the
// first-order path re-scales what presolve produced, and postsolve has to
// lift the answer back into the caller's column space. A sign or index slip
// in any of those produces a solver that converges beautifully to the wrong
// answer for the original problem.
//
// So these compare against the SIMPLEX path on the same model rather than
// against a hand-computed value: the question here is not "is PDHG correct"
// (test_pdlp.cpp settles that) but "do the two methods agree once both have
// been through the same presolve, scaling and postsolve".

#include "test_framework.hpp"

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <cmath>
#include <string>

using sihps::LpMethod;
using sihps::LpProblem;
using sihps::LpSolution;
using sihps::LpSolverOptions;
using sihps::LpStatus;

namespace {

LpProblem netlib(const char* stem) {
    const auto model = sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) +
                                             "/data/netlib_lp/feasible/" + stem + ".mps");
    return sihps::lp_problem_from_mps(model);
}

LpSolverOptions first_order_options() {
    LpSolverOptions o;
    o.method = LpMethod::FIRST_ORDER;
    o.pdlp.eps_optimal = 1e-8;
    o.pdlp.restart_period = 64;
    o.pdlp.time_limit_seconds = 30.0;
    return o;
}

} // namespace

SIHPS_TEST(first_order_pipeline_matches_the_simplex_on_afiro) {
    const LpProblem p = netlib("afiro");

    const LpSolution simplex = sihps::solve_lp(p, LpSolverOptions{});
    const LpSolution pdlp = sihps::solve_lp(p, first_order_options());

    SIHPS_ASSERT_TRUE(simplex.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(pdlp.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(pdlp.used_first_order);
    SIHPS_ASSERT_NEAR(pdlp.objective_value, simplex.objective_value,
                       1e-5 * (1.0 + std::fabs(simplex.objective_value)));
}

SIHPS_TEST(first_order_pipeline_matches_the_simplex_on_sctap1) {
    // Larger, and presolve actually removes rows and columns here -- so the
    // model PDLP sees is NOT the model the caller handed in, and postsolve
    // has real work to do lifting the answer back.
    const LpProblem p = netlib("sctap1");

    const LpSolution simplex = sihps::solve_lp(p, LpSolverOptions{});
    const LpSolution pdlp = sihps::solve_lp(p, first_order_options());

    SIHPS_ASSERT_TRUE(simplex.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(pdlp.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(pdlp.objective_value, simplex.objective_value,
                       1e-5 * (1.0 + std::fabs(simplex.objective_value)));
}

SIHPS_TEST(first_order_result_passes_the_same_original_space_gate) {
    // A FIRST_ORDER result reported OPTIMAL must satisfy the SAME
    // original-space primal residual check every simplex result satisfies
    // (NUMERICS.md 6). Converging on PDLP's own relative KKT triple is a
    // claim about the scaled, presolved problem -- this asserts the claim
    // solve_lp actually makes, which is about the caller's model.
    const LpProblem p = netlib("sctap1");
    const LpSolution pdlp = sihps::solve_lp(p, first_order_options());

    SIHPS_ASSERT_TRUE(pdlp.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(pdlp.primal_residual < 1e-6);
    SIHPS_ASSERT_EQ(static_cast<int>(pdlp.x.size()), static_cast<int>(p.n_cols()));
}

SIHPS_TEST(first_order_pipeline_keeps_its_sync_count_low) {
    // The reason this path exists at all is that it does not synchronize
    // per iteration (PDLP.md 1). Asserted at the pipeline level too, so a
    // regression cannot hide behind presolve.
    const LpProblem p = netlib("sctap1");
    const LpSolution pdlp = sihps::solve_lp(p, first_order_options());

    SIHPS_ASSERT_TRUE(pdlp.used_first_order);
    SIHPS_ASSERT_TRUE(pdlp.pdlp.iterations > 0);
    SIHPS_ASSERT_TRUE(pdlp.pdlp.host_syncs < pdlp.pdlp.iterations / 8 + 16);
}

SIHPS_TEST(simplex_remains_the_default_method) {
    // The default must stay SIMPLEX. Every validated result in this
    // repository -- 89/89 Netlib, the infeasible set -- was produced by it,
    // and a first-order method returns a near-optimal point rather than an
    // exact vertex. Flipping the default would silently change what
    // "OPTIMAL" means for every existing caller.
    const LpSolverOptions defaults;
    SIHPS_ASSERT_TRUE(defaults.method == LpMethod::SIMPLEX);

    const LpProblem p = netlib("afiro");
    const LpSolution s = sihps::solve_lp(p, defaults);
    SIHPS_ASSERT_TRUE(s.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(!s.used_first_order);
}
