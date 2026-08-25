#include "../test_framework.hpp"

#include <cmath>
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/Simplex.hpp"
#include "sparse/Triplet.hpp"

using sihps::CSRMatrix;
using sihps::kInfinity;
using sihps::LpAlgorithm;
using sihps::LpProblem;
using sihps::LpStatus;
using sihps::PricingBackend;
using sihps::Simplex;
using sihps::Triplet;
using sihps::read_mps_file;

namespace {
std::string afiro_path() {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/afiro.mps";
}

// minimize -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x,y >= 0
//
// Hand-derived (not looked up): the feasible region's vertices are
// (0,0), (2,0), (0,2), and the intersection of the two constraints,
// found by solving x+2y=4 and 3x+y=6 simultaneously: x=4-2y, so
// 3(4-2y)+y=6 => 12-5y=6 => y=1.2, x=1.6. Objective x+y at each vertex:
// 0, 2, 2, and 2.8 respectively -- the last is the maximum of x+y, so
// minimizing -x-y gives optimal value -2.8 at (1.6, 1.2).
LpProblem tiny_lp() {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {4.0, 6.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);
    return p;
}

// x <= 3 (row L) and x >= 5 (row G) with x >= 0 -- infeasible by
// construction (3 < 5).
LpProblem tiny_infeasible_lp() {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 1, t);
    p.obj = {0.0};
    p.rhs = {3.0, 5.0};
    p.row_types = {'L', 'G'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    return p;
}

// minimize -x, s.t. -x <= 0 (i.e. x >= 0, restated as a row so the
// augmented system is non-trivial), x >= 0, no upper bound -- unbounded:
// objective -> -infinity as x -> infinity.
LpProblem tiny_unbounded_lp() {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, -1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {0.0};
    p.row_types = {'L'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    return p;
}
} // namespace

// minimize -x  s.t.  x <= 10 (L row, RANGES r=4) , x >= 0.
// Per LpProblem.cpp's L-row RANGES formula, slack bounds become [0, |r|]
// = [0, 4], i.e. s = 10 - x in [0,4] => x in [6, 10]. Intersected with the
// default x >= 0, feasible x is [6,10]; maximizing x (minimizing -x)
// gives x=10, objective=-10 -- hand-derived, not looked up.
SIHPS_TEST(simplex_handles_l_row_ranges_correctly) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {10.0};
    p.row_types = {'L'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    p.slack_lower[0] = 0.0;
    p.slack_upper[0] = 4.0; // RANGES r=4 on an L row
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], 10.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-6);
}

// minimize x  s.t.  x >= 2 (G row, RANGES r=3), x >= 0.
// G-row RANGES formula: slack bounds [-|r|, 0] = [-3, 0], s = 2 - x in
// [-3,0] => x in [2, 5]. Minimizing x gives x=2, objective=2.
SIHPS_TEST(simplex_handles_g_row_ranges_correctly) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {1.0};
    p.rhs = {2.0};
    p.row_types = {'G'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    p.slack_lower[0] = -3.0;
    p.slack_upper[0] = 0.0; // RANGES r=3 on a G row
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], 2.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, 2.0, 1e-6);
}

// minimize x  s.t.  x = 5 (E row, RANGES r=3, r>=0), x >= 0.
// E-row, r>=0 formula: slack bounds [-r, 0] = [-3, 0], s = 5 - x in
// [-3,0] => x in [5, 8]. Minimizing x gives x=5, objective=5.
SIHPS_TEST(simplex_handles_e_row_ranges_nonneg_correctly) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {1.0};
    p.rhs = {5.0};
    p.row_types = {'E'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    p.slack_lower[0] = -3.0;
    p.slack_upper[0] = 0.0; // RANGES r=3 (r>=0) on an E row
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], 5.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, 5.0, 1e-6);
}

// minimize -x  s.t.  x = 5 (E row, RANGES r=-3, r<0), x >= 0.
// E-row, r<0 formula: slack bounds [0, -r] = [0, 3], s = 5 - x in [0,3]
// => x in [2, 5]. Maximizing x (minimizing -x) gives x=5, objective=-5.
SIHPS_TEST(simplex_handles_e_row_ranges_negative_correctly) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {5.0};
    p.row_types = {'E'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    p.slack_lower[0] = 0.0;
    p.slack_upper[0] = 3.0; // RANGES r=-3 (r<0) on an E row
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], 5.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, -5.0, 1e-6);
}

// A free variable (both bounds infinite, MPS 'FR'): minimize x subject to
// x >= -5 expressed as a G row with rhs -5. The G-row slack lives in
// (-inf, 0], so x + s = -5 with s <= 0 gives x >= -5. Minimizing x drives
// it to the row bound: x = -5, s = 0, objective -5. Hand-derived.
SIHPS_TEST(simplex_supports_free_variable_bounded_by_a_row) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {1.0};
    p.rhs = {-5.0};
    p.row_types = {'G'};
    p.lower = {-kInfinity};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], -5.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, -5.0, 1e-6);
}

// A free variable with no row bounding it in the objective's direction of
// improvement is genuinely unbounded, and must be reported as such rather
// than silently returning some arbitrary vertex.
SIHPS_TEST(simplex_detects_unbounded_free_variable) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {1.0};
    p.rhs = {5.0};
    p.row_types = {'L'}; // x <= 5 bounds x above, but x may fall freely
    p.lower = {-kInfinity};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::UNBOUNDED);
}

// The false-INFEASIBLE regression this rewrite fixes, reduced to its
// minimum: a variable whose lower bound is NONZERO, so the phase-1
// starting point has A*x != 0. Constraint x + y = 10 with x in [4, 4]
// (fixed) and y >= 0 forces y = 6; objective y therefore optimizes to 6.
// Initialising the artificial to |rhs| instead of the true row residual
// made this report INFEASIBLE.
SIHPS_TEST(simplex_handles_nonzero_lower_bounds_in_phase1) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, 1.0};
    p.rhs = {10.0};
    p.row_types = {'E'};
    p.lower = {4.0, 0.0};
    p.upper = {4.0, kInfinity};
    sihps::apply_default_row_bounds(p);
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], 4.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 6.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, 6.0, 1e-6);
}

// Same shape, but with a NEGATIVE lower bound -- the exact configuration
// present in boeing1/boeing2 (e.g. 'LO INTBOU GRDTIMN1 -105.').
// x fixed at -105, x + y = -100 forces y = 5.
SIHPS_TEST(simplex_handles_negative_lower_bounds_in_phase1) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, 1.0};
    p.rhs = {-100.0};
    p.row_types = {'E'};
    p.lower = {-105.0, 0.0};
    p.upper = {-105.0, kInfinity};
    sihps::apply_default_row_bounds(p);
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.x[0], -105.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 5.0, 1e-6);
}

SIHPS_TEST(simplex_solves_tiny_hand_verified_lp_cpu) {
    LpProblem p = tiny_lp();
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 1.2, 1e-6);
    SIHPS_ASSERT_TRUE(result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(result.dual_residual < 1e-6);
}

SIHPS_TEST(simplex_solves_tiny_hand_verified_lp_gpu) {
    LpProblem p = tiny_lp();
    Simplex simplex(p, PricingBackend::GPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 1.2, 1e-6);
}

SIHPS_TEST(simplex_detects_infeasibility) {
    LpProblem p = tiny_infeasible_lp();
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::INFEASIBLE);
}

SIHPS_TEST(simplex_detects_unboundedness) {
    LpProblem p = tiny_unbounded_lp();
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::UNBOUNDED);
}

SIHPS_TEST(simplex_solves_afiro_end_to_end) {
    auto model = read_mps_file(afiro_path());
    LpProblem p = sihps::lp_problem_from_mps(model);
    Simplex simplex(p, PricingBackend::CPU);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(result.dual_residual < 1e-6);
    // Every structural variable must respect its bounds in the final
    // solution -- a direct, independent sanity check beyond the engine's
    // own residual computation.
    for (double x : result.x) {
        SIHPS_ASSERT_TRUE(x >= -1e-6);
    }
}

SIHPS_TEST(simplex_cpu_and_gpu_backends_agree_on_afiro) {
    auto model = read_mps_file(afiro_path());
    LpProblem p = sihps::lp_problem_from_mps(model);

    Simplex cpu_simplex(p, PricingBackend::CPU);
    auto cpu_result = cpu_simplex.solve();

    Simplex gpu_simplex(p, PricingBackend::GPU);
    auto gpu_result = gpu_simplex.solve();

    SIHPS_ASSERT_TRUE(cpu_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(gpu_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(cpu_result.objective_value, gpu_result.objective_value, 1e-5);
}

// --- Dual simplex path -------------------------------------------------
//
// AUTO resolves to the primal path on a cold start (Simplex::solve, and
// docs/architecture/LP.md \S2's decision table), so nothing above reaches
// run_dual_simplex. These pin the dual path explicitly: it exists for the
// warm-started B&B re-solves to come, and an untested algorithm would rot
// before that caller is written.

SIHPS_TEST(dual_simplex_solves_tiny_hand_verified_lp) {
    LpProblem p = tiny_lp();
    Simplex simplex(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                     LpAlgorithm::DUAL);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    // Same optimum as the primal path reaches, to the same tolerance: the
    // two algorithms must agree on the answer, not merely both succeed.
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 1.2, 1e-6);
    SIHPS_ASSERT_TRUE(result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(result.dual_residual < 1e-6);
}

SIHPS_TEST(dual_simplex_detects_infeasibility) {
    LpProblem p = tiny_infeasible_lp();
    Simplex simplex(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                     LpAlgorithm::DUAL);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::INFEASIBLE);
}

SIHPS_TEST(dual_simplex_falls_back_to_primal_without_a_dual_feasible_start) {
    // Unbounded below with an infinite upper bound on the only variable:
    // no assignment of nonbasic variables to bounds makes the all-slack
    // basis dual feasible, so setup_dual_feasible_start() must decline and
    // solve() must run the primal path rather than reporting a failure.
    LpProblem p = tiny_unbounded_lp();
    Simplex simplex(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                     LpAlgorithm::DUAL);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::UNBOUNDED);
    SIHPS_ASSERT_TRUE(!result.used_dual_simplex);
    SIHPS_ASSERT_TRUE(result.dual_iterations == 0);
}

// sctap1 rather than afiro: afiro has 4 columns whose cost points toward
// an infinite bound, so it admits NO dual-feasible all-slack start and the
// dual path is never entered on it. sctap1 admits one and the dual path
// carries its result all the way through verification, which is what makes
// used_dual_simplex assertable here.
SIHPS_TEST(dual_simplex_agrees_with_primal_on_sctap1) {
    auto model = read_mps_file(std::string(SIHPS_PROJECT_ROOT) +
                                "/data/netlib_lp/feasible/sctap1.mps");
    LpProblem p = sihps::lp_problem_from_mps(model);

    Simplex primal(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                    LpAlgorithm::PRIMAL);
    auto primal_result = primal.solve();

    Simplex dual(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::DUAL);
    auto dual_result = dual.solve();

    SIHPS_ASSERT_TRUE(primal_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(dual_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(dual_result.used_dual_simplex);
    SIHPS_ASSERT_TRUE(dual_result.dual_iterations > 0);
    // Relative rather than absolute agreement, so the assertion means the
    // same thing on a model of any magnitude.
    SIHPS_ASSERT_TRUE(std::fabs(primal_result.objective_value - dual_result.objective_value) <=
                       1e-9 * (1.0 + std::fabs(primal_result.objective_value)));
    SIHPS_ASSERT_TRUE(dual_result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(dual_result.dual_residual < 1e-6);
}

// REGRESSION: the dual ratio test was single-pass, taking the minimum
// ratio with only a 1e-12 tie-break toward larger pivots. On a model whose
// eligible pivots span several orders of magnitude that admits a pivot near
// kPivotTol, and since the primal step is delta / alpha_enter, one such
// pivot converts a small bound violation into an enormous one. Netlib
// grow15 diverged to a 1.4e+12 basic infeasibility and then reported
// INFEASIBLE on a feasible model. The two-pass Harris form fixed it; this
// is a scaled-down deterministic version of the same trap.
//
// Hand-derived: minimize -x1 - x2 subject to
//   r0:  1e-6 x1 +      x2 <= 1
//   r1:       x1 + 1e-6 x2 <= 1
// with x1, x2 in [0, 10]. Both costs are negative and both variables have
// finite upper bounds, so the all-slack basis IS dual feasible and the
// dual path is genuinely taken. Both variables start at their upper bound
// 10, making both rows violated, and the row coefficients differ by 1e6 so
// a single-pass rule is free to choose the 1e-6 pivot. The optimum is the
// intersection of the two rows, x1 = x2 = 1/(1 + 1e-6) by symmetry, with
// objective -2/(1 + 1e-6).
SIHPS_TEST(dual_simplex_ratio_test_rejects_tiny_pivots) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1e-6}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1e-6}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {1.0, 1.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {10.0, 10.0};
    sihps::apply_default_row_bounds(p);

    Simplex simplex(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                     LpAlgorithm::DUAL);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    const double expected = 1.0 / (1.0 + 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], expected, 1e-9);
    SIHPS_ASSERT_NEAR(result.x[1], expected, 1e-9);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.0 * expected, 1e-9);
    SIHPS_ASSERT_TRUE(result.primal_residual < 1e-9);
}

// --- Warm-started dual simplex (docs/architecture/LP.md \S1/\S2) ------
//
// The MILP B&B node loop is the caller these exist for: a child node's LP
// differs from its parent's by exactly one tightened variable bound, and
// the parent's optimal basis stays dual-feasible for the child by
// construction (reduced costs depend only on cost_/basis_, neither of
// which a bound-only change touches). These pin that mechanism directly
// against Simplex, independent of MilpSolver -- see test_milp.cpp for the
// MILP-level differential tests that exercise it end to end.

SIHPS_TEST(warm_start_basis_round_trips_export_and_seat) {
    LpProblem p = tiny_lp();

    Simplex cold(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::PRIMAL);
    auto cold_result = cold.solve();
    SIHPS_ASSERT_TRUE(cold_result.status == LpStatus::OPTIMAL);
    const Simplex::Basis basis = cold.export_basis();

    // Same problem, no bound change at all -- the parent's basis should
    // seat and be immediately recognized as optimal.
    Simplex warm(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    warm.set_warm_start_basis(&basis);
    auto warm_result = warm.solve();

    SIHPS_ASSERT_TRUE(warm_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.warm_start_attempted);
    SIHPS_ASSERT_TRUE(warm_result.used_warm_start);
    SIHPS_ASSERT_NEAR(warm_result.objective_value, cold_result.objective_value, 1e-9);
    SIHPS_ASSERT_NEAR(warm_result.x[0], cold_result.x[0], 1e-9);
    SIHPS_ASSERT_NEAR(warm_result.x[1], cold_result.x[1], 1e-9);
}

// The core differential test: a child whose bound differs from the
// parent by exactly one tightened upper bound, matching the B&B node
// shape. child.upper[0] = 1.0 simulates a left branch on x, floor(1.6).
//
// Hand-derived: along the binding row x + 2y <= 4, x+y = (x+4)/2, which
// increases with x -- so capping x at 1 (below its unconstrained optimum
// 1.6) pushes the new optimum to the cap: x=1, y=(4-1)/2=1.5, checked
// against row 2 (3x+y=4.5 <= 6, not binding). Objective -(x+y) = -2.5.
SIHPS_TEST(warm_start_dual_simplex_matches_cold_solve_after_bound_tightening) {
    LpProblem parent = tiny_lp();
    Simplex parent_simplex(parent, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                            LpAlgorithm::PRIMAL);
    auto parent_result = parent_simplex.solve();
    SIHPS_ASSERT_TRUE(parent_result.status == LpStatus::OPTIMAL);
    const Simplex::Basis parent_basis = parent_simplex.export_basis();

    LpProblem child = parent;
    child.upper[0] = 1.0;

    Simplex cold(child, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    auto cold_result = cold.solve();

    Simplex warm(child, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    warm.set_warm_start_basis(&parent_basis);
    auto warm_result = warm.solve();

    SIHPS_ASSERT_TRUE(cold_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.used_warm_start);
    SIHPS_ASSERT_TRUE(warm_result.dual_iterations >= 1);
    SIHPS_ASSERT_NEAR(cold_result.objective_value, -2.5, 1e-6);
    SIHPS_ASSERT_NEAR(warm_result.objective_value, cold_result.objective_value, 1e-9);
    SIHPS_ASSERT_NEAR(warm_result.x[0], cold_result.x[0], 1e-9);
    SIHPS_ASSERT_NEAR(warm_result.x[1], cold_result.x[1], 1e-9);
    SIHPS_ASSERT_TRUE(warm_result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(warm_result.dual_residual < 1e-6);
}

// A shape mismatch (wrong n_struct) must be refused before seat_basis even
// runs -- solve()'s own guard, ahead of seat_basis's internal one -- and
// the solve must still reach the correct cold answer rather than fail.
SIHPS_TEST(warm_start_falls_back_when_basis_shape_mismatches) {
    LpProblem p = tiny_lp();
    Simplex::Basis bad_basis;
    bad_basis.n_struct = 999;
    bad_basis.n_slack = 2;
    bad_basis.n_art = 2;

    Simplex simplex(p, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    simplex.set_warm_start_basis(&bad_basis);
    auto result = simplex.solve();

    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(!result.warm_start_attempted);
    SIHPS_ASSERT_TRUE(!result.used_warm_start);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
}

// A nonbasic FREE (AT_ZERO) column whose bounds have since become finite
// is the one structural case seat_basis refuses outright rather than
// guesses at. x has zero cost and an all-zero column, so it never leaves
// AT_ZERO in the parent; the child gives it a finite lower bound it did
// not have before.
SIHPS_TEST(warm_start_free_variable_edge_case_falls_back_safely) {
    LpProblem parent;
    std::vector<Triplet> t = {{0, 1, 1.0}}; // only y (col 1) appears in the row
    parent.A = CSRMatrix::from_triplets(1, 2, t);
    parent.obj = {0.0, -1.0}; // minimize -y; x has no objective pressure
    parent.rhs = {5.0};
    parent.row_types = {'L'};
    parent.lower = {-kInfinity, 0.0};
    parent.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(parent);

    Simplex parent_simplex(parent, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                            LpAlgorithm::PRIMAL);
    auto parent_result = parent_simplex.solve();
    SIHPS_ASSERT_TRUE(parent_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(parent_result.x[0], 0.0, 1e-9); // x rests AT_ZERO
    SIHPS_ASSERT_NEAR(parent_result.x[1], 5.0, 1e-9);
    const Simplex::Basis parent_basis = parent_simplex.export_basis();

    LpProblem child = parent;
    child.lower[0] = 0.0; // x now has a finite side it did not have before

    Simplex warm(child, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    warm.set_warm_start_basis(&parent_basis);
    auto warm_result = warm.solve();

    SIHPS_ASSERT_TRUE(warm_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.warm_start_attempted);
    SIHPS_ASSERT_TRUE(!warm_result.used_warm_start); // seat_basis refused; cold path answered
    SIHPS_ASSERT_NEAR(warm_result.objective_value, -5.0, 1e-6);
    SIHPS_ASSERT_NEAR(warm_result.x[1], 5.0, 1e-6);
    SIHPS_ASSERT_TRUE(warm_result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(warm_result.dual_residual < 1e-6);
}

// sctap1-scale differential test -- not just a hand-built 2x2. Tightens
// whichever structural variable the parent solution rests largest on, a
// stand-in for a branching decision chosen this way so the test does not
// depend on sctap1's specific column meanings.
SIHPS_TEST(warm_start_agrees_with_cold_solve_on_sctap1_after_a_bound_change) {
    auto model = read_mps_file(std::string(SIHPS_PROJECT_ROOT) +
                                "/data/netlib_lp/feasible/sctap1.mps");
    LpProblem parent = sihps::lp_problem_from_mps(model);

    Simplex parent_simplex(parent, PricingBackend::CPU, true, sihps::PricingRule::DEVEX,
                            LpAlgorithm::DUAL);
    auto parent_result = parent_simplex.solve();
    SIHPS_ASSERT_TRUE(parent_result.status == LpStatus::OPTIMAL);
    const Simplex::Basis parent_basis = parent_simplex.export_basis();

    std::size_t branch_col = 0;
    double branch_value = parent_result.x[0];
    for (std::size_t j = 1; j < parent_result.x.size(); ++j) {
        if (parent_result.x[j] > branch_value) {
            branch_value = parent_result.x[j];
            branch_col = j;
        }
    }
    SIHPS_ASSERT_TRUE(branch_value > 1.0); // otherwise this test proves nothing

    LpProblem child = parent;
    child.upper[branch_col] = branch_value / 2.0;

    Simplex cold(child, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    auto cold_result = cold.solve();

    Simplex warm(child, PricingBackend::CPU, true, sihps::PricingRule::DEVEX, LpAlgorithm::AUTO);
    warm.set_warm_start_basis(&parent_basis);
    auto warm_result = warm.solve();

    SIHPS_ASSERT_TRUE(cold_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm_result.used_warm_start);
    SIHPS_ASSERT_TRUE(warm_result.dual_iterations >= 1);
    SIHPS_ASSERT_TRUE(std::fabs(warm_result.objective_value - cold_result.objective_value) <=
                       1e-9 * (1.0 + std::fabs(cold_result.objective_value)));
    SIHPS_ASSERT_TRUE(warm_result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(warm_result.dual_residual < 1e-6);
}
