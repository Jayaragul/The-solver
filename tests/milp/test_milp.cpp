#include "test_framework.hpp"

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "milp/MilpProblem.hpp"
#include "milp/MilpSolver.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <string>
#include <vector>

using sihps::CSRMatrix;
using sihps::LpProblem;
using sihps::MilpBranchingRule;
using sihps::MilpProblem;
using sihps::MilpStatus;
using sihps::MilpSolverOptions;
using sihps::Triplet;
using sihps::VariableType;

namespace {

MilpProblem binary_knapsack() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 2, {Triplet{0, 0, 6.0}, Triplet{0, 1, 5.0}});
    lp.obj = {-10.0, -9.0};
    lp.rhs = {7.0};
    lp.row_types = {'L'};
    lp.lower = {0.0, 0.0};
    lp.upper = {1.0, 1.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::BINARY, VariableType::BINARY}};
}

MilpProblem integer_cover() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {1.0};
    lp.rhs = {3.0};
    lp.row_types = {'G'};
    lp.lower = {0.0};
    lp.upper = {10.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

MilpProblem impossible_integer_equality() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {0.0};
    lp.rhs = {1.0};
    lp.row_types = {'E'};
    lp.lower = {0.0};
    lp.upper = {1.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

MilpProblem mixed_nonnegative_packing_row() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(
        1, 3, {Triplet{0, 0, 6.0}, Triplet{0, 1, 5.0}, Triplet{0, 2, 1.0}});
    lp.obj = {-10.0, -9.0, 0.0};
    lp.rhs = {7.0};
    lp.row_types = {'L'};
    lp.lower = {0.0, 0.0, 0.0};
    lp.upper = {1.0, 1.0, 10.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp),
                       {VariableType::BINARY, VariableType::BINARY, VariableType::CONTINUOUS}};
}

} // namespace

SIHPS_TEST(milp_finds_integer_optimum_with_fractional_lp_root) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.has_incumbent);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.x[0], 1.0, 0.0);
    SIHPS_ASSERT_NEAR(result.x[1], 0.0, 0.0);
    SIHPS_ASSERT_TRUE(result.nodes_processed >= 3);
    SIHPS_ASSERT_TRUE(result.nodes_pruned >= 1);
    SIHPS_ASSERT_TRUE(result.strong_branching_probes >= 2);
    SIHPS_ASSERT_NEAR(result.relative_gap, 0.0, 0.0);
}

SIHPS_TEST(milp_parallel_strong_branching_preserves_certificate) {
    MilpSolverOptions serial_options;
    serial_options.use_rounding_heuristic = false;
    serial_options.enable_root_cover_cuts = false;
    serial_options.lp_options.parallel_mode = sihps::ParallelMode::SERIAL;
    const auto serial = sihps::solve_milp(binary_knapsack(), serial_options);

    MilpSolverOptions parallel_options = serial_options;
    parallel_options.lp_options.parallel_mode = sihps::ParallelMode::PARALLEL;
    const auto parallel = sihps::solve_milp(binary_knapsack(), parallel_options);

    SIHPS_ASSERT_TRUE(serial.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(parallel.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(serial.has_incumbent && parallel.has_incumbent);
    SIHPS_ASSERT_NEAR(parallel.objective_value, serial.objective_value, 1e-8);
    SIHPS_ASSERT_NEAR(parallel.best_bound, serial.best_bound, 1e-8);
    SIHPS_ASSERT_TRUE(parallel.strong_branching_probes == serial.strong_branching_probes);
}

SIHPS_TEST(milp_handles_general_integer_variables) {
    const auto result = sihps::solve_milp(integer_cover());

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 2.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.x[0], 2.0, 0.0);
}

SIHPS_TEST(milp_reports_maximization_objective_in_original_sense) {
    auto problem = binary_knapsack();
    problem.maximize = true;
    for (double& coefficient : problem.relaxation.obj) coefficient = -coefficient;

    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 10.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.best_bound, 10.0, 1e-8);
}

SIHPS_TEST(milp_proves_integer_infeasibility_by_exhausting_nodes) {
    const auto result = sihps::solve_milp(impossible_integer_equality());

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::INFEASIBLE);
    SIHPS_ASSERT_TRUE(!result.has_incumbent);
    SIHPS_ASSERT_TRUE(result.nodes_processed >= 3);
}

SIHPS_TEST(milp_node_limit_is_not_reported_as_optimal) {
    MilpSolverOptions options;
    options.node_limit = 1;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::NODE_LIMIT);
    SIHPS_ASSERT_TRUE(result.status != MilpStatus::OPTIMAL);
}

SIHPS_TEST(milp_root_cover_cut_is_valid_and_improves_the_root_relaxation) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(milp_mixed_nonnegative_packing_row_gets_valid_cover_cut) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(milp_unit_bounded_integer_variables_get_binary_cover_cuts) {
    auto problem = binary_knapsack();
    problem.variable_types = {VariableType::INTEGER, VariableType::INTEGER};
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(problem, options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(milp_maps_unbounded_pure_continuous_relaxation_to_unbounded) {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(0, 1, {});
    lp.obj = {-1.0};
    lp.rhs = {};
    lp.row_types = {};
    lp.lower = {0.0};
    lp.upper = {sihps::kInfinity};
    MilpProblem problem{std::move(lp), {VariableType::CONTINUOUS}};
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::UNBOUNDED);
}

SIHPS_TEST(mps_integer_markers_and_binary_bounds_are_preserved) {
    const auto model =
        sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/tests/data/tiny_milp.mps");
    SIHPS_ASSERT_EQ(model.col_types.size(), static_cast<std::size_t>(2));
    SIHPS_ASSERT_TRUE(model.col_types[0] == VariableType::BINARY);
    SIHPS_ASSERT_TRUE(model.col_types[1] == VariableType::INTEGER);

    const auto problem = sihps::milp_problem_from_mps(model);
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(mps_maximize_sense_is_converted_and_reported_correctly) {
    const auto model = sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) +
                                             "/tests/data/tiny_milp_max.mps");
    SIHPS_ASSERT_TRUE(model.objective_sense == sihps::ObjectiveSense::MAXIMIZE);
    const auto problem = sihps::milp_problem_from_mps(model);
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 10.0, 1e-8);
}

// --- Warm-started dual simplex for node relaxations
// (docs/architecture/LP.md \S1/\S2, MilpSolverOptions::
// warm_start_node_relaxations) -----------------------------------------
//
// Cover cuts disabled and rounding disabled here, matching
// milp_finds_integer_optimum_with_fractional_lp_root's recipe exactly:
// with the cut applied, binary_knapsack's root relaxation becomes
// integral immediately and no node ever reaches depth >= 2, which is
// where the warm path (seeded from a depth-1 node's exported basis)
// would actually be exercised.

SIHPS_TEST(milp_warm_start_matches_cold_start_on_binary_knapsack) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto cold = sihps::solve_milp(binary_knapsack(), options);

    options.warm_start_node_relaxations = true;
    const auto warm = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(cold.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.has_incumbent);
    SIHPS_ASSERT_NEAR(warm.objective_value, cold.objective_value, 1e-8);
    SIHPS_ASSERT_NEAR(warm.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_NEAR(warm.x[0], cold.x[0], 0.0);
    SIHPS_ASSERT_NEAR(warm.x[1], cold.x[1], 0.0);
    // Proves the path was actually exercised, not silently skipped.
    SIHPS_ASSERT_TRUE(warm.warm_started_relaxations > 0);
}

SIHPS_TEST(milp_warm_start_matches_cold_start_on_mixed_nonnegative_packing_row) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto cold = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    options.warm_start_node_relaxations = true;
    const auto warm = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    SIHPS_ASSERT_TRUE(cold.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(warm.objective_value, cold.objective_value, 1e-8);
    SIHPS_ASSERT_NEAR(warm.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_TRUE(warm.root_cover_cuts >= 1);
}

SIHPS_TEST(milp_warm_start_reproduces_tiny_milp_mps_result) {
    const auto model =
        sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/tests/data/tiny_milp.mps");
    const auto problem = sihps::milp_problem_from_mps(model);
    MilpSolverOptions options;
    options.warm_start_node_relaxations = true;
    const auto result = sihps::solve_milp(problem, options);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

// A nonzero fallback count on a small, well-scaled instance would be a
// signal worth investigating, not an accepted steady state -- this pins
// the expectation that it stays at zero here.
SIHPS_TEST(milp_warm_start_fallback_counter_is_zero_or_explained) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.warm_start_node_relaxations = true;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.warm_start_verification_fallbacks == 0);
}
