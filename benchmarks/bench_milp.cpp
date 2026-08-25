// MILP correctness/performance benchmark with a dynamic-programming oracle
// for the binary knapsack instance. The oracle is independent of the solver
// and validates both the objective and the reported OPTIMAL status.

#include "milp/MilpSolver.hpp"
#include "sparse/Triplet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace sihps;

namespace {

MilpProblem make_knapsack(const std::vector<int>& weights, const std::vector<int>& profits,
                          int capacity) {
    MilpProblem problem;
    const auto n = static_cast<std::int32_t>(weights.size());
    std::vector<Triplet> entries;
    entries.reserve(weights.size());
    for (std::int32_t j = 0; j < n; ++j) {
        entries.push_back({0, j, static_cast<double>(weights[static_cast<std::size_t>(j)])});
    }
    problem.relaxation.A = CSRMatrix::from_triplets(1, n, entries);
    problem.relaxation.obj.resize(weights.size());
    for (std::size_t j = 0; j < weights.size(); ++j) {
        problem.relaxation.obj[j] = -static_cast<double>(profits[j]);
    }
    problem.relaxation.rhs = {static_cast<double>(capacity)};
    problem.relaxation.row_types = {'L'};
    problem.relaxation.lower.assign(weights.size(), 0.0);
    problem.relaxation.upper.assign(weights.size(), 1.0);
    apply_default_row_bounds(problem.relaxation);
    problem.variable_types.assign(weights.size(), VariableType::BINARY);
    validate_milp_problem(problem);
    return problem;
}

int knapsack_oracle(const std::vector<int>& weights, const std::vector<int>& profits,
                   int capacity) {
    std::vector<int> best(static_cast<std::size_t>(capacity) + 1, 0);
    for (std::size_t j = 0; j < weights.size(); ++j) {
        for (int c = capacity; c >= weights[j]; --c) {
            best[static_cast<std::size_t>(c)] =
                std::max(best[static_cast<std::size_t>(c)],
                         best[static_cast<std::size_t>(c - weights[j])] + profits[j]);
        }
    }
    return best[static_cast<std::size_t>(capacity)];
}

} // namespace

int main() {
    const MilpProblem fractional_root = make_knapsack({6, 5}, {10, 9}, 7);
    MilpSolverOptions fractional_options;
    fractional_options.use_rounding_heuristic = false;
    fractional_options.enable_root_cover_cuts = false;
    const auto fractional_start = std::chrono::steady_clock::now();
    const MilpSolution fractional_result = solve_milp(fractional_root, fractional_options);
    const double fractional_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fractional_start).count();
    const bool fractional_pass = fractional_result.status == MilpStatus::OPTIMAL &&
                                 std::fabs(fractional_result.objective_value + 10.0) <= 1e-8;
    std::printf("fractional_root status=%s objective=%.12g expected=-10 nodes=%llu probes=%llu "
                "cuts=%llu time=%.6fs oracle_check=%s\n",
                fractional_result.status == MilpStatus::OPTIMAL ? "OPTIMAL" : "NOT_OPTIMAL",
                fractional_result.objective_value,
                static_cast<unsigned long long>(fractional_result.nodes_processed),
                static_cast<unsigned long long>(fractional_result.strong_branching_probes),
                static_cast<unsigned long long>(fractional_result.root_cover_cuts),
                fractional_seconds, fractional_pass ? "PASS" : "FAIL");

    const std::vector<int> weights = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                                      5,  9,  15, 21, 27, 33, 39, 45, 51, 57, 63, 69};
    const std::vector<int> profits = {19, 31, 37, 43, 47, 53, 61, 67, 71, 73, 79, 83,
                                      17, 29, 41, 59, 64, 68, 72, 81, 86, 91, 97, 101};
    constexpr int capacity = 260;

    const MilpProblem problem = make_knapsack(weights, profits, capacity);
    const double expected = -static_cast<double>(knapsack_oracle(weights, profits, capacity));

    MilpSolverOptions options;
    // Disable the optional heuristic for this baseline so the reported node
    // count measures the branch-and-bound core rather than a lucky rounded
    // incumbent at the root.
    options.use_rounding_heuristic = false;
    const auto t0 = std::chrono::steady_clock::now();
    const MilpSolution result = solve_milp(problem, options);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double error = std::fabs(result.objective_value - expected);
    const bool pass = fractional_pass && result.status == MilpStatus::OPTIMAL && error <= 1e-8 &&
                      result.has_incumbent;
    std::printf("binary_knapsack items=%zu capacity=%d\n", weights.size(), capacity);
    std::printf("status=%s objective=%.12g expected=%.12g abs_error=%.3e\n",
                result.status == MilpStatus::OPTIMAL ? "OPTIMAL" : "NOT_OPTIMAL",
                result.objective_value, expected, error);
    std::printf("nodes=%llu lp_relaxations=%llu probes=%llu cuts=%llu pruned=%llu "
                "incumbents=%llu time=%.6fs\n",
                static_cast<unsigned long long>(result.nodes_processed),
                static_cast<unsigned long long>(result.lp_relaxations),
                static_cast<unsigned long long>(result.strong_branching_probes),
                static_cast<unsigned long long>(result.root_cover_cuts),
                static_cast<unsigned long long>(result.nodes_pruned),
                static_cast<unsigned long long>(result.incumbent_updates), seconds);
    std::printf("oracle_check=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
