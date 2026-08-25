#pragma once

#include "MilpProblem.hpp"
#include "../lp/LpSolver.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

enum class MilpStatus {
    OPTIMAL,
    INFEASIBLE,
    UNBOUNDED,
    UNBOUNDED_RELAXATION,
    NODE_LIMIT,
    TIME_LIMIT,
    NUMERICAL_FAILURE
};

enum class MilpBranchingRule { MOST_FRACTIONAL, PSEUDOCOST, RELIABILITY };

struct MilpSolverOptions {
    // MILP bounds must be certified LP optima. The solver therefore forces
    // SIMPLEX for relaxations even if a caller's general LP preference is
    // HYBRID or FIRST_ORDER; approximate first-order points are never used
    // as proof bounds.
    LpSolverOptions lp_options;

    std::uint64_t node_limit = 0; // zero means unlimited
    double time_limit_seconds = 0.0; // zero means unlimited
    double integrality_tolerance = 1e-7;
    double feasibility_tolerance = 1e-6;
    double objective_tolerance = 1e-8;
    bool use_rounding_heuristic = true;
    MilpBranchingRule branching_rule = MilpBranchingRule::RELIABILITY;
    std::uint32_t reliability_threshold = 2;
    std::uint32_t strong_branching_candidates = 4;
    bool enable_root_cover_cuts = true;
    std::uint32_t max_root_cover_cuts = 64;
    double cut_violation_tolerance = 1e-7;
};

struct MilpSolution {
    MilpStatus status = MilpStatus::NUMERICAL_FAILURE;
    bool has_incumbent = false;
    std::vector<double> x; // original variable space; empty without incumbent
    double objective_value = 0.0;
    double best_bound = 0.0;
    double relative_gap = 0.0;

    std::uint64_t nodes_processed = 0;
    std::uint64_t nodes_pruned = 0;
    std::uint64_t lp_relaxations = 0;
    std::uint64_t strong_branching_probes = 0;
    std::uint64_t root_cover_cuts = 0;
    std::uint64_t incumbent_updates = 0;
};

MilpSolution solve_milp(const MilpProblem& problem,
                        const MilpSolverOptions& options = {});

} // namespace sihps
