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
    // Round every MILP node's integer bounds inward before its LP
    // relaxation. This is unconditionally sound: an integer-feasible point
    // already satisfies the rounded box. It also makes fractional bounds
    // derived by branching or propagation explicit to the LP engine.
    bool enable_integer_bound_rounding = true;
    bool use_rounding_heuristic = true;
    // Deterministic LP diving only proposes incumbents; it never changes
    // node pruning or optimality certification. It is limited so a failed
    // dive cannot consume an unbounded fraction of the B&B budget.
    bool use_diving_heuristic = true;
    std::uint32_t diving_max_depth = 32;
    std::uint32_t diving_max_lp_relaxations = 64;
    bool use_local_improvement = true;
    std::uint32_t local_improvement_passes = 3;
    std::uint32_t local_improvement_max_trials = 128;
    MilpBranchingRule branching_rule = MilpBranchingRule::RELIABILITY;
    std::uint32_t reliability_threshold = 2;
    std::uint32_t strong_branching_candidates = 4;
    bool enable_root_cover_cuts = true;
    std::uint32_t max_root_cover_cuts = 64;
    double cut_violation_tolerance = 1e-7;

    // Warm-started dual simplex for node relaxations (docs/architecture/
    // LP.md \S1/\S2, MILP.md's stated prerequisite): a non-root node
    // solves its LP relaxation by seating its parent's terminal basis and
    // repairing primal feasibility, instead of a full cold solve_lp call.
    // Root and its direct children never warm
    // start (the root solve goes through solve_lp's presolve, which a
    // warm basis is not guaranteed to remain valid under).
    // Enabled by default after the optimized MIPLIB KPI refresh. Callers can
    // disable it for cold-start ablations or when node-local presolve is more
    // valuable than basis reuse on a particular model family.
    bool warm_start_node_relaxations = true;
    // Detect modularly impossible all-integer equality rows before invoking
    // the LP relaxation. Non-integral coefficients, bounds, or continuous
    // variables make the check conservatively skip the row.
    bool enable_integer_gcd_tightening = true;
    bool enable_integer_equality_propagation = true;
    bool enable_integer_inequality_rounding = true;
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
    std::uint64_t cover_cuts = 0;
    std::uint64_t incumbent_updates = 0;
    std::uint64_t diving_heuristic_lp_relaxations = 0;
    std::uint64_t local_improvement_lp_relaxations = 0;

    // Populated only when warm_start_node_relaxations is true: how many
    // non-root node relaxations actually took the warm path, and how many
    // of those had to fall back to a cold solve after a verification
    // failure. A nonzero fallback count on a small, well-scaled instance
    // is a signal worth investigating, not an accepted steady state.
    std::uint64_t warm_started_relaxations = 0;
    std::uint64_t warm_start_verification_fallbacks = 0;
    std::uint64_t integer_gcd_prunes = 0;
    std::uint64_t integer_propagation_prunes = 0;
    std::uint64_t integer_bound_tightenings = 0;
    std::uint64_t integer_rhs_tightenings = 0;
};

MilpSolution solve_milp(const MilpProblem& problem,
                        const MilpSolverOptions& options = {});

} // namespace sihps
