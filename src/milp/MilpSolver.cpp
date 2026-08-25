#include "MilpSolver.hpp"

#include "../parallel/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sihps {
namespace {

constexpr double kInfinityValue = std::numeric_limits<double>::infinity();

struct BoundChange {
    std::int32_t variable = -1;
    double lower = -kInfinityValue;
    double upper = kInfinityValue;
};

struct SearchNode {
    std::shared_ptr<const SearchNode> parent;
    BoundChange change;
    int depth = 0;
    std::uint64_t order = 0;

    // Branch metadata used to learn pseudocosts when this node's relaxation
    // is solved. It is not used for correctness or pruning.
    std::int32_t branch_variable = -1;
    int branch_direction = 0; // -1: x <= floor(parent x), +1: x >= ceil(parent x)
    double branch_distance = 0.0;

    // A parent's LP lower bound is also a valid lower bound for either child.
    // It is used for best-bound ordering before the child relaxation is run.
    double priority_bound = -kInfinityValue;
};

struct NodeCompare {
    bool operator()(const std::shared_ptr<const SearchNode>& lhs,
                    const std::shared_ptr<const SearchNode>& rhs) const {
        if (lhs->priority_bound != rhs->priority_bound) {
            return lhs->priority_bound > rhs->priority_bound;
        }
        // Deterministic tie-breaks avoid pointer-address ordering, which can
        // change between processes and would make a benchmark irreproducible.
        if (lhs->depth != rhs->depth) return lhs->depth > rhs->depth;
        return lhs->order > rhs->order;
    }
};

using NodeQueue = std::priority_queue<std::shared_ptr<const SearchNode>,
                                      std::vector<std::shared_ptr<const SearchNode>>,
                                      NodeCompare>;

bool within(double value, double target, double tolerance) {
    return std::fabs(value - target) <= tolerance * (1.0 + std::fabs(target));
}

double objective_value(const LpProblem& problem, const std::vector<double>& x) {
    double value = 0.0;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        value += problem.obj[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    }
    return value;
}

void materialize_bounds(const SearchNode& node, const std::vector<double>& root_lower,
                        const std::vector<double>& root_upper, std::vector<double>& lower,
                        std::vector<double>& upper) {
    lower = root_lower;
    upper = root_upper;

    std::vector<const SearchNode*> path;
    for (const SearchNode* current = &node; current != nullptr; current = current->parent.get()) {
        path.push_back(current);
    }
    std::reverse(path.begin(), path.end());
    for (const SearchNode* current : path) {
        if (current->change.variable < 0) continue;
        const auto j = static_cast<std::size_t>(current->change.variable);
        lower[j] = std::max(lower[j], current->change.lower);
        upper[j] = std::min(upper[j], current->change.upper);
    }
}

bool bounds_are_valid(const std::vector<double>& lower, const std::vector<double>& upper) {
    if (lower.size() != upper.size()) return false;
    for (std::size_t j = 0; j < lower.size(); ++j) {
        if (std::isnan(lower[j]) || std::isnan(upper[j]) || lower[j] > upper[j]) return false;
    }
    return true;
}

bool feasible_point(const MilpProblem& problem, const std::vector<double>& x,
                    const std::vector<double>& lower, const std::vector<double>& upper,
                    double feasibility_tolerance, ParallelMode parallel_mode) {
    const LpProblem& lp = problem.relaxation;
    if (x.size() != static_cast<std::size_t>(lp.n_cols())) return false;

    for (std::int32_t j = 0; j < lp.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (!std::isfinite(x[jj])) return false;
        if (x[jj] < lower[jj] - feasibility_tolerance * (1.0 + std::fabs(lower[jj])) ||
            x[jj] > upper[jj] + feasibility_tolerance * (1.0 + std::fabs(upper[jj]))) {
            return false;
        }
    }

    std::vector<double> ax(static_cast<std::size_t>(lp.n_rows()), 0.0);
    if (lp.n_rows() > 0) lp.A.multiply(x.data(), ax.data(), parallel_mode);
    double rhs_norm = 0.0;
    for (double value : lp.rhs) rhs_norm = std::max(rhs_norm, std::fabs(value));
    double row_violation = 0.0;
    for (std::int32_t i = 0; i < lp.n_rows(); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double lo = lp.rhs[ii] - lp.slack_upper[ii];
        const double hi = lp.rhs[ii] - lp.slack_lower[ii];
        if (std::isfinite(lo)) row_violation = std::max(row_violation, lo - ax[ii]);
        if (std::isfinite(hi)) row_violation = std::max(row_violation, ax[ii] - hi);
    }
    return std::max(0.0, row_violation) / (1.0 + rhs_norm) <= feasibility_tolerance;
}

bool integral_point(const MilpProblem& problem, const std::vector<double>& x,
                    double integrality_tolerance) {
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj]) ||
            !within(x[jj], std::round(x[jj]), integrality_tolerance)) {
            return false;
        }
    }
    return true;
}

std::vector<double> rounded_point(const MilpProblem& problem, const std::vector<double>& x,
                                  const std::vector<double>& lower,
                                  const std::vector<double>& upper) {
    std::vector<double> rounded = x;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        double value = std::round(x[jj]);
        if (problem.variable_types[jj] == VariableType::BINARY) {
            value = std::clamp(value, 0.0, 1.0);
        }
        if (std::isfinite(lower[jj])) value = std::max(value, std::ceil(lower[jj]));
        if (std::isfinite(upper[jj])) value = std::min(value, std::floor(upper[jj]));
        rounded[jj] = value;
    }
    return rounded;
}

struct FractionalCandidate {
    std::int32_t variable = -1;
    double fraction = 0.0;
    double fractionality = 0.0;
};

struct CoverCut {
    std::vector<std::int32_t> variables;
};

std::vector<CoverCut> separate_root_cover_cuts(const MilpProblem& problem,
                                                const std::vector<double>& x,
                                                double violation_tolerance,
                                                std::uint32_t limit) {
    std::vector<CoverCut> cuts;
    const LpProblem& lp = problem.relaxation;
    for (std::int32_t i = 0; i < lp.n_rows() && cuts.size() < limit; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double upper = lp.rhs[ii] - lp.slack_lower[ii];
        if (!std::isfinite(upper)) continue;

        struct Term {
            std::int32_t variable;
            double coefficient;
            double value;
        };
        std::vector<Term> terms;
        const std::int32_t begin = lp.A.row_ptr()[i];
        const std::int32_t end = lp.A.row_ptr()[i + 1];
        bool pure_binary_knapsack = true;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const std::int32_t j = lp.A.col_idx()[kk];
            const auto jj = static_cast<std::size_t>(j);
            if (problem.variable_types[jj] != VariableType::BINARY ||
                lp.lower[jj] > 0.0 || lp.upper[jj] < 1.0 || lp.A.values()[kk] <= 0.0) {
                pure_binary_knapsack = false;
                break;
            }
            terms.push_back({j, lp.A.values()[kk], x[jj]});
        }
        if (!pure_binary_knapsack || terms.size() < 2) continue;
        std::sort(terms.begin(), terms.end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.value != rhs.value) return lhs.value > rhs.value;
            return lhs.variable < rhs.variable;
        });

        double coefficient_sum = 0.0;
        double fractional_activity = 0.0;
        CoverCut cut;
        for (const Term& term : terms) {
            coefficient_sum += term.coefficient;
            fractional_activity += term.value;
            cut.variables.push_back(term.variable);
            if (coefficient_sum > upper + violation_tolerance) break;
        }
        if (coefficient_sum <= upper + violation_tolerance || cut.variables.empty() ||
            fractional_activity <= static_cast<double>(cut.variables.size() - 1) +
                                       violation_tolerance) {
            continue;
        }
        cuts.push_back(std::move(cut));
    }
    return cuts;
}

void append_cover_cuts(LpProblem& workspace, const std::vector<CoverCut>& cuts) {
    const std::int32_t old_rows = workspace.n_rows();
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) +
                    std::accumulate(cuts.begin(), cuts.end(), std::size_t{0},
                                    [](std::size_t total, const CoverCut& cut) {
                                        return total + cut.variables.size();
                                    }));
    for (std::int32_t i = 0; i < old_rows; ++i) {
        for (std::int32_t k = workspace.A.row_ptr()[i]; k < workspace.A.row_ptr()[i + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            entries.push_back({i, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
        }
    }
    for (std::size_t cut_index = 0; cut_index < cuts.size(); ++cut_index) {
        const auto row = old_rows + static_cast<std::int32_t>(cut_index);
        for (std::int32_t variable : cuts[cut_index].variables) {
            entries.push_back({row, variable, 1.0});
        }
    }
    workspace.A = CSRMatrix::from_triplets(
        old_rows + static_cast<std::int32_t>(cuts.size()), workspace.n_cols(), entries);
    for (const CoverCut& cut : cuts) {
        workspace.rhs.push_back(static_cast<double>(cut.variables.size() - 1));
        workspace.row_types.push_back('L');
        workspace.slack_lower.push_back(0.0);
        workspace.slack_upper.push_back(kInfinityValue);
    }
}

std::vector<FractionalCandidate> fractional_candidates(const MilpProblem& problem,
                                                        const std::vector<double>& x,
                                                        double integrality_tolerance) {
    std::vector<FractionalCandidate> candidates;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj])) continue;
        const double floor_value = std::floor(x[jj]);
        const double fraction = x[jj] - floor_value;
        if (fraction <= integrality_tolerance || 1.0 - fraction <= integrality_tolerance) {
            continue;
        }
        const double fractionality = std::min(fraction, 1.0 - fraction);
        candidates.push_back({j, fraction, fractionality});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.fractionality != rhs.fractionality) {
            return lhs.fractionality > rhs.fractionality;
        }
        return lhs.variable < rhs.variable;
    });
    return candidates;
}

double current_best_bound(const NodeQueue& queue, bool has_incumbent, double incumbent) {
    if (!queue.empty()) return queue.top()->priority_bound;
    return has_incumbent ? incumbent : kInfinityValue;
}

double relative_gap(bool has_incumbent, double incumbent, double best_bound) {
    if (!has_incumbent || !std::isfinite(best_bound)) return kInfinityValue;
    return std::max(0.0, (incumbent - best_bound) / (1.0 + std::fabs(incumbent)));
}

} // namespace

MilpSolution solve_milp(const MilpProblem& problem, const MilpSolverOptions& options) {
    validate_milp_problem(problem);
    if (options.integrality_tolerance < 0.0 || options.feasibility_tolerance < 0.0 ||
        options.objective_tolerance < 0.0 || options.time_limit_seconds < 0.0) {
        throw std::invalid_argument("MilpSolverOptions: tolerances and limits must be nonnegative");
    }
    if (options.reliability_threshold == 0 &&
        options.branching_rule == MilpBranchingRule::RELIABILITY) {
        throw std::invalid_argument("MilpSolverOptions: reliability threshold must be positive");
    }

    MilpSolution solution;
    // This is the conservative terminal value if every queued relaxation is
    // proven infeasible. It is replaced by an explicit limit/failure status
    // whenever the search stops for any other reason.
    solution.status = MilpStatus::INFEASIBLE;
    const auto start = std::chrono::steady_clock::now();
    const auto timed_out = [&]() {
        return options.time_limit_seconds > 0.0 &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
                   options.time_limit_seconds;
    };

    // One mutable workspace reuses the original sparse matrix for every
    // node. Copying the matrix per node would make a large B&B tree
    // memory-bound before the LP solver had a chance to work.
    LpProblem workspace = problem.relaxation;
    // LP is a minimization engine. Keep the public MILP model in its natural
    // objective sense and normalize only this private relaxation workspace.
    if (problem.maximize) {
        for (double& coefficient : workspace.obj) coefficient = -coefficient;
    }
    std::vector<double> root_lower = workspace.lower;
    std::vector<double> root_upper = workspace.upper;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::BINARY) {
            root_lower[jj] = std::max(root_lower[jj], 0.0);
            root_upper[jj] = std::min(root_upper[jj], 1.0);
        }
    }
    if (!bounds_are_valid(root_lower, root_upper)) {
        solution.status = MilpStatus::INFEASIBLE;
        return solution;
    }

    auto root = std::make_shared<SearchNode>();
    std::uint64_t next_node_order = 1;
    NodeQueue open;
    open.push(root);

    double incumbent = kInfinityValue;
    std::vector<double> incumbent_x;
    bool relaxation_unbounded = false;
    bool root_cuts_separated = false;

    LpSolverOptions relaxation_options = options.lp_options;
    relaxation_options.method = LpMethod::SIMPLEX;
    bool has_integer_variables = false;
    for (VariableType type : problem.variable_types) {
        has_integer_variables |= type != VariableType::CONTINUOUS;
    }

    std::vector<double> down_pseudocost(static_cast<std::size_t>(problem.n_cols()), 0.0);
    std::vector<double> up_pseudocost(static_cast<std::size_t>(problem.n_cols()), 0.0);
    std::vector<std::uint32_t> down_observations(static_cast<std::size_t>(problem.n_cols()), 0);
    std::vector<std::uint32_t> up_observations(static_cast<std::size_t>(problem.n_cols()), 0);

    auto record_pseudocost = [&](const SearchNode& child, double child_bound,
                                 bool infeasible) {
        if (child.branch_variable < 0 || child.branch_distance <= 0.0 ||
            !std::isfinite(child.priority_bound)) {
            return;
        }
        const auto j = static_cast<std::size_t>(child.branch_variable);
        const double unit_cost = infeasible
                                     ? kInfinityValue
                                     : std::max(0.0, child_bound - child.priority_bound) /
                                           child.branch_distance;
        if (child.branch_direction < 0) {
            down_pseudocost[j] += unit_cost;
            ++down_observations[j];
        } else {
            up_pseudocost[j] += unit_cost;
            ++up_observations[j];
        }
    };

    const auto reliable = [&](std::size_t j) {
        return down_observations[j] >= options.reliability_threshold &&
               up_observations[j] >= options.reliability_threshold;
    };
    const auto pseudocost_score = [&](const FractionalCandidate& candidate) {
        const auto j = static_cast<std::size_t>(candidate.variable);
        if (down_observations[j] == 0 || up_observations[j] == 0) return -1.0;
        const double down = (down_pseudocost[j] / down_observations[j]) * candidate.fraction;
        const double up = (up_pseudocost[j] / up_observations[j]) * (1.0 - candidate.fraction);
        return std::min(down, up);
    };

    const auto observe_pseudocost = [&](std::int32_t variable, int direction,
                                        double unit_cost) {
        const auto j = static_cast<std::size_t>(variable);
        if (direction < 0) {
            down_pseudocost[j] += unit_cost;
            ++down_observations[j];
        } else {
            up_pseudocost[j] += unit_cost;
            ++up_observations[j];
        }
    };

    while (!open.empty()) {
        if (timed_out()) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        if (options.node_limit > 0 && solution.nodes_processed >= options.node_limit) {
            solution.status = MilpStatus::NODE_LIMIT;
            break;
        }

        const auto node = open.top();
        open.pop();
        ++solution.nodes_processed;

        if (std::isfinite(incumbent) &&
            node->priority_bound >= incumbent -
                                         options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            ++solution.nodes_pruned;
            continue;
        }

        std::vector<double> lower;
        std::vector<double> upper;
        materialize_bounds(*node, root_lower, root_upper, lower, upper);
        if (!bounds_are_valid(lower, upper)) {
            ++solution.nodes_pruned;
            continue;
        }
        workspace.lower = lower;
        workspace.upper = upper;

        ++solution.lp_relaxations;
        const LpSolution relaxation = solve_lp(workspace, relaxation_options);
        if (relaxation.status == LpStatus::INFEASIBLE) {
            record_pseudocost(*node, node->priority_bound, true);
            ++solution.nodes_pruned;
            continue;
        }
        if (relaxation.status == LpStatus::UNBOUNDED) {
            relaxation_unbounded = true;
            solution.status = has_integer_variables ? MilpStatus::UNBOUNDED_RELAXATION
                                                     : MilpStatus::UNBOUNDED;
            break;
        }
        if (relaxation.status != LpStatus::OPTIMAL ||
            relaxation.x.size() != static_cast<std::size_t>(problem.n_cols())) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        const double lower_bound = relaxation.objective_value;
        record_pseudocost(*node, lower_bound, false);

        // Separate only once at the root in this milestone. The cuts are
        // globally valid cover inequalities for binary variables, so they
        // remain valid in every descendant; a node-local cut pool is a later
        // extension with a separate postsolve/accounting contract.
        if (node->depth == 0 && !root_cuts_separated && options.enable_root_cover_cuts) {
            root_cuts_separated = true;
            const auto cuts = separate_root_cover_cuts(
                problem, relaxation.x, options.cut_violation_tolerance,
                options.max_root_cover_cuts);
            if (!cuts.empty()) {
                append_cover_cuts(workspace, cuts);
                solution.root_cover_cuts = cuts.size();
                open.push(node);
                continue;
            }
        }
        if (std::isfinite(incumbent) &&
            lower_bound >= incumbent -
                               options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            ++solution.nodes_pruned;
            continue;
        }

        const bool candidate_integral =
            integral_point(problem, relaxation.x, options.integrality_tolerance);
        const std::vector<double> rounded = rounded_point(problem, relaxation.x, lower, upper);
        if (candidate_integral || options.use_rounding_heuristic) {
            if (feasible_point(problem, rounded, lower, upper, options.feasibility_tolerance,
                               options.lp_options.parallel_mode) &&
                integral_point(problem, rounded, options.integrality_tolerance)) {
                const double candidate_objective = objective_value(workspace, rounded);
                if (!std::isfinite(incumbent) ||
                    candidate_objective < incumbent -
                                             options.objective_tolerance *
                                                 (1.0 + std::fabs(incumbent))) {
                    incumbent = candidate_objective;
                    incumbent_x = rounded;
                    ++solution.incumbent_updates;
                }
            } else if (candidate_integral) {
                // The LP claimed an integral point, but the exact integer
                // candidate did not clear the original-model gate. Do not
                // branch on a point that should already be terminal: this is
                // a numerical inconsistency, not proof of infeasibility.
                solution.status = MilpStatus::NUMERICAL_FAILURE;
                break;
            }
        }

        if (candidate_integral) continue;

        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            const auto jx = static_cast<std::size_t>(j);
            if (problem.variable_types[jx] != VariableType::CONTINUOUS &&
                !std::isfinite(relaxation.x[jx])) {
                solution.status = MilpStatus::NUMERICAL_FAILURE;
                break;
            }
        }
        if (solution.status == MilpStatus::NUMERICAL_FAILURE) {
            break;
        }

        const std::vector<FractionalCandidate> candidates =
            fractional_candidates(problem, relaxation.x, options.integrality_tolerance);
        if (candidates.empty()) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        std::int32_t branch_variable = candidates.front().variable;
        if (options.branching_rule == MilpBranchingRule::PSEUDOCOST ||
            options.branching_rule == MilpBranchingRule::RELIABILITY) {
            if (options.branching_rule == MilpBranchingRule::RELIABILITY) {
                std::uint32_t probes = 0;
                for (const FractionalCandidate& candidate : candidates) {
                    const auto j = static_cast<std::size_t>(candidate.variable);
                    if (reliable(j)) continue;
                    if (probes >= options.strong_branching_candidates) break;
                    if (timed_out()) {
                        open.push(node);
                        solution.status = MilpStatus::TIME_LIMIT;
                        break;
                    }

                    const double floor_probe = std::floor(relaxation.x[j]);
                    const double ceil_probe = std::ceil(relaxation.x[j]);
                    const double down_distance = candidate.fraction;
                    const double up_distance = 1.0 - candidate.fraction;

                    auto probe_child = [&](int direction, double bound,
                                           double distance) -> std::pair<bool, double> {
                        std::vector<double> probe_lower = lower;
                        std::vector<double> probe_upper = upper;
                        if (direction < 0) {
                            probe_upper[j] = std::min(probe_upper[j], bound);
                        } else {
                            probe_lower[j] = std::max(probe_lower[j], bound);
                        }
                        if (!bounds_are_valid(probe_lower, probe_upper)) {
                            return {true, kInfinityValue};
                        }
                        workspace.lower = probe_lower;
                        workspace.upper = probe_upper;
                        ++solution.lp_relaxations;
                        ++solution.strong_branching_probes;
                        const LpSolution probe = solve_lp(workspace, relaxation_options);
                        workspace.lower = lower;
                        workspace.upper = upper;
                        if (probe.status == LpStatus::INFEASIBLE) {
                            return {true, kInfinityValue};
                        }
                        if (probe.status != LpStatus::OPTIMAL || distance <= 0.0) {
                            return {false, 0.0};
                        }
                        return {true, std::max(0.0, probe.objective_value - lower_bound) /
                                            distance};
                    };

                    const auto down_probe = probe_child(-1, floor_probe, down_distance);
                    const auto up_probe = probe_child(+1, ceil_probe, up_distance);
                    if (down_probe.first) {
                        observe_pseudocost(candidate.variable, -1, down_probe.second);
                    }
                    if (up_probe.first) {
                        observe_pseudocost(candidate.variable, +1, up_probe.second);
                    }
                    ++probes;
                }
                if (solution.status == MilpStatus::TIME_LIMIT) break;
            }

            double best_score = -1.0;
            for (const FractionalCandidate& candidate : candidates) {
                const auto j = static_cast<std::size_t>(candidate.variable);
                if (options.branching_rule == MilpBranchingRule::RELIABILITY &&
                    !reliable(j)) {
                    continue;
                }
                const double score = pseudocost_score(candidate);
                if (score > best_score) {
                    best_score = score;
                    branch_variable = candidate.variable;
                }
            }
        }

        const auto jj = static_cast<std::size_t>(branch_variable);
        const double floor_value = std::floor(relaxation.x[jj]);
        const double ceil_value = std::ceil(relaxation.x[jj]);
        if (floor_value >= ceil_value || !std::isfinite(floor_value) ||
            !std::isfinite(ceil_value)) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        auto left = std::make_shared<SearchNode>();
        left->parent = node;
        left->depth = node->depth + 1;
        left->order = next_node_order++;
        left->priority_bound = lower_bound;
        left->change.variable = branch_variable;
        left->change.upper = floor_value;
        left->branch_variable = branch_variable;
        left->branch_direction = -1;
        left->branch_distance = relaxation.x[jj] - floor_value;

        auto right = std::make_shared<SearchNode>();
        right->parent = node;
        right->depth = node->depth + 1;
        right->order = next_node_order++;
        right->priority_bound = lower_bound;
        right->change.variable = branch_variable;
        right->change.lower = ceil_value;
        right->branch_variable = branch_variable;
        right->branch_direction = +1;
        right->branch_distance = ceil_value - relaxation.x[jj];

        open.push(std::move(left));
        open.push(std::move(right));
    }

    solution.has_incumbent = std::isfinite(incumbent);
    if (solution.has_incumbent) {
        solution.x = std::move(incumbent_x);
        solution.objective_value = problem.maximize ? -incumbent : incumbent;
    }
    const double minimization_bound = current_best_bound(open, solution.has_incumbent, incumbent);
    solution.best_bound = problem.maximize ? -minimization_bound : minimization_bound;
    solution.relative_gap = relative_gap(solution.has_incumbent, incumbent, minimization_bound);

    if (relaxation_unbounded) return solution;
    if (solution.status == MilpStatus::TIME_LIMIT || solution.status == MilpStatus::NODE_LIMIT ||
        solution.status == MilpStatus::NUMERICAL_FAILURE) {
        return solution;
    }
    if (open.empty()) {
        solution.status = solution.has_incumbent ? MilpStatus::OPTIMAL : MilpStatus::INFEASIBLE;
    } else {
        solution.status = MilpStatus::NUMERICAL_FAILURE;
    }
    return solution;
}

} // namespace sihps
