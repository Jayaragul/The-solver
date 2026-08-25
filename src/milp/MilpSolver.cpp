#include "MilpSolver.hpp"

#include "../parallel/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
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

std::int32_t choose_branch_variable(const MilpProblem& problem, const std::vector<double>& x,
                                    double integrality_tolerance) {
    std::int32_t best = -1;
    double best_fractionality = -1.0;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj])) return -2;
        const double floor_value = std::floor(x[jj]);
        const double fraction = x[jj] - floor_value;
        if (fraction <= integrality_tolerance || 1.0 - fraction <= integrality_tolerance) {
            continue;
        }
        const double fractionality = std::min(fraction, 1.0 - fraction);
        if (fractionality > best_fractionality) {
            best_fractionality = fractionality;
            best = j;
        }
    }
    return best;
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
    if (options.branching_rule != MilpBranchingRule::MOST_FRACTIONAL) {
        throw std::invalid_argument("MilpSolverOptions: unsupported branching rule");
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

    LpSolverOptions relaxation_options = options.lp_options;
    relaxation_options.method = LpMethod::SIMPLEX;
    bool has_integer_variables = false;
    for (VariableType type : problem.variable_types) {
        has_integer_variables |= type != VariableType::CONTINUOUS;
    }

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

        const std::int32_t branch_variable =
            choose_branch_variable(problem, relaxation.x, options.integrality_tolerance);
        if (branch_variable == -2) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }
        if (branch_variable < 0) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
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

        auto right = std::make_shared<SearchNode>();
        right->parent = node;
        right->depth = node->depth + 1;
        right->order = next_node_order++;
        right->priority_bound = lower_bound;
        right->change.variable = branch_variable;
        right->change.lower = ceil_value;

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
