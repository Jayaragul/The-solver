#include "MilpSolver.hpp"

#include "../parallel/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
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

void round_integer_bounds(const MilpProblem& problem, std::vector<double>& lower,
                          std::vector<double>& upper) {
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (std::isfinite(lower[jj])) lower[jj] = std::ceil(lower[jj]);
        if (std::isfinite(upper[jj])) upper[jj] = std::floor(upper[jj]);
    }
}

bool integer_equality_gcd_infeasible(const MilpProblem& problem,
                                     const std::vector<double>& lower) {
    const LpProblem& lp = problem.relaxation;
    constexpr double kIntegerTolerance = 1e-9;
    for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        if (lp.row_types[static_cast<std::size_t>(row)] != 'E') continue;
        const auto begin = lp.A.row_ptr()[row];
        const auto end = lp.A.row_ptr()[row + 1];
        std::int64_t divisor = 0;
        long double residual = lp.rhs[static_cast<std::size_t>(row)];
        bool applicable = true;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto j = static_cast<std::size_t>(lp.A.col_idx()[k]);
            if (problem.variable_types[j] == VariableType::CONTINUOUS ||
                !std::isfinite(lower[j])) {
                applicable = false;
                break;
            }
            const double coefficient = lp.A.values()[k];
            const double integral_coefficient = std::round(coefficient);
            if (std::fabs(coefficient - integral_coefficient) > kIntegerTolerance ||
                std::fabs(lower[j] - std::round(lower[j])) > kIntegerTolerance ||
                std::fabs(integral_coefficient) >
                    static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
                applicable = false;
                break;
            }
            residual -= static_cast<long double>(integral_coefficient) * lower[j];
            divisor = std::gcd(divisor,
                               static_cast<std::int64_t>(std::llabs(
                                   static_cast<std::int64_t>(integral_coefficient))));
        }
        if (!applicable || divisor <= 1) continue;
        const long double nearest = std::round(residual);
        if (std::fabs(static_cast<double>(residual - nearest)) > kIntegerTolerance) {
            return true;
        }
        if (std::fabs(static_cast<double>(nearest)) >
            static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            continue;
        }
        if (std::llabs(static_cast<std::int64_t>(nearest)) % divisor != 0) return true;
    }
    return false;
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

bool binary_domain(const MilpProblem& problem, const LpProblem& lp, std::int32_t variable) {
    const auto j = static_cast<std::size_t>(variable);
    if (problem.variable_types[j] == VariableType::BINARY) return true;
    // MPS commonly encodes binary variables as INTEGER inside INTORG/INTEND
    // with explicit [0,1] bounds. Treating those as binary is a semantic
    // classification for valid cover separation, not a relaxation change.
    return problem.variable_types[j] == VariableType::INTEGER && lp.lower[j] >= 0.0 &&
           lp.upper[j] <= 1.0;
}

std::vector<CoverCut> separate_cover_cuts(const MilpProblem& problem,
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
        bool valid_cover_row = true;
        double noncover_minimum = 0.0;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const std::int32_t j = lp.A.col_idx()[kk];
            const auto jj = static_cast<std::size_t>(j);
            const double coefficient = lp.A.values()[kk];
            const double lower_term = coefficient * lp.lower[jj];
            const double upper_term = coefficient * lp.upper[jj];
            const double minimum_term = std::min(lower_term, upper_term);
            if (!std::isfinite(minimum_term)) {
                // A term with an unbounded contribution below could cancel
                // the cover activity, so no cover inequality is inferred.
                valid_cover_row = false;
                break;
            }
            if (binary_domain(problem, lp, j) && coefficient > 0.0) {
                terms.push_back({j, coefficient, x[jj]});
            } else {
                noncover_minimum += minimum_term;
            }
        }
        if (!valid_cover_row || terms.size() < 2) continue;
        const double effective_upper = upper - noncover_minimum;
        std::vector<std::vector<Term>> orderings;
        orderings.push_back(terms);
        orderings.push_back(terms);
        orderings.push_back(terms);
        orderings.push_back(terms);
        std::sort(orderings[0].begin(), orderings[0].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.value != rhs.value) return lhs.value > rhs.value;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[1].begin(), orderings[1].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.coefficient != rhs.coefficient) return lhs.coefficient > rhs.coefficient;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[2].begin(), orderings[2].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.coefficient != rhs.coefficient) return lhs.coefficient < rhs.coefficient;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[3].begin(), orderings[3].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.value != rhs.value) return lhs.value < rhs.value;
            return lhs.variable < rhs.variable;
        });

        for (auto& ordering : orderings) {
            if (cuts.size() >= limit) break;
            double coefficient_sum = 0.0;
            CoverCut cut;
            for (const Term& term : ordering) {
                coefficient_sum += term.coefficient;
                cut.variables.push_back(term.variable);
                if (coefficient_sum > effective_upper + violation_tolerance) break;
            }
            if (coefficient_sum <= effective_upper + violation_tolerance ||
                cut.variables.empty()) {
                continue;
            }

            // Remove redundant cover members. The resulting inequality is
            // still valid, and minimal covers are generally stronger than a
            // greedy nonminimal superset.
            for (std::size_t p = 0; p < cut.variables.size();) {
                const auto variable = cut.variables[p];
                double without = 0.0;
                for (const Term& term : terms) {
                    if (std::find(cut.variables.begin(), cut.variables.end(), term.variable) !=
                            cut.variables.end() &&
                        term.variable != variable) {
                        without += term.coefficient;
                    }
                }
                if (without > effective_upper + violation_tolerance) {
                    cut.variables.erase(cut.variables.begin() + static_cast<std::ptrdiff_t>(p));
                } else {
                    ++p;
                }
            }

            double fractional_activity = 0.0;
            for (std::int32_t variable : cut.variables) {
                fractional_activity += x[static_cast<std::size_t>(variable)];
            }
            if (cut.variables.size() < 2 ||
                fractional_activity <= static_cast<double>(cut.variables.size() - 1) +
                                           violation_tolerance) {
                continue;
            }
            std::sort(cut.variables.begin(), cut.variables.end());
            const bool duplicate = std::any_of(
                cuts.begin(), cuts.end(), [&](const CoverCut& existing) {
                    return existing.variables == cut.variables;
                });
            if (!duplicate) cuts.push_back(std::move(cut));
        }
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

    // Warm-started dual simplex for node relaxations
    // (docs/architecture/LP.md \S1/\S2). Keyed by SearchNode::order,
    // populated when a node's children are created and consumed-and-erased
    // the moment that child is popped -- NOT a SearchNode field, since
    // SearchNode::parent already keeps the whole ancestor chain alive for
    // the rest of the search, and a Basis stored there would outlive its
    // usefulness. This bounds the map to roughly the current queue width
    // rather than the size of the whole tree.
    std::unordered_map<std::uint64_t, std::shared_ptr<const Simplex::Basis>> pending_basis;
    // Ruiz factors for workspace.A, computed once (lazily, on first use)
    // AFTER root cover cuts have settled its final shape, and reused by
    // every subsequent node's direct Simplex construction. workspace.A
    // never changes again once cuts are separated (only lower_/upper_ do,
    // one variable at a time), so recomputing this per node would spend
    // exactly the cost warm-starting exists to avoid.
    bool node_scale_ready = false;
    ScaleFactors node_scale;

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

    const auto consider_incumbent = [&](const std::vector<double>& candidate,
                                        const std::vector<double>& candidate_lower,
                                        const std::vector<double>& candidate_upper) {
        if (!feasible_point(problem, candidate, candidate_lower, candidate_upper,
                            options.feasibility_tolerance, options.lp_options.parallel_mode) ||
            !integral_point(problem, candidate, options.integrality_tolerance)) {
            return false;
        }
        const double candidate_objective = objective_value(workspace, candidate);
        if (!std::isfinite(incumbent) ||
            candidate_objective < incumbent -
                                     options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            incumbent = candidate_objective;
            incumbent_x = candidate;
            ++solution.incumbent_updates;
            return true;
        }
        return false;
    };

    bool heuristic_timeout = false;
    const auto attempt_lp_dive = [&](const LpSolution& starting,
                                     const std::vector<double>& starting_lower,
                                     const std::vector<double>& starting_upper) {
        if (!options.use_diving_heuristic || options.diving_max_depth == 0 ||
            options.diving_max_lp_relaxations == 0 || std::isfinite(incumbent)) {
            return;
        }

        LpSolution current = starting;
        std::vector<double> dive_lower = starting_lower;
        std::vector<double> dive_upper = starting_upper;
        std::uint32_t dive_relaxations = 0;
        for (std::uint32_t depth = 0; depth < options.diving_max_depth; ++depth) {
            if (timed_out()) {
                heuristic_timeout = true;
                return;
            }
            if (integral_point(problem, current.x, options.integrality_tolerance)) {
                consider_incumbent(current.x, dive_lower, dive_upper);
                return;
            }
            const auto candidates =
                fractional_candidates(problem, current.x, options.integrality_tolerance);
            if (candidates.empty()) return;

            const FractionalCandidate& candidate = candidates.front();
            const auto j = static_cast<std::size_t>(candidate.variable);
            const double floor_value = std::floor(current.x[j]);
            const double ceil_value = std::ceil(current.x[j]);
            // Prefer the side indicated by the objective, then try the other
            // side if the preferred LP is infeasible. This is a deterministic
            // objective-guided dive, not a relaxation bound used for proof.
            const int preferred_direction = workspace.obj[j] < 0.0 ? +1 : -1;
            const auto solve_dive_child = [&](int direction) -> bool {
                std::vector<double> child_lower = dive_lower;
                std::vector<double> child_upper = dive_upper;
                if (direction < 0) {
                    child_upper[j] = std::min(child_upper[j], floor_value);
                } else {
                    child_lower[j] = std::max(child_lower[j], ceil_value);
                }
                if (!bounds_are_valid(child_lower, child_upper)) return false;
                workspace.lower = child_lower;
                workspace.upper = child_upper;
                ++solution.lp_relaxations;
                ++solution.diving_heuristic_lp_relaxations;
                ++dive_relaxations;
                const LpSolution child = solve_lp(workspace, relaxation_options);
                workspace.lower = dive_lower;
                workspace.upper = dive_upper;
                if (child.status != LpStatus::OPTIMAL ||
                    child.x.size() != static_cast<std::size_t>(problem.n_cols())) {
                    return false;
                }
                current = child;
                dive_lower = std::move(child_lower);
                dive_upper = std::move(child_upper);
                return true;
            };

            if (dive_relaxations >= options.diving_max_lp_relaxations ||
                (!solve_dive_child(preferred_direction) &&
                 (dive_relaxations >= options.diving_max_lp_relaxations ||
                  !solve_dive_child(-preferred_direction)))) {
                return;
            }
        }
        if (integral_point(problem, current.x, options.integrality_tolerance)) {
            consider_incumbent(current.x, dive_lower, dive_upper);
        }
    };

    const auto attempt_local_improvement = [&](const std::vector<double>& node_lower,
                                               const std::vector<double>& node_upper) {
        if (!options.use_local_improvement || options.local_improvement_passes == 0 ||
            options.local_improvement_max_trials == 0 || !std::isfinite(incumbent)) {
            return;
        }

        std::vector<double> current = incumbent_x;
        for (std::uint32_t pass = 0; pass < options.local_improvement_passes; ++pass) {
            bool improved = false;
            std::uint32_t trials = 0;
            for (std::int32_t j = 0; j < problem.n_cols() &&
                                      trials < options.local_improvement_max_trials;
                 ++j) {
                const auto jj = static_cast<std::size_t>(j);
                if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
                if (!std::isfinite(current[jj])) continue;

                std::vector<double> trial_lower = node_lower;
                std::vector<double> trial_upper = node_upper;
                for (std::int32_t k = 0; k < problem.n_cols(); ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    if (problem.variable_types[kk] == VariableType::CONTINUOUS) continue;
                    const double fixed = std::round(current[kk]);
                    trial_lower[kk] = std::max(trial_lower[kk], fixed);
                    trial_upper[kk] = std::min(trial_upper[kk], fixed);
                }

                const double current_value = std::round(current[jj]);
                double trial_value = current_value;
                if (binary_domain(problem, problem.relaxation, j)) {
                    trial_value = current_value <= 0.5 ? 1.0 : 0.0;
                } else {
                    const double up = current_value + 1.0;
                    const double down = current_value - 1.0;
                    if (up <= trial_upper[jj]) trial_value = up;
                    else if (down >= trial_lower[jj]) trial_value = down;
                    else continue;
                }
                trial_lower[jj] = std::max(trial_lower[jj], trial_value);
                trial_upper[jj] = std::min(trial_upper[jj], trial_value);
                if (!bounds_are_valid(trial_lower, trial_upper)) continue;

                LpProblem local = workspace;
                local.lower = trial_lower;
                local.upper = trial_upper;
                ++trials;
                ++solution.lp_relaxations;
                ++solution.local_improvement_lp_relaxations;
                const LpSolution local_solution = solve_lp(local, relaxation_options);
                if (local_solution.status != LpStatus::OPTIMAL ||
                    local_solution.x.size() != static_cast<std::size_t>(problem.n_cols())) {
                    continue;
                }
                const double before = incumbent;
                if (consider_incumbent(local_solution.x, node_lower, node_upper) &&
                    incumbent < before) {
                    current = incumbent_x;
                    improved = true;
                }
                if (timed_out()) return;
            }
            if (!improved) break;
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

        std::shared_ptr<const Simplex::Basis> node_parent_basis;
        if (options.warm_start_node_relaxations) {
            auto pending_it = pending_basis.find(node->order);
            if (pending_it != pending_basis.end()) {
                node_parent_basis = pending_it->second;
                pending_basis.erase(pending_it);
            }
        }

        if (std::isfinite(incumbent) &&
            node->priority_bound >= incumbent -
                                         options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            ++solution.nodes_pruned;
            continue;
        }

        std::vector<double> lower;
        std::vector<double> upper;
        materialize_bounds(*node, root_lower, root_upper, lower, upper);
        if (options.enable_integer_bound_rounding) {
            round_integer_bounds(problem, lower, upper);
        }
        if (!bounds_are_valid(lower, upper)) {
            ++solution.nodes_pruned;
            continue;
        }
        if (options.enable_integer_gcd_tightening &&
            integer_equality_gcd_infeasible(problem, lower)) {
            ++solution.nodes_pruned;
            ++solution.integer_gcd_prunes;
            continue;
        }
        workspace.lower = lower;
        workspace.upper = upper;

        ++solution.lp_relaxations;
        LpSolution relaxation;
        std::shared_ptr<const Simplex::Basis> node_basis;
        if (node->depth == 0 || !options.warm_start_node_relaxations) {
            // Root always takes this path: its solve goes through
            // solve_lp's presolve, and a warm basis is only valid for a
            // child that solves over the SAME augmented column space --
            // not guaranteed once presolve's bound-dependent reductions
            // are in the picture. Every node also takes this path when the
            // feature is off, which is exactly today's behavior.
            relaxation = solve_lp(workspace, relaxation_options);
        } else {
            if (!node_scale_ready) {
                node_scale = relaxation_options.use_ruiz_scaling
                                 ? compute_ruiz_scaling(workspace.A)
                                 : ScaleFactors::identity(workspace.n_rows(), workspace.n_cols());
                node_scale_ready = true;
            }
            Simplex simplex(workspace, relaxation_options.backend,
                             relaxation_options.use_ruiz_scaling, relaxation_options.pricing_rule,
                             LpAlgorithm::AUTO, relaxation_options.parallel_mode, &node_scale);
            if (relaxation_options.simplex_time_budget_seconds > 0.0) {
                simplex.set_time_budget(relaxation_options.simplex_time_budget_seconds);
            }
            if (node_parent_basis) simplex.set_warm_start_basis(node_parent_basis.get());

            const LpResult lp = simplex.solve();
            relaxation.status = lp.status;
            relaxation.x = lp.x;
            relaxation.objective_value = lp.objective_value;

            if (lp.used_warm_start) ++solution.warm_started_relaxations;
            if (lp.warm_start_attempted && !lp.used_warm_start) {
                ++solution.warm_start_verification_fallbacks;
            }
            if (lp.status == LpStatus::OPTIMAL) {
                node_basis = std::make_shared<const Simplex::Basis>(simplex.export_basis());
            }
        }
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

        if (node->depth == 0 && !root_cuts_separated && options.enable_root_cover_cuts) {
            root_cuts_separated = true;
            const auto cuts = separate_cover_cuts(
                problem, relaxation.x, options.cut_violation_tolerance,
                options.max_root_cover_cuts);
            if (!cuts.empty()) {
                append_cover_cuts(workspace, cuts);
                solution.root_cover_cuts = cuts.size();
                solution.cover_cuts = cuts.size();
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
            const bool accepted = consider_incumbent(rounded, lower, upper);
            if (!accepted && candidate_integral) {
                // The LP claimed an integral point, but the exact integer
                // candidate did not clear the original-model gate. Do not
                // branch on a point that should already be terminal: this is
                // a numerical inconsistency, not proof of infeasibility.
                solution.status = MilpStatus::NUMERICAL_FAILURE;
                break;
            }
        }

        if (candidate_integral) continue;

        // Dive at the root and at only the first few levels when no
        // incumbent exists. Repeating a failed dive at every deep node can
        // spend more LP work on heuristics than on certified search.
        if (node->depth <= 2 && !std::isfinite(incumbent)) {
            attempt_lp_dive(relaxation, lower, upper);
            if (heuristic_timeout) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }
        if (node->depth == 0 && std::isfinite(incumbent)) {
            attempt_local_improvement(lower, upper);
            if (timed_out()) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }

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
                std::vector<std::size_t> probe_candidates;
                for (const FractionalCandidate& candidate : candidates) {
                    const auto j = static_cast<std::size_t>(candidate.variable);
                    if (reliable(j)) continue;
                    if (probe_candidates.size() >= options.strong_branching_candidates) break;
                    probe_candidates.push_back(static_cast<std::size_t>(&candidate - candidates.data()));
                }

                struct ProbeOutcome {
                    bool down_ok = false;
                    bool up_ok = false;
                    double down_cost = 0.0;
                    double up_cost = 0.0;
                    std::uint32_t solves = 0;
                };
                std::vector<ProbeOutcome> outcomes(probe_candidates.size());
                SIHPS_OMP(omp parallel for schedule(static) if(probe_candidates.size() > 1 &&
                                                               options.lp_options.parallel_mode != ParallelMode::SERIAL &&
                                                               (options.lp_options.parallel_mode == ParallelMode::PARALLEL ||
                                                                workspace.A.nnz() >= kParallelNnzThreshold)))
                for (std::int32_t probe_index = 0;
                     probe_index < static_cast<std::int32_t>(probe_candidates.size());
                     ++probe_index) {
                    const FractionalCandidate& candidate = candidates[probe_candidates[probe_index]];
                    const auto j = static_cast<std::size_t>(candidate.variable);
                    const double floor_probe = std::floor(relaxation.x[j]);
                    const double ceil_probe = std::ceil(relaxation.x[j]);
                    const double down_distance = candidate.fraction;
                    const double up_distance = 1.0 - candidate.fraction;
                    // Each probe owns its bounds and LP workspace. This is the
                    // only safe parallel region in B&B: pseudocost updates and
                    // queue mutations remain serialized below.
                    auto probe_child = [&](int direction, double bound,
                                           double distance) -> std::pair<bool, double> {
                        std::vector<double> probe_lower = lower;
                        std::vector<double> probe_upper = upper;
                        if (direction < 0) probe_upper[j] = std::min(probe_upper[j], bound);
                        else probe_lower[j] = std::max(probe_lower[j], bound);
                        if (!bounds_are_valid(probe_lower, probe_upper)) return {true, kInfinityValue};
                        LpProblem probe_workspace = workspace;
                        probe_workspace.lower = std::move(probe_lower);
                        probe_workspace.upper = std::move(probe_upper);
                        LpSolverOptions probe_options = relaxation_options;
                        probe_options.parallel_mode = ParallelMode::SERIAL;
                        const LpSolution probe = solve_lp(probe_workspace, probe_options);
                        ++outcomes[probe_index].solves;
                        if (probe.status == LpStatus::INFEASIBLE) return {true, kInfinityValue};
                        if (probe.status != LpStatus::OPTIMAL || distance <= 0.0) return {false, 0.0};
                        return {true, std::max(0.0, probe.objective_value - lower_bound) / distance};
                    };
                    const auto down_probe = probe_child(-1, floor_probe, down_distance);
                    const auto up_probe = probe_child(+1, ceil_probe, up_distance);
                    outcomes[probe_index].down_ok = down_probe.first;
                    outcomes[probe_index].down_cost = down_probe.second;
                    outcomes[probe_index].up_ok = up_probe.first;
                    outcomes[probe_index].up_cost = up_probe.second;
                }
                for (std::size_t probe_index = 0; probe_index < probe_candidates.size(); ++probe_index) {
                    const FractionalCandidate& candidate = candidates[probe_candidates[probe_index]];
                    const ProbeOutcome& outcome = outcomes[probe_index];
                    solution.lp_relaxations += outcome.solves;
                    solution.strong_branching_probes += outcome.solves;
                    if (outcome.down_ok) observe_pseudocost(candidate.variable, -1, outcome.down_cost);
                    if (outcome.up_ok) observe_pseudocost(candidate.variable, +1, outcome.up_cost);
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

        if (node_basis) {
            // Same shared_ptr, refcounted rather than duplicated -- both
            // children start from the same parent basis, one bound-change
            // delta apart from it in opposite directions.
            pending_basis.emplace(left->order, node_basis);
            pending_basis.emplace(right->order, node_basis);
        }

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
