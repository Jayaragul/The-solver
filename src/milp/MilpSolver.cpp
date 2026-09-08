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

// Stay inside binary64's consecutive-integer range. This also excludes
// INT64_MIN and the rounded double representation of INT64_MAX before casts.
constexpr double kExactIntegerLimit = 9007199254740991.0;

bool exact_integer(double value) {
    return std::isfinite(value) && std::fabs(value) <= kExactIntegerLimit &&
           value == std::round(value);
}

bool integer_equality_gcd_infeasible(const MilpProblem& problem) {
    const LpProblem& lp = problem.relaxation;
    for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        const auto rr = static_cast<std::size_t>(row);
        if (lp.row_types[rr] != 'E' || lp.slack_lower[rr] != 0.0 ||
            lp.slack_upper[rr] != 0.0) continue;
        const double rhs = lp.rhs[rr];
        if (!std::isfinite(rhs) || std::fabs(rhs) > kExactIntegerLimit) continue;
        const auto begin = lp.A.row_ptr()[row];
        const auto end = lp.A.row_ptr()[row + 1];
        std::int64_t divisor = 0;
        bool applicable = true;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto j = static_cast<std::size_t>(lp.A.col_idx()[k]);
            const double coefficient = lp.A.values()[k];
            if (problem.variable_types[j] == VariableType::CONTINUOUS ||
                !exact_integer(coefficient)) {
                applicable = false;
                break;
            }
            divisor = std::gcd(divisor, static_cast<std::int64_t>(std::fabs(coefficient)));
        }
        if (!applicable || divisor == 0) continue;
        // gcd(a) divides a*x for every integer x, independent of node bounds.
        // Avoid subtracting a*lower: it adds cancellation and overflow risk.
        if (!exact_integer(rhs) || static_cast<std::int64_t>(rhs) % divisor != 0) return true;
    }
    return false;
}

bool propagate_integer_equality_bounds(const MilpProblem& problem,
                                       std::vector<double>& lower,
                                       std::vector<double>& upper,
                                       std::uint64_t& tightenings) {
    const LpProblem& lp = problem.relaxation;
    constexpr double kIntegerTolerance = 1e-9;
    const std::int32_t max_passes = std::max<std::int32_t>(1, std::min<std::int32_t>(8, lp.n_rows()));
    for (std::int32_t pass = 0; pass < max_passes; ++pass) {
        const std::uint64_t pass_start = tightenings;
        for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        const auto rr = static_cast<std::size_t>(row);
        if (lp.row_types[rr] != 'E' || lp.slack_lower[rr] != 0.0 ||
            lp.slack_upper[rr] != 0.0 || !exact_integer(lp.rhs[rr])) continue;
        const auto begin = lp.A.row_ptr()[row];
        const auto end = lp.A.row_ptr()[row + 1];
        if (begin == end) continue;

        double min_activity = 0.0;
        double max_activity = 0.0;
        double activity_budget = std::fabs(lp.rhs[rr]);
        bool applicable = true;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto j = static_cast<std::size_t>(lp.A.col_idx()[k]);
            const double coefficient = lp.A.values()[k];
            if (problem.variable_types[j] == VariableType::CONTINUOUS ||
                !exact_integer(coefficient) || !exact_integer(lower[j]) ||
                !exact_integer(upper[j])) {
                applicable = false;
                break;
            }
            const double first = coefficient * lower[j];
            const double second = coefficient * upper[j];
            const double magnitude = std::max(std::fabs(first), std::fabs(second));
            // All products, sums and rhs-minus-other-activity must be exact
            // integers before division. Skip the row if that cannot be assured.
            if (!exact_integer(first) || !exact_integer(second) ||
                magnitude > kExactIntegerLimit - activity_budget) {
                applicable = false;
                break;
            }
            activity_budget += magnitude;
            min_activity += std::min(first, second);
            max_activity += std::max(first, second);
        }
        if (!applicable) continue;

        for (std::int32_t k = begin; k < end; ++k) {
            const auto j = static_cast<std::size_t>(lp.A.col_idx()[k]);
            const double coefficient = std::round(lp.A.values()[k]);
            if (coefficient == 0.0) continue;
            const double first = coefficient * lower[j];
            const double second = coefficient * upper[j];
            const double other_min = min_activity - std::min(first, second);
            const double other_max = max_activity - std::max(first, second);
            double implied_lower = (lp.rhs[static_cast<std::size_t>(row)] - other_max) /
                                   coefficient;
            double implied_upper = (lp.rhs[static_cast<std::size_t>(row)] - other_min) /
                                   coefficient;
            if (implied_lower > implied_upper) std::swap(implied_lower, implied_upper);
            implied_lower = std::ceil(implied_lower - kIntegerTolerance);
            implied_upper = std::floor(implied_upper + kIntegerTolerance);
            const double new_lower = std::max(lower[j], implied_lower);
            const double new_upper = std::min(upper[j], implied_upper);
            if (new_lower > new_upper) return false;
            if (new_lower > lower[j] || new_upper < upper[j]) {
                lower[j] = new_lower;
                upper[j] = new_upper;
                ++tightenings;
            }
        }
        }
        if (tightenings == pass_start) break;
    }
    return bounds_are_valid(lower, upper);
}

void round_integer_inequality_rhs(const MilpProblem& problem, LpProblem& workspace,
                                  std::uint64_t& tightenings) {
    constexpr double kIntegerTolerance = 1e-9;
    const LpProblem& lp = problem.relaxation;
    for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        const auto ii = static_cast<std::size_t>(row);
        const char type = lp.row_types[ii];
        if (type != 'L' && type != 'G') continue;
        // Ranged rows have two active sides; leave them unchanged until a
        // range-aware lattice transformation is implemented.
        if (std::isfinite(lp.slack_upper[ii]) && type == 'L') continue;
        if (std::isfinite(lp.slack_lower[ii]) && type == 'G') continue;
        // Only zero-endpoint one-sided slacks here. Root cuts handle the
        // general effective side; rounding the stored RHS alone would be invalid.
        if ((type == 'L' && lp.slack_lower[ii] != 0.0) ||
            (type == 'G' && lp.slack_upper[ii] != 0.0)) continue;
        const auto begin = lp.A.row_ptr()[row];
        const auto end = lp.A.row_ptr()[row + 1];
        if (begin == end || !std::isfinite(workspace.rhs[ii]) ||
            std::fabs(workspace.rhs[ii]) > kExactIntegerLimit) continue;
        std::int64_t divisor = 0;
        bool applicable = true;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto j = static_cast<std::size_t>(lp.A.col_idx()[k]);
            const double coefficient = lp.A.values()[k];
            if (problem.variable_types[j] == VariableType::CONTINUOUS ||
                !exact_integer(coefficient)) {
                applicable = false;
                break;
            }
            divisor = std::gcd(divisor, static_cast<std::int64_t>(std::fabs(coefficient)));
        }
        if (!applicable || divisor <= 1) continue;
        const double rhs = workspace.rhs[ii];
        const double quotient = rhs / static_cast<double>(divisor);
        const double rounded_residual = type == 'L'
                                            ? std::floor(quotient + kIntegerTolerance) * divisor
                                            : std::ceil(quotient - kIntegerTolerance) * divisor;
        const double rounded_rhs = rounded_residual;
        if (!exact_integer(rounded_rhs)) continue;
        if ((type == 'L' && rounded_rhs < rhs - kIntegerTolerance) ||
            (type == 'G' && rounded_rhs > rhs + kIntegerTolerance)) {
            workspace.rhs[ii] = rounded_rhs;
            ++tightenings;
        }
    }
}

// Tighten integer bounds from one-sided rows without changing the row itself.
// For a normalized row c*x <= b, finite bounds on every other term give
//
//     c_j*x_j <= b - min_{k != j} c_k*x_k.
//
// This is an interval proof, not a floating-point cut: only rows whose terms
// are all integer variables and whose bounds are finite are considered.  The
// computed quotient is rounded outward by an error envelope before floor/ceil,
// so cancellation can only miss a tightening, never exclude an integer point.
// Ranged rows, continuous terms, non-finite data, and overflow-prone rows are
// deliberately left for the certified LP relaxation.
bool propagate_integer_inequality_bounds(const MilpProblem& problem,
                                          std::vector<double>& lower,
                                          std::vector<double>& upper,
                                          std::uint64_t& tightenings) {
    constexpr long double kMachineGuard =
        64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
    constexpr long double kRelativeGuard = 1e-12L;
    const LpProblem& lp = problem.relaxation;
    const std::int32_t max_passes =
        std::max<std::int32_t>(1, std::min<std::int32_t>(4, lp.n_rows()));

    for (std::int32_t pass = 0; pass < max_passes; ++pass) {
        const std::uint64_t pass_start = tightenings;
        for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
            const auto rr = static_cast<std::size_t>(row);
            const char type = lp.row_types[rr];
            if ((type != 'L' && type != 'G') ||
                (type == 'L' && std::isfinite(lp.slack_upper[rr])) ||
                (type == 'G' && std::isfinite(lp.slack_lower[rr]))) {
                continue;
            }

            const double source_rhs =
                lp.rhs[rr] - (type == 'L' ? lp.slack_lower[rr] : lp.slack_upper[rr]);
            if (!std::isfinite(source_rhs)) continue;
            const long double sign = type == 'L' ? 1.0L : -1.0L;
            const long double rhs = sign * static_cast<long double>(source_rhs);
            if (!std::isfinite(rhs)) continue;

            const std::int32_t begin = lp.A.row_ptr()[row];
            const std::int32_t end = lp.A.row_ptr()[row + 1];
            if (begin == end || end - begin > 256) continue;

            bool valid = true;
            long double minimum_activity = 0.0L;
            long double absolute_scale = std::fabs(rhs);
            for (std::int32_t k = begin; k < end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                const auto jj = static_cast<std::size_t>(lp.A.col_idx()[kk]);
                const double coefficient = lp.A.values()[kk];
                if (problem.variable_types[jj] == VariableType::CONTINUOUS ||
                    !std::isfinite(coefficient) || !std::isfinite(lower[jj]) ||
                    !std::isfinite(upper[jj])) {
                    valid = false;
                    break;
                }
                const long double normalized = sign * static_cast<long double>(coefficient);
                const long double first = normalized * static_cast<long double>(lower[jj]);
                const long double second = normalized * static_cast<long double>(upper[jj]);
                if (!std::isfinite(normalized) || !std::isfinite(first) ||
                    !std::isfinite(second)) {
                    valid = false;
                    break;
                }
                minimum_activity += std::min(first, second);
                absolute_scale += std::fabs(first) + std::fabs(second);
                if (!std::isfinite(minimum_activity) || !std::isfinite(absolute_scale)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            for (std::int32_t k = begin; k < end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                const auto variable = lp.A.col_idx()[kk];
                const auto jj = static_cast<std::size_t>(variable);
                const long double coefficient =
                    sign * static_cast<long double>(lp.A.values()[kk]);
                if (coefficient == 0.0L) continue;

                const long double own_lower =
                    std::min(coefficient * static_cast<long double>(lower[jj]),
                             coefficient * static_cast<long double>(upper[jj]));
                const long double remainder = minimum_activity - own_lower;
                const long double numerator = rhs - remainder;
                if (!std::isfinite(remainder) || !std::isfinite(numerator)) continue;

                const long double quotient = numerator / coefficient;
                if (!std::isfinite(quotient)) continue;
                const long double error =
                    (kMachineGuard + kRelativeGuard) *
                    (1.0L + absolute_scale + std::fabs(numerator)) /
                    std::max(1.0L, std::fabs(coefficient));
                if (!std::isfinite(error)) continue;

                double candidate_lower = lower[jj];
                double candidate_upper = upper[jj];
                if (coefficient > 0.0L) {
                    const long double outward = quotient + error;
                    if (outward < -static_cast<long double>(kExactIntegerLimit) ||
                        outward > static_cast<long double>(kExactIntegerLimit)) {
                        continue;
                    }
                    candidate_upper = std::floor(static_cast<double>(outward));
                } else {
                    const long double outward = quotient - error;
                    if (outward < -static_cast<long double>(kExactIntegerLimit) ||
                        outward > static_cast<long double>(kExactIntegerLimit)) {
                        continue;
                    }
                    candidate_lower = std::ceil(static_cast<double>(outward));
                }
                const double new_lower = std::max(lower[jj], candidate_lower);
                const double new_upper = std::min(upper[jj], candidate_upper);
                if (new_lower > new_upper) return false;
                if (new_lower > lower[jj] || new_upper < upper[jj]) {
                    lower[jj] = new_lower;
                    upper[jj] = new_upper;
                    ++tightenings;
                }
            }
        }
        if (tightenings == pass_start) break;
    }
    return bounds_are_valid(lower, upper);
}

bool feasible_point(const MilpProblem& problem, const std::vector<double>& x,
                    const std::vector<double>& lower, const std::vector<double>& upper,
                    double feasibility_tolerance, ParallelMode parallel_mode) {
    const LpProblem& lp = problem.relaxation;
    if (x.size() != static_cast<std::size_t>(lp.n_cols())) return false;

    for (std::int32_t j = 0; j < lp.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (!std::isfinite(x[jj])) return false;
        if (x[jj] < lower[jj] - feasibility_tolerance ||
            x[jj] > upper[jj] + feasibility_tolerance) {
            return false;
        }
    }

    std::vector<double> ax(static_cast<std::size_t>(lp.n_rows()), 0.0);
    if (lp.n_rows() > 0) lp.A.multiply(x.data(), ax.data(), parallel_mode);
    double row_violation = 0.0;
    for (std::int32_t i = 0; i < lp.n_rows(); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        if (!std::isfinite(ax[ii])) return false;
        const double lo = lp.rhs[ii] - lp.slack_upper[ii];
        const double hi = lp.rhs[ii] - lp.slack_lower[ii];
        if (std::isfinite(lo)) row_violation = std::max(row_violation, lo - ax[ii]);
        if (std::isfinite(hi)) row_violation = std::max(row_violation, ax[ii] - hi);
    }
    return std::max(0.0, row_violation) <= feasibility_tolerance;
}

bool integral_point(const MilpProblem& problem, const std::vector<double>& x,
                    double integrality_tolerance) {
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj]) ||
            std::fabs(x[jj] - std::round(x[jj])) > integrality_tolerance) {
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

struct BinarySlackStructure {
    std::vector<std::int32_t> binary_columns;
    std::vector<std::int32_t> slack_columns;
    std::vector<std::vector<std::pair<std::int32_t, double>>> binary_terms;
    std::vector<double> rhs;
};

bool extract_binary_slack_structure(const MilpProblem& problem,
                                    BinarySlackStructure& structure) {
    const LpProblem& lp = problem.relaxation;
    if (problem.maximize || lp.n_rows() == 0 || lp.n_rows() > 16) return false;
    for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        if (lp.row_types[static_cast<std::size_t>(row)] != 'E') return false;
    }
    structure.rhs = lp.rhs;
    structure.slack_columns.assign(static_cast<std::size_t>(lp.n_rows()), -1);
    std::vector<std::vector<std::pair<std::int32_t, double>>> by_column(
        static_cast<std::size_t>(lp.n_cols()));
    for (std::int32_t row = 0; row < lp.n_rows(); ++row) {
        for (std::int32_t k = lp.A.row_ptr()[row]; k < lp.A.row_ptr()[row + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            by_column[static_cast<std::size_t>(lp.A.col_idx()[kk])].emplace_back(
                row, lp.A.values()[kk]);
        }
    }
    for (std::int32_t j = 0; j < lp.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const bool fixed_zero = lp.lower[jj] == 0.0 && lp.upper[jj] == 0.0;
        if (problem.variable_types[jj] != VariableType::CONTINUOUS) {
            if (lp.lower[jj] != 0.0 || lp.upper[jj] != 1.0 || lp.obj[jj] != 0.0) return false;
            for (const auto& term : by_column[jj]) {
                if (term.second < 0.0 || !exact_integer(term.second)) return false;
            }
            structure.binary_columns.push_back(j);
            structure.binary_terms.push_back(by_column[jj]);
        } else if (!fixed_zero) {
            if (by_column[jj].size() != 1 || lp.obj[jj] != 1.0 || lp.lower[jj] != 0.0) {
                return false;
            }
            const auto [row, coefficient] = by_column[jj][0];
            if (coefficient != 1.0 || lp.upper[jj] < 0.0) return false;
            auto& slack = structure.slack_columns[static_cast<std::size_t>(row)];
            if (slack >= 0) return false;
            slack = j;
        }
    }
    if (structure.binary_columns.empty()) return false;
    for (std::int32_t slack : structure.slack_columns) if (slack < 0) return false;
    return true;
}

bool binary_slack_repair(const MilpProblem& problem, const std::vector<double>& seed,
                         std::uint32_t max_iterations, double time_limit_seconds,
                         std::vector<double>& repaired) {
    BinarySlackStructure structure;
    if (!extract_binary_slack_structure(problem, structure) ||
        seed.size() != static_cast<std::size_t>(problem.n_cols())) return false;

    const auto start = std::chrono::steady_clock::now();
    const std::int32_t rows = problem.relaxation.n_rows();
    const std::size_t count = structure.binary_columns.size();
    std::vector<unsigned char> bits(count, 0);
    std::vector<double> activity(static_cast<std::size_t>(rows), 0.0);
    for (std::size_t b = 0; b < count; ++b) {
        bits[b] = static_cast<unsigned char>(std::clamp(std::round(
            seed[static_cast<std::size_t>(structure.binary_columns[b])]), 0.0, 1.0));
        if (!bits[b]) continue;
        for (const auto& term : structure.binary_terms[b]) {
            activity[static_cast<std::size_t>(term.first)] += term.second;
        }
    }
    auto score = [&](const std::vector<double>& a) {
        double value = 0.0;
        for (std::int32_t row = 0; row < rows; ++row) {
            const auto rr = static_cast<std::size_t>(row);
            value += std::max(0.0, structure.rhs[rr] - a[rr]);
            value += 1000.0 * std::max(0.0, a[rr] - structure.rhs[rr]);
        }
        return value;
    };
    double current_score = score(activity);
    double best_feasible = kInfinityValue;
    std::vector<unsigned char> best_bits;
    std::vector<unsigned> tabu(count, 0);
    std::uint32_t stalled = 0;
    std::uint32_t pair_attempts = 0;
    for (std::uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        if (time_limit_seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
                time_limit_seconds) break;
        bool feasible = true;
        double objective = 0.0;
        for (std::int32_t row = 0; row < rows; ++row) {
            const auto rr = static_cast<std::size_t>(row);
            if (activity[rr] > structure.rhs[rr] + 1e-9) feasible = false;
            objective += std::max(0.0, structure.rhs[rr] - activity[rr]);
        }
        if (feasible && objective < best_feasible) {
            best_feasible = objective;
            best_bits = bits;
        }

        double best_move_score = kInfinityValue;
        std::size_t best_move = count;
        for (std::size_t b = 0; b < count; ++b) {
            for (const auto& term : structure.binary_terms[b]) {
                const auto rr = static_cast<std::size_t>(term.first);
                activity[rr] += bits[b] ? -term.second : term.second;
            }
            const double candidate_score = score(activity);
            for (const auto& term : structure.binary_terms[b]) {
                const auto rr = static_cast<std::size_t>(term.first);
                activity[rr] += bits[b] ? term.second : -term.second;
            }
            if (tabu[b] == 0 && candidate_score < best_move_score) {
                best_move_score = candidate_score;
                best_move = b;
            }
        }
        if (best_move == count) {
            for (std::size_t b = 0; b < count; ++b) {
                for (const auto& term : structure.binary_terms[b]) {
                    const auto rr = static_cast<std::size_t>(term.first);
                    activity[rr] += bits[b] ? -term.second : term.second;
                }
                const double candidate_score = score(activity);
                for (const auto& term : structure.binary_terms[b]) {
                    const auto rr = static_cast<std::size_t>(term.first);
                    activity[rr] += bits[b] ? term.second : -term.second;
                }
                if (candidate_score < best_move_score) {
                    best_move_score = candidate_score;
                    best_move = b;
                }
            }
        }
        if (best_move == count) break;
        const double previous_score = current_score;
        bits[best_move] = static_cast<unsigned char>(!bits[best_move]);
        for (const auto& term : structure.binary_terms[best_move]) {
            activity[static_cast<std::size_t>(term.first)] +=
                bits[best_move] ? term.second : -term.second;
        }
        for (unsigned& value : tabu) if (value > 0) --value;
        tabu[best_move] = 7;
        current_score = best_move_score;
        if (!std::isfinite(current_score)) break;
        if (current_score + 1e-9 >= previous_score) ++stalled;
        else stalled = 0;

        // One-bit descent can get trapped on the market-split instances.
        // Periodically test a bounded 1->0 / 0->1 exchange to cross that
        // plateau without introducing an unbounded neighborhood search.
        if (stalled >= 256 && (iteration & 255u) == 0u && count <= 128 && pair_attempts < 8) {
            ++pair_attempts;
            double pair_score = current_score;
            std::size_t remove = count, add = count;
            for (std::size_t a = 0; a < count; ++a) {
                if (!bits[a]) continue;
                for (std::size_t b = 0; b < count; ++b) {
                    if (bits[b] || a == b) continue;
                    for (const auto& term : structure.binary_terms[a])
                        activity[static_cast<std::size_t>(term.first)] -= term.second;
                    for (const auto& term : structure.binary_terms[b])
                        activity[static_cast<std::size_t>(term.first)] += term.second;
                    const double candidate = score(activity);
                    for (const auto& term : structure.binary_terms[b])
                        activity[static_cast<std::size_t>(term.first)] -= term.second;
                    for (const auto& term : structure.binary_terms[a])
                        activity[static_cast<std::size_t>(term.first)] += term.second;
                    if (candidate + 1e-9 < pair_score) {
                        pair_score = candidate;
                        remove = a;
                        add = b;
                    }
                }
            }
            if (remove < count) {
                bits[remove] = 0;
                bits[add] = 1;
                for (const auto& term : structure.binary_terms[remove])
                    activity[static_cast<std::size_t>(term.first)] -= term.second;
                for (const auto& term : structure.binary_terms[add])
                    activity[static_cast<std::size_t>(term.first)] += term.second;
                current_score = pair_score;
                stalled = 0;
            }
        }
    }
    if (best_bits.empty() || !std::isfinite(best_feasible)) return false;

    repaired.assign(static_cast<std::size_t>(problem.n_cols()), 0.0);
    for (std::size_t b = 0; b < count; ++b) {
        repaired[static_cast<std::size_t>(structure.binary_columns[b])] = best_bits[b] ? 1.0 : 0.0;
    }
    std::fill(activity.begin(), activity.end(), 0.0);
    for (std::size_t b = 0; b < count; ++b) {
        if (!best_bits[b]) continue;
        for (const auto& term : structure.binary_terms[b]) {
            activity[static_cast<std::size_t>(term.first)] += term.second;
        }
    }
    for (std::int32_t row = 0; row < rows; ++row) {
        repaired[static_cast<std::size_t>(structure.slack_columns[static_cast<std::size_t>(row)])] =
            structure.rhs[static_cast<std::size_t>(row)] - activity[static_cast<std::size_t>(row)];
    }
    return true;
}

struct FractionalCandidate {
    std::int32_t variable = -1;
    double fraction = 0.0;
    double fractionality = 0.0;
};

struct CoverCut {
    std::vector<std::int32_t> variables;
};

struct IntegerRoundingCut {
    std::vector<std::pair<std::int32_t, double>> terms;
    char row_type = 'L';
    double rhs = 0.0;
};

struct GeneralCut {
    std::vector<std::pair<std::int32_t, double>> terms;
    char row_type = 'G';
    double rhs = 0.0;
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

std::vector<IntegerRoundingCut> separate_integer_rounding_cuts(
    const MilpProblem& problem, const std::vector<double>& x, double violation_tolerance,
    std::uint32_t limit) {
    std::vector<IntegerRoundingCut> cuts;
    const LpProblem& lp = problem.relaxation;
    constexpr double integer_tolerance = 1e-9;
    for (std::int32_t row = 0; row < lp.n_rows() && cuts.size() < limit; ++row) {
        const auto rr = static_cast<std::size_t>(row);
        const char type = lp.row_types[rr];
        if (type != 'L' && type != 'G') continue;
        // Ranged rows have two coupled sides; defer until a range-aware
        // transformation exists rather than infer an invalid cut.
        if ((type == 'L' && std::isfinite(lp.slack_lower[rr]) &&
             std::isfinite(lp.slack_upper[rr])) ||
            (type == 'G' && std::isfinite(lp.slack_lower[rr]) &&
             std::isfinite(lp.slack_upper[rr]))) continue;
        // The slack bounds define the actual side of Ax+s=rhs. A custom
        // one-sided slack need not have its finite endpoint at zero.
        const double rhs = lp.rhs[rr] -
                           (type == 'L' ? lp.slack_lower[rr] : lp.slack_upper[rr]);
        if (!std::isfinite(rhs)) continue;
        const double rounded_rhs = type == 'L' ? std::floor(rhs + integer_tolerance)
                                               : std::ceil(rhs - integer_tolerance);
        if (std::fabs(rhs - rounded_rhs) <= integer_tolerance) continue;

        IntegerRoundingCut cut;
        cut.row_type = type;
        cut.rhs = rounded_rhs;
        double activity = 0.0;
        bool valid = true;
        for (std::int32_t k = lp.A.row_ptr()[row]; k < lp.A.row_ptr()[row + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const std::int32_t variable = lp.A.col_idx()[kk];
            const auto jj = static_cast<std::size_t>(variable);
            const double coefficient = lp.A.values()[kk];
            const double integral_coefficient = std::round(coefficient);
            // Exact integrality is required for this proof. A tiny rounding
            // error can have arbitrarily large activity with large bounds.
            if (!std::isfinite(coefficient) || coefficient != integral_coefficient ||
                problem.variable_types[jj] == VariableType::CONTINUOUS) {
                valid = false;
                break;
            }
            if (integral_coefficient != 0.0) {
                cut.terms.emplace_back(variable, integral_coefficient);
                activity += integral_coefficient * x[jj];
            }
        }
        if (!valid || cut.terms.empty()) continue;
        const bool violated = type == 'L' ? activity > cut.rhs + violation_tolerance
                                          : activity < cut.rhs - violation_tolerance;
        if (violated) cuts.push_back(std::move(cut));
    }
    return cuts;
}

void append_integer_rounding_cuts(LpProblem& workspace,
                                  const std::vector<IntegerRoundingCut>& cuts) {
    const std::int32_t old_rows = workspace.n_rows();
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) +
                    std::accumulate(cuts.begin(), cuts.end(), std::size_t{0},
                                    [](std::size_t total, const IntegerRoundingCut& cut) {
                                        return total + cut.terms.size();
                                    }));
    for (std::int32_t row = 0; row < old_rows; ++row) {
        for (std::int32_t k = workspace.A.row_ptr()[row]; k < workspace.A.row_ptr()[row + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            entries.push_back({row, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
        }
    }
    for (std::size_t cut_index = 0; cut_index < cuts.size(); ++cut_index) {
        const auto row = old_rows + static_cast<std::int32_t>(cut_index);
        for (const auto& term : cuts[cut_index].terms) {
            entries.push_back({row, term.first, term.second});
        }
    }
    workspace.A = CSRMatrix::from_triplets(
        old_rows + static_cast<std::int32_t>(cuts.size()), workspace.n_cols(), entries);
    for (const IntegerRoundingCut& cut : cuts) {
        workspace.rhs.push_back(cut.rhs);
        workspace.row_types.push_back(cut.row_type);
        if (cut.row_type == 'L') {
            workspace.slack_lower.push_back(0.0);
            workspace.slack_upper.push_back(kInfinityValue);
        } else {
            workspace.slack_lower.push_back(-kInfinityValue);
            workspace.slack_upper.push_back(0.0);
        }
    }
}

// Validity-preserving coefficient-floor strengthening for pure-integer rows.
// Normalize every one-sided row to a*x <= b.  With integer x >= l, let
// y=x-l >= 0 and c=floor(a).  Then c*y <= a*y <= b-a*l, and c*y is integer,
// so c*y <= floor(b-a*l).  The routine skips continuous variables, ranged
// rows, non-finite bounds, overflow-prone activity, and rows where no
// coefficient was actually rounded.  This is intentionally narrower than a
// general MIR separator: it is simple enough to audit and never weakens the
// certificate path.
std::vector<IntegerRoundingCut> separate_integer_coefficient_rounding_cuts(
    const MilpProblem& problem, const std::vector<double>& x, double violation_tolerance,
    std::uint32_t limit) {
    std::vector<IntegerRoundingCut> cuts;
    const LpProblem& lp = problem.relaxation;
    constexpr double integer_tolerance = 1e-10;
    constexpr double max_exact_integer = 9007199254740991.0;
    for (std::int32_t row = 0; row < lp.n_rows() && cuts.size() < limit; ++row) {
        const auto rr = static_cast<std::size_t>(row);
        const char type = lp.row_types[rr];
        if (type != 'L' && type != 'G') continue;
        if (std::isfinite(lp.slack_lower[rr]) && std::isfinite(lp.slack_upper[rr])) continue;
        const double slack_endpoint = type == 'L' ? lp.slack_lower[rr] : lp.slack_upper[rr];
        const double source_rhs = lp.rhs[rr] - slack_endpoint;
        if (!std::isfinite(source_rhs)) continue;
        const double sign = type == 'L' ? 1.0 : -1.0;
        const double normalized_rhs = sign * source_rhs;
        if (!std::isfinite(normalized_rhs)) continue;

        IntegerRoundingCut cut;
        cut.row_type = 'L';
        bool valid = true;
        bool coefficient_changed = false;
        double lower_shift = 0.0;
        double activity = 0.0;
        for (std::int32_t k = lp.A.row_ptr()[row]; k < lp.A.row_ptr()[row + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const std::int32_t variable = lp.A.col_idx()[kk];
            const auto jj = static_cast<std::size_t>(variable);
            if (problem.variable_types[jj] == VariableType::CONTINUOUS ||
                !std::isfinite(lp.lower[jj])) {
                valid = false;
                break;
            }
            const double integer_lower = std::ceil(lp.lower[jj] - integer_tolerance);
            if (!std::isfinite(integer_lower) || std::fabs(integer_lower) > max_exact_integer ||
                !std::isfinite(lp.A.values()[kk])) {
                valid = false;
                break;
            }
            const double normalized = sign * lp.A.values()[kk];
            const double coefficient = std::floor(normalized);
            if (!std::isfinite(coefficient) || std::fabs(coefficient) > max_exact_integer) {
                valid = false;
                break;
            }
            if (std::fabs(coefficient - normalized) > integer_tolerance) coefficient_changed = true;
            lower_shift += coefficient * integer_lower;
            if (!std::isfinite(lower_shift) || std::fabs(lower_shift) > max_exact_integer) {
                valid = false;
                break;
            }
            if (coefficient != 0.0) {
                cut.terms.emplace_back(variable, coefficient);
                activity += coefficient * x[jj];
            }
        }
        if (!valid || !coefficient_changed || cut.terms.empty()) continue;
        const double cut_rhs = std::floor(normalized_rhs - lower_shift + integer_tolerance) +
                               lower_shift;
        if (!std::isfinite(cut_rhs) || activity <= cut_rhs + violation_tolerance) continue;
        cut.rhs = cut_rhs;
        cuts.push_back(std::move(cut));
    }
    return cuts;
}

void append_general_cuts(LpProblem& workspace, const std::vector<GeneralCut>& cuts) {
    const std::int32_t old_rows = workspace.n_rows();
    std::size_t new_nnz = 0;
    for (const GeneralCut& cut : cuts) new_nnz += cut.terms.size();
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) + new_nnz);
    for (std::int32_t row = 0; row < old_rows; ++row) {
        for (std::int32_t k = workspace.A.row_ptr()[row];
             k < workspace.A.row_ptr()[row + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            entries.push_back({row, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
        }
    }
    for (std::size_t cut_index = 0; cut_index < cuts.size(); ++cut_index) {
        const auto row = old_rows + static_cast<std::int32_t>(cut_index);
        for (const auto& term : cuts[cut_index].terms) {
            entries.push_back({row, term.first, term.second});
        }
    }
    workspace.A = CSRMatrix::from_triplets(
        old_rows + static_cast<std::int32_t>(cuts.size()), workspace.n_cols(), entries);
    for (const GeneralCut& cut : cuts) {
        workspace.rhs.push_back(cut.rhs);
        workspace.row_types.push_back(cut.row_type);
        if (cut.row_type == 'L') {
            workspace.slack_lower.push_back(0.0);
            workspace.slack_upper.push_back(kInfinityValue);
        } else {
            workspace.slack_lower.push_back(-kInfinityValue);
            workspace.slack_upper.push_back(0.0);
        }
    }
}

// Numerically guarded Gomory mixed-integer cuts from the terminal root
// tableau. The separator is deliberately opt-in: a cut is discarded when
// its fractional row or coefficient scale is too close to floating-point
// noise, so an optional strengthening can never compromise certification.
std::vector<GeneralCut> separate_gmi_cuts(const MilpProblem& problem,
                                          const LpProblem& workspace,
                                          const Simplex& simplex,
                                          const std::vector<double>& x,
                                          double violation_tolerance,
                                          std::uint32_t limit) {
    constexpr double min_fractionality = 0.01;
    constexpr double relative_zero = 1e-9;
    constexpr double max_dynamic_range = 1e8;
    constexpr double max_relative_magnitude = 1e4;
    std::vector<GeneralCut> cuts;
    const std::int32_t n = workspace.n_cols();
    const std::int32_t m = workspace.n_rows();
    double matrix_max = 0.0;
    for (std::int32_t k = 0; k < workspace.A.nnz(); ++k) {
        matrix_max = std::max(matrix_max, std::fabs(workspace.A.values()[k]));
    }
    if (matrix_max == 0.0) return cuts;
    std::vector<double> coefficients(static_cast<std::size_t>(n), 0.0);

    for (std::int32_t basic = 0; basic < n && cuts.size() < limit; ++basic) {
        const auto bb = static_cast<std::size_t>(basic);
        if (problem.variable_types[bb] == VariableType::CONTINUOUS) continue;
        const std::int32_t row = simplex.basic_row_of(basic);
        if (row < 0 || !std::isfinite(x[bb])) continue;
        const double floor_value = std::floor(x[bb]);
        const double f = x[bb] - floor_value;
        if (f < min_fractionality || 1.0 - f < min_fractionality) continue;

        std::fill(coefficients.begin(), coefficients.end(), 0.0);
        double constant = 0.0;
        bool reject = false;
        const auto tableau = simplex.tableau_row(row);
        for (std::int32_t j = 0; j < simplex.n_total() && !reject; ++j) {
            if (j == basic) continue;
            const double rho = tableau[static_cast<std::size_t>(j)];
            if (rho == 0.0 || simplex.status_of(j) == Simplex::VarStatus::BASIC) continue;
            const auto status = simplex.status_of(j);
            if (status == Simplex::VarStatus::AT_ZERO) { reject = true; break; }
            const bool at_lower = status == Simplex::VarStatus::AT_LOWER;
            double bound = 0.0;
            if (j < n) {
                bound = at_lower ? workspace.lower[static_cast<std::size_t>(j)]
                                 : workspace.upper[static_cast<std::size_t>(j)];
            } else if (j < n + m) {
                const auto r = static_cast<std::size_t>(j - n);
                bound = at_lower ? workspace.slack_lower[r] : workspace.slack_upper[r];
            } else {
                continue;
            }
            if (!std::isfinite(bound)) { reject = true; break; }
            const double bar = at_lower ? rho : -rho;
            const bool integer = j < n &&
                problem.variable_types[static_cast<std::size_t>(j)] != VariableType::CONTINUOUS;
            const double fj = bar - std::floor(bar);
            const double coefficient = integer
                ? ((fj <= f) ? fj / f : (1.0 - fj) / (1.0 - f))
                : ((bar >= 0.0) ? bar / f : -bar / (1.0 - f));
            if (!std::isfinite(coefficient) || coefficient == 0.0) continue;
            const double signed_coefficient = at_lower ? coefficient : -coefficient;
            if (j < n) {
                coefficients[static_cast<std::size_t>(j)] += signed_coefficient;
                constant -= signed_coefficient * bound;
            } else {
                const std::int32_t source_row = j - n;
                const auto rr = static_cast<std::size_t>(source_row);
                constant += signed_coefficient * workspace.rhs[rr] - signed_coefficient * bound;
                for (std::int32_t k = workspace.A.row_ptr()[source_row];
                     k < workspace.A.row_ptr()[source_row + 1]; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    coefficients[static_cast<std::size_t>(workspace.A.col_idx()[kk])] -=
                        signed_coefficient * workspace.A.values()[kk];
                }
            }
        }
        if (reject) continue;
        double row_max = 0.0;
        for (double c : coefficients) row_max = std::max(row_max, std::fabs(c));
        if (row_max == 0.0) continue;
        const double zero_floor = relative_zero * row_max;
        double row_min = std::numeric_limits<double>::infinity();
        row_max = 0.0;
        for (double& c : coefficients) {
            if (c != 0.0 && std::fabs(c) < zero_floor) c = 0.0;
            if (c != 0.0) {
                row_min = std::min(row_min, std::fabs(c));
                row_max = std::max(row_max, std::fabs(c));
            }
        }
        if (row_max == 0.0 || row_max / row_min > max_dynamic_range ||
            row_max > max_relative_magnitude * matrix_max) continue;
        const double rhs = 1.0 - constant;
        double activity = 0.0;
        GeneralCut cut;
        cut.row_type = 'G';
        cut.rhs = rhs;
        for (std::int32_t j = 0; j < n; ++j) {
            const double c = coefficients[static_cast<std::size_t>(j)];
            if (c != 0.0) {
                activity += c * x[static_cast<std::size_t>(j)];
                cut.terms.push_back({j, c});
            }
        }
        if (activity < rhs - violation_tolerance && !cut.terms.empty()) cuts.push_back(std::move(cut));
    }
    return cuts;
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
        options.objective_tolerance < 0.0 || options.time_limit_seconds < 0.0 ||
        options.feasibility_pump_objective_weight < 0.0) {
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
    bool root_gmi_separated = false;

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
    const auto apply_remaining_lp_budget = [&](LpSolverOptions& lp_options) {
        if (options.time_limit_seconds <= 0.0) return;
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const double remaining = std::max(0.0, options.time_limit_seconds - elapsed);
        if (lp_options.simplex_time_budget_seconds <= 0.0 ||
            remaining < lp_options.simplex_time_budget_seconds) {
            lp_options.simplex_time_budget_seconds = remaining;
        }
    };
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
        if (!std::isfinite(candidate_objective)) return false;
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
                auto dive_options = relaxation_options;
                apply_remaining_lp_budget(dive_options);
                const LpSolution child = solve_lp(workspace, dive_options);
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

    const auto attempt_rens = [&](const LpSolution& starting,
                                  const std::vector<double>& starting_lower,
                                  const std::vector<double>& starting_upper) {
        if (!options.use_rens_heuristic || std::isfinite(incumbent) ||
            integral_point(problem, starting.x, options.integrality_tolerance)) {
            return;
        }
        std::vector<double> restricted_lower = starting_lower;
        std::vector<double> restricted_upper = starting_upper;
        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            const auto jj = static_cast<std::size_t>(j);
            if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
            const double value = starting.x[jj];
            if (!std::isfinite(value)) return;
            const double floor_value = std::floor(value);
            const double ceil_value = std::ceil(value);
            if (value - floor_value <= options.integrality_tolerance ||
                ceil_value - value <= options.integrality_tolerance) {
                const double fixed = std::round(value);
                restricted_lower[jj] = std::max(restricted_lower[jj], fixed);
                restricted_upper[jj] = std::min(restricted_upper[jj], fixed);
            } else {
                restricted_lower[jj] = std::max(restricted_lower[jj], floor_value);
                restricted_upper[jj] = std::min(restricted_upper[jj], ceil_value);
            }
        }
        if (!bounds_are_valid(restricted_lower, restricted_upper)) return;
        LpProblem restricted = workspace;
        restricted.lower = std::move(restricted_lower);
        restricted.upper = std::move(restricted_upper);
        auto rens_options = relaxation_options;
        apply_remaining_lp_budget(rens_options);
        ++solution.lp_relaxations;
        ++solution.rens_heuristic_lp_relaxations;
        const LpSolution result = solve_lp(restricted, rens_options);
        if (result.status == LpStatus::OPTIMAL &&
            result.x.size() == static_cast<std::size_t>(problem.n_cols())) {
            consider_incumbent(result.x, starting_lower, starting_upper);
        }
    };

    const auto attempt_feasibility_pump = [&](const LpSolution& starting,
                                              const std::vector<double>& starting_lower,
                                              const std::vector<double>& starting_upper) {
        if (!options.use_feasibility_pump || options.feasibility_pump_max_iterations == 0 ||
            options.feasibility_pump_max_lp_relaxations == 0 || std::isfinite(incumbent)) {
            return;
        }

        std::vector<std::int32_t> integer_columns;
        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            if (problem.variable_types[static_cast<std::size_t>(j)] != VariableType::CONTINUOUS) {
                integer_columns.push_back(j);
            }
        }
        if (integer_columns.empty()) return;

        std::vector<double> target = rounded_point(problem, starting.x,
                                                   starting_lower, starting_upper);
        std::vector<double> previous_target;
        std::uint32_t relaxations = 0;

        for (std::uint32_t iteration = 0;
             iteration < options.feasibility_pump_max_iterations;
             ++iteration) {
            if (timed_out()) {
                heuristic_timeout = true;
                return;
            }
            if (target == previous_target) {
                // A deterministic cycle break: flip the most fractional
                // integer target that has an adjacent value in the node box.
                std::int32_t selected = -1;
                double best_fractionality = -1.0;
                for (std::int32_t j : integer_columns) {
                    const auto jj = static_cast<std::size_t>(j);
                    const double value = starting.x[jj];
                    const double fractionality =
                        std::min(value - std::floor(value), std::ceil(value) - value);
                    if (fractionality > best_fractionality &&
                        std::isfinite(starting_lower[jj]) && std::isfinite(starting_upper[jj]) &&
                        starting_lower[jj] < starting_upper[jj]) {
                        selected = j;
                        best_fractionality = fractionality;
                    }
                }
                if (selected < 0) return;
                const auto jj = static_cast<std::size_t>(selected);
                const double down = target[jj] - 1.0;
                const double up = target[jj] + 1.0;
                if (down >= starting_lower[jj]) target[jj] = down;
                else if (up <= starting_upper[jj]) target[jj] = up;
                else return;
            }
            previous_target = target;

            const std::int32_t n = workspace.n_cols();
            const std::int32_t m = workspace.n_rows();
            const std::int32_t integer_count = static_cast<std::int32_t>(integer_columns.size());
            std::vector<Triplet> entries;
            entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) +
                            static_cast<std::size_t>(2 * integer_count));
            for (std::int32_t row = 0; row < m; ++row) {
                for (std::int32_t k = workspace.A.row_ptr()[row];
                     k < workspace.A.row_ptr()[row + 1]; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    entries.push_back({row, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
                }
            }
            for (std::int32_t p = 0; p < integer_count; ++p) {
                const std::int32_t variable = integer_columns[static_cast<std::size_t>(p)];
                const std::int32_t distance_column = n + p;
                const std::int32_t lower_row = m + 2 * p;
                const std::int32_t upper_row = lower_row + 1;
                entries.push_back({lower_row, variable, 1.0});
                entries.push_back({lower_row, distance_column, -1.0});
                entries.push_back({upper_row, variable, -1.0});
                entries.push_back({upper_row, distance_column, -1.0});
            }

            LpProblem pump;
            pump.A = CSRMatrix::from_triplets(m + 2 * integer_count,
                                              n + integer_count, entries);
            pump.obj.assign(static_cast<std::size_t>(n + integer_count), 0.0);
            double objective_scale = 0.0;
            for (double coefficient : workspace.obj) {
                objective_scale = std::max(objective_scale, std::fabs(coefficient));
            }
            if (objective_scale == 0.0) objective_scale = 1.0;
            if (options.feasibility_pump_objective_weight > 0.0) {
                for (std::int32_t j = 0; j < n; ++j) {
                    pump.obj[static_cast<std::size_t>(j)] =
                        options.feasibility_pump_objective_weight *
                        workspace.obj[static_cast<std::size_t>(j)] / objective_scale;
                }
            }
            pump.lower = starting_lower;
            pump.upper = starting_upper;
            pump.lower.resize(static_cast<std::size_t>(n + integer_count), 0.0);
            pump.upper.resize(static_cast<std::size_t>(n + integer_count), kInfinityValue);
            for (std::int32_t p = 0; p < integer_count; ++p) {
                pump.obj[static_cast<std::size_t>(n + p)] = 1.0;
            }
            pump.rhs = workspace.rhs;
            pump.row_types = workspace.row_types;
            pump.slack_lower = workspace.slack_lower;
            pump.slack_upper = workspace.slack_upper;
            pump.rhs.resize(static_cast<std::size_t>(m + 2 * integer_count), 0.0);
            pump.row_types.resize(static_cast<std::size_t>(m + 2 * integer_count), 'L');
            pump.slack_lower.resize(static_cast<std::size_t>(m + 2 * integer_count), 0.0);
            pump.slack_upper.resize(static_cast<std::size_t>(m + 2 * integer_count), kInfinityValue);
            for (std::int32_t p = 0; p < integer_count; ++p) {
                const auto pp = static_cast<std::size_t>(p);
                const std::int32_t variable = integer_columns[pp];
                pump.rhs[static_cast<std::size_t>(m + 2 * p)] = target[static_cast<std::size_t>(variable)];
                pump.rhs[static_cast<std::size_t>(m + 2 * p + 1)] =
                    -target[static_cast<std::size_t>(variable)];
            }
            pump.lower.resize(static_cast<std::size_t>(n + integer_count));
            pump.upper.resize(static_cast<std::size_t>(n + integer_count));
            for (std::int32_t p = 0; p < integer_count; ++p) {
                pump.lower[static_cast<std::size_t>(n + p)] = 0.0;
                pump.upper[static_cast<std::size_t>(n + p)] = kInfinityValue;
            }

            if (relaxations >= options.feasibility_pump_max_lp_relaxations) return;
            auto pump_options = relaxation_options;
            apply_remaining_lp_budget(pump_options);
            ++relaxations;
            ++solution.lp_relaxations;
            ++solution.feasibility_pump_lp_relaxations;
            const LpSolution result = solve_lp(pump, pump_options);
            if (result.status != LpStatus::OPTIMAL ||
                result.x.size() != static_cast<std::size_t>(n + integer_count)) {
                return;
            }
            std::vector<double> candidate(result.x.begin(), result.x.begin() + n);
            if (integral_point(problem, candidate, options.integrality_tolerance)) {
                if (consider_incumbent(candidate, starting_lower, starting_upper)) return;
            }
            const std::vector<double> next_target = rounded_point(
                problem, candidate, starting_lower, starting_upper);
            if (next_target == target) return;
            target = next_target;
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
                auto local_options = relaxation_options;
                apply_remaining_lp_budget(local_options);
                const LpSolution local_solution = solve_lp(local, local_options);
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
        if (options.enable_integer_equality_propagation &&
            !propagate_integer_equality_bounds(problem, lower, upper,
                                               solution.integer_bound_tightenings)) {
            ++solution.nodes_pruned;
            ++solution.integer_propagation_prunes;
            continue;
        }
        if (options.enable_integer_inequality_propagation &&
            !propagate_integer_inequality_bounds(problem, lower, upper,
                                                  solution.integer_bound_tightenings)) {
            ++solution.nodes_pruned;
            ++solution.integer_propagation_prunes;
            continue;
        }
        if (options.enable_integer_gcd_tightening &&
            integer_equality_gcd_infeasible(problem)) {
            ++solution.nodes_pruned;
            ++solution.integer_gcd_prunes;
            continue;
        }
        workspace.lower = lower;
        workspace.upper = upper;
        if (options.enable_integer_inequality_rounding) {
            round_integer_inequality_rhs(problem, workspace,
                                         solution.integer_rhs_tightenings);
        }

        if (timed_out()) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        ++solution.lp_relaxations;
        auto node_options = relaxation_options;
        apply_remaining_lp_budget(node_options);
        LpSolution relaxation;
        std::shared_ptr<const Simplex::Basis> node_basis;
        if (node->depth == 0 || !options.warm_start_node_relaxations) {
            // Root always takes this path: its solve goes through
            // solve_lp's presolve, and a warm basis is only valid for a
            // child that solves over the SAME augmented column space --
            // not guaranteed once presolve's bound-dependent reductions
            // are in the picture. Every node also takes this path when the
            // feature is off, which is exactly today's behavior.
            relaxation = solve_lp(workspace, node_options);
        } else {
            if (!node_scale_ready) {
                node_scale = relaxation_options.use_ruiz_scaling
                                 ? compute_ruiz_scaling(workspace.A)
                                 : ScaleFactors::identity(workspace.n_rows(), workspace.n_cols());
                node_scale_ready = true;
            }
            Simplex simplex(workspace, node_options.backend,
                             node_options.use_ruiz_scaling, node_options.pricing_rule,
                             LpAlgorithm::AUTO, relaxation_options.parallel_mode, &node_scale);
            if (node_options.simplex_time_budget_seconds > 0.0) {
                simplex.set_time_budget(node_options.simplex_time_budget_seconds);
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
        if (relaxation.status == LpStatus::ITERATION_LIMIT && timed_out()) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        if (relaxation.status != LpStatus::OPTIMAL ||
            relaxation.x.size() != static_cast<std::size_t>(problem.n_cols())) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        const double lower_bound = relaxation.objective_value;
        record_pseudocost(*node, lower_bound, false);

        if (node->depth == 0 && !root_cuts_separated &&
            (options.enable_root_cover_cuts || options.enable_root_integer_rounding_cuts ||
             options.enable_root_integer_coefficient_rounding_cuts)) {
            root_cuts_separated = true;
            const auto cover_cuts = options.enable_root_cover_cuts
                                        ? separate_cover_cuts(problem, relaxation.x,
                                                               options.cut_violation_tolerance,
                                                               options.max_root_cover_cuts)
                                        : std::vector<CoverCut>{};
            const auto integer_cuts = options.enable_root_integer_rounding_cuts
                                          ? separate_integer_rounding_cuts(
                                                problem, relaxation.x,
                                                options.cut_violation_tolerance,
                                                options.max_root_integer_rounding_cuts)
                                          : std::vector<IntegerRoundingCut>{};
            const auto coefficient_cuts = options.enable_root_integer_coefficient_rounding_cuts
                                              ? separate_integer_coefficient_rounding_cuts(
                                                    problem, relaxation.x,
                                                    options.cut_violation_tolerance,
                                                    options.max_root_integer_coefficient_rounding_cuts)
                                              : std::vector<IntegerRoundingCut>{};
            if (!cover_cuts.empty() || !integer_cuts.empty() || !coefficient_cuts.empty()) {
                if (!cover_cuts.empty()) {
                    append_cover_cuts(workspace, cover_cuts);
                    solution.root_cover_cuts = cover_cuts.size();
                    solution.cover_cuts = cover_cuts.size();
                }
                if (!integer_cuts.empty()) {
                    append_integer_rounding_cuts(workspace, integer_cuts);
                    solution.root_integer_rounding_cuts = integer_cuts.size();
                }
                if (!coefficient_cuts.empty()) {
                    append_integer_rounding_cuts(workspace, coefficient_cuts);
                    solution.root_integer_coefficient_rounding_cuts = coefficient_cuts.size();
                }
                open.push(node);
                continue;
            }
        }
        if (node->depth == 0 && !root_gmi_separated && options.enable_root_gmi_cuts) {
            root_gmi_separated = true;
            Simplex gmi_simplex(workspace, PricingBackend::CPU, false,
                                relaxation_options.pricing_rule, LpAlgorithm::AUTO,
                                relaxation_options.parallel_mode);
            if (node_options.simplex_time_budget_seconds > 0.0) {
                gmi_simplex.set_time_budget(node_options.simplex_time_budget_seconds);
            }
            ++solution.lp_relaxations;
            const LpResult gmi_result = gmi_simplex.solve();
            if (gmi_result.status == LpStatus::OPTIMAL &&
                gmi_result.x.size() == static_cast<std::size_t>(problem.n_cols())) {
                const auto cuts = separate_gmi_cuts(problem, workspace, gmi_simplex,
                                                     gmi_result.x,
                                                     options.cut_violation_tolerance,
                                                     options.max_root_gmi_cuts);
                if (!cuts.empty()) {
                    append_general_cuts(workspace, cuts);
                    solution.root_gmi_cuts += cuts.size();
                    open.push(node);
                    continue;
                }
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

        if (node->depth == 0 && options.use_binary_slack_heuristic &&
            options.binary_slack_max_iterations > 0 && !timed_out()) {
            const std::vector<double>& seed = std::isfinite(incumbent) ? incumbent_x : rounded;
            std::vector<double> repaired;
            if (binary_slack_repair(problem, seed, options.binary_slack_max_iterations,
                                    options.binary_slack_time_limit_seconds, repaired)) {
                consider_incumbent(repaired, lower, upper);
            }
            if (timed_out()) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }

        if (candidate_integral) continue;

        if (node->depth == 0 && !std::isfinite(incumbent)) {
            attempt_rens(relaxation, lower, upper);
            if (timed_out()) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
            attempt_feasibility_pump(relaxation, lower, upper);
            if (heuristic_timeout) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }

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
                        apply_remaining_lp_budget(probe_options);
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
            const auto retain_basis = [&](std::uint64_t order) {
                if (options.max_pending_warm_start_bases == 0 ||
                    pending_basis.size() < options.max_pending_warm_start_bases) {
                    pending_basis.emplace(order, node_basis);
                } else {
                    ++solution.warm_start_basis_cap_skips;
                }
            };
            retain_basis(left->order);
            retain_basis(right->order);
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
