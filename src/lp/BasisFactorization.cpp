#include "BasisFactorization.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sihps {
namespace {

// Absolute floor below which a candidate is never accepted as a pivot, no
// matter how it compares to its column's other entries -- guards against
// selecting a pivot out of a column that is numerically all noise.
constexpr double kPivotFloor = 1e-11;

// Threshold partial pivoting (Duff/Erisman/Reid; the same device SOTA.md
// \S1.4.1 cites via Markowitz): accept any candidate within this factor of
// the column's largest magnitude, then break the tie by sparsity. Relaxing
// exact partial pivoting (which would force factor = 1.0) trades a bounded
// increase in multiplier growth for substantially less fill-in; 0.01 keeps
// multipliers below 100x while leaving real freedom to the sparsity
// heuristic.
constexpr double kPivotThreshold = 0.01;

// Entries below this are dropped from an eta vector. They cannot influence
// a solve beyond round-off, and retaining them would let the eta file grow
// dense over a long update sequence.
constexpr double kEtaDropTol = 1e-14;

} // namespace

std::int32_t BasisFactorization::sparse_lsolve(const SparseColumn& column,
                                                std::int32_t mark_value) {
    // Depth-first search over the graph of L to find, in topological order,
    // exactly the entries of the solution that can be nonzero (Gilbert &
    // Peierls 1988). Without this the solve would cost O(m) per column
    // regardless of sparsity, which is the whole point of the algorithm.
    std::int32_t top = m_;

    for (const auto& [start_row, unused_value] : column) {
        (void)unused_value;
        if (mark_[static_cast<std::size_t>(start_row)] == mark_value) continue;

        std::int32_t head = 0;
        dfs_node_[0] = start_row;

        while (head >= 0) {
            const std::int32_t node = dfs_node_[static_cast<std::size_t>(head)];
            const std::int32_t pivot_step = pinv_[static_cast<std::size_t>(node)];

            if (mark_[static_cast<std::size_t>(node)] != mark_value) {
                mark_[static_cast<std::size_t>(node)] = mark_value;
                // Skip the unit diagonal, stored first in each L column.
                dfs_next_[static_cast<std::size_t>(head)] =
                    (pivot_step < 0) ? 0 : Lp_[static_cast<std::size_t>(pivot_step)] + 1;
            }

            const std::int32_t end =
                (pivot_step < 0) ? 0 : Lp_[static_cast<std::size_t>(pivot_step) + 1];

            bool descended = false;
            for (std::int32_t p = dfs_next_[static_cast<std::size_t>(head)]; p < end; ++p) {
                const std::int32_t child = Li_[static_cast<std::size_t>(p)];
                if (mark_[static_cast<std::size_t>(child)] == mark_value) continue;
                dfs_next_[static_cast<std::size_t>(head)] = p + 1;
                dfs_node_[static_cast<std::size_t>(++head)] = child;
                descended = true;
                break;
            }
            if (!descended) {
                --head;
                pattern_[static_cast<std::size_t>(--top)] = node;
            }
        }
    }

    for (std::int32_t p = top; p < m_; ++p) {
        work_[static_cast<std::size_t>(pattern_[static_cast<std::size_t>(p)])] = 0.0;
    }
    for (const auto& [row, value] : column) {
        work_[static_cast<std::size_t>(row)] = value;
    }

    // Topological order guarantees every L column is applied only after the
    // value that scales it is final.
    for (std::int32_t p = top; p < m_; ++p) {
        const std::int32_t row = pattern_[static_cast<std::size_t>(p)];
        const std::int32_t pivot_step = pinv_[static_cast<std::size_t>(row)];
        if (pivot_step < 0) continue;
        const double value = work_[static_cast<std::size_t>(row)];
        if (value == 0.0) continue;
        const std::int32_t begin = Lp_[static_cast<std::size_t>(pivot_step)] + 1;
        const std::int32_t end = Lp_[static_cast<std::size_t>(pivot_step) + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            work_[static_cast<std::size_t>(Li_[static_cast<std::size_t>(k)])] -=
                Lx_[static_cast<std::size_t>(k)] * value;
        }
    }
    return top;
}

BasisFactorization::FactorizeResult
BasisFactorization::factorize(std::int32_t m, const std::vector<SparseColumn>& columns) {
    FactorizeResult result;
    m_ = m;

    eta_pivot_.clear();
    eta_pivot_value_.clear();
    eta_index_.clear();
    eta_value_.clear();
    eta_ptr_.assign(1, 0);

    const auto um = static_cast<std::size_t>(m);
    pinv_.assign(um, -1);
    Lp_.assign(um + 1, 0);
    Up_.assign(um + 1, 0);
    Li_.clear();
    Lx_.clear();
    Ui_.clear();
    Ux_.clear();
    work_.assign(um, 0.0);
    permuted_.assign(um, 0.0);
    pattern_.assign(um, 0);
    dfs_node_.assign(um, 0);
    dfs_next_.assign(um, 0);
    mark_.assign(um, -1);

    if (m == 0) {
        result.ok = true;
        return result;
    }

    // Column ordering: ascending nonzero count. Cheap, and for a simplex
    // basis it is more than cosmetic -- the many unit columns contributed
    // by slacks and artificials sort to the front and each pivots
    // immediately, triangularizing most of the matrix before any general
    // elimination happens. This is the poor man's version of the explicit
    // triangularization phase production codes run first.
    q_.resize(um);
    std::iota(q_.begin(), q_.end(), 0);
    std::stable_sort(q_.begin(), q_.end(), [&columns](std::int32_t a, std::int32_t b) {
        return columns[static_cast<std::size_t>(a)].size() <
               columns[static_cast<std::size_t>(b)].size();
    });

    // Static row counts of B, used only to break ties between numerically
    // acceptable pivots. A row that is already dense is a poor pivot row
    // because eliminating with it spreads that density into every column it
    // touches.
    row_count_.assign(um, 0);
    for (const auto& col : columns) {
        for (const auto& [row, value] : col) {
            (void)value;
            ++row_count_[static_cast<std::size_t>(row)];
        }
    }

    // Rows still available as pivots, scanned by a moving cursor so the
    // singular-column fallback stays O(m) in total rather than O(m^2).
    std::int32_t free_row_cursor = 0;

    for (std::int32_t step = 0; step < m; ++step) {
        const std::int32_t position = q_[static_cast<std::size_t>(step)];
        const std::int32_t top = sparse_lsolve(columns[static_cast<std::size_t>(position)], step);

        double max_magnitude = 0.0;
        for (std::int32_t p = top; p < m; ++p) {
            const std::int32_t row = pattern_[static_cast<std::size_t>(p)];
            if (pinv_[static_cast<std::size_t>(row)] < 0) {
                max_magnitude =
                    std::max(max_magnitude, std::fabs(work_[static_cast<std::size_t>(row)]));
            }
        }

        std::int32_t pivot_row = -1;
        std::int32_t best_row_count = 0;
        if (max_magnitude > kPivotFloor) {
            for (std::int32_t p = top; p < m; ++p) {
                const std::int32_t row = pattern_[static_cast<std::size_t>(p)];
                if (pinv_[static_cast<std::size_t>(row)] >= 0) continue;
                const double magnitude = std::fabs(work_[static_cast<std::size_t>(row)]);
                if (magnitude < kPivotThreshold * max_magnitude || magnitude <= kPivotFloor) {
                    continue;
                }
                const std::int32_t count = row_count_[static_cast<std::size_t>(row)];
                if (pivot_row < 0 || count < best_row_count) {
                    pivot_row = row;
                    best_row_count = count;
                }
            }
        }

        // U's off-diagonal entries are those landing in already-pivoted
        // rows; they are written before the diagonal so that the diagonal
        // is always the LAST entry of the column, which both triangular
        // solves rely on.
        if (pivot_row >= 0) {
            for (std::int32_t p = top; p < m; ++p) {
                const std::int32_t row = pattern_[static_cast<std::size_t>(p)];
                const std::int32_t where = pinv_[static_cast<std::size_t>(row)];
                if (where < 0) continue;
                const double value = work_[static_cast<std::size_t>(row)];
                if (value == 0.0) continue;
                Ui_.push_back(where);
                Ux_.push_back(value);
            }
        } else {
            // Structurally or numerically dependent column. Substitute the
            // unit column on the first still-unpivoted row so the
            // factorization stays well defined and USABLE, and report it so
            // the caller can repair the basis to match (NUMERICS.md \S5).
            while (free_row_cursor < m && pinv_[static_cast<std::size_t>(free_row_cursor)] >= 0) {
                ++free_row_cursor;
            }
            if (free_row_cursor >= m) {
                result.ok = false;
                return result;
            }
            pivot_row = free_row_cursor;
            work_[static_cast<std::size_t>(pivot_row)] = 1.0;
            result.singular.emplace_back(position, pivot_row);
        }

        const double pivot_value = work_[static_cast<std::size_t>(pivot_row)];
        Ui_.push_back(step);
        Ux_.push_back(pivot_value);
        Up_[static_cast<std::size_t>(step) + 1] = static_cast<std::int32_t>(Ux_.size());

        pinv_[static_cast<std::size_t>(pivot_row)] = step;

        // Unit diagonal first, so both the DFS and the solves can skip it
        // by index arithmetic alone.
        Li_.push_back(pivot_row);
        Lx_.push_back(1.0);
        for (std::int32_t p = top; p < m; ++p) {
            const std::int32_t row = pattern_[static_cast<std::size_t>(p)];
            if (pinv_[static_cast<std::size_t>(row)] >= 0) continue;
            const double value = work_[static_cast<std::size_t>(row)];
            if (value == 0.0) continue;
            Li_.push_back(row);
            Lx_.push_back(value / pivot_value);
        }
        Lp_[static_cast<std::size_t>(step) + 1] = static_cast<std::int32_t>(Lx_.size());

        for (std::int32_t p = top; p < m; ++p) {
            work_[static_cast<std::size_t>(pattern_[static_cast<std::size_t>(p)])] = 0.0;
        }
        work_[static_cast<std::size_t>(pivot_row)] = 0.0;
    }

    // Remap L's row indices from matrix rows into pivot-step space; from
    // here on L is genuinely lower triangular by index.
    for (auto& row : Li_) {
        row = pinv_[static_cast<std::size_t>(row)];
    }

    result.ok = true;
    return result;
}

void BasisFactorization::ftran(std::vector<double>& x) const {
    // B = P^T L U Q^T, so B x = b becomes L U (Q^T x) = P b.
    for (std::int32_t i = 0; i < m_; ++i) {
        permuted_[static_cast<std::size_t>(pinv_[static_cast<std::size_t>(i)])] =
            x[static_cast<std::size_t>(i)];
    }

    for (std::int32_t j = 0; j < m_; ++j) {
        const double value = permuted_[static_cast<std::size_t>(j)];
        if (value == 0.0) continue;
        const std::int32_t begin = Lp_[static_cast<std::size_t>(j)] + 1;
        const std::int32_t end = Lp_[static_cast<std::size_t>(j) + 1];
        for (std::int32_t p = begin; p < end; ++p) {
            permuted_[static_cast<std::size_t>(Li_[static_cast<std::size_t>(p)])] -=
                Lx_[static_cast<std::size_t>(p)] * value;
        }
    }

    for (std::int32_t j = m_ - 1; j >= 0; --j) {
        const std::int32_t diagonal = Up_[static_cast<std::size_t>(j) + 1] - 1;
        const double value =
            permuted_[static_cast<std::size_t>(j)] / Ux_[static_cast<std::size_t>(diagonal)];
        permuted_[static_cast<std::size_t>(j)] = value;
        if (value == 0.0) continue;
        for (std::int32_t p = Up_[static_cast<std::size_t>(j)]; p < diagonal; ++p) {
            permuted_[static_cast<std::size_t>(Ui_[static_cast<std::size_t>(p)])] -=
                Ux_[static_cast<std::size_t>(p)] * value;
        }
    }

    for (std::int32_t k = 0; k < m_; ++k) {
        x[static_cast<std::size_t>(q_[static_cast<std::size_t>(k)])] =
            permuted_[static_cast<std::size_t>(k)];
    }

    // B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1: the etas apply after the LU solve,
    // oldest first.
    const std::int32_t etas = eta_count();
    for (std::int32_t k = 0; k < etas; ++k) {
        const auto pivot = static_cast<std::size_t>(eta_pivot_[static_cast<std::size_t>(k)]);
        const double scaled = x[pivot] / eta_pivot_value_[static_cast<std::size_t>(k)];
        if (scaled == 0.0) continue;
        x[pivot] = scaled;
        const std::int32_t begin = eta_ptr_[static_cast<std::size_t>(k)];
        const std::int32_t end = eta_ptr_[static_cast<std::size_t>(k) + 1];
        for (std::int32_t p = begin; p < end; ++p) {
            x[static_cast<std::size_t>(eta_index_[static_cast<std::size_t>(p)])] -=
                eta_value_[static_cast<std::size_t>(p)] * scaled;
        }
    }
}

void BasisFactorization::btran(std::vector<double>& x) const {
    // B_k^-T = B_0^-T E_1^-T ... E_k^-T: the etas apply first, newest
    // first, and each touches only its own pivot component.
    for (std::int32_t k = eta_count() - 1; k >= 0; --k) {
        const auto pivot = static_cast<std::size_t>(eta_pivot_[static_cast<std::size_t>(k)]);
        double accumulated = x[pivot];
        const std::int32_t begin = eta_ptr_[static_cast<std::size_t>(k)];
        const std::int32_t end = eta_ptr_[static_cast<std::size_t>(k) + 1];
        for (std::int32_t p = begin; p < end; ++p) {
            accumulated -= eta_value_[static_cast<std::size_t>(p)] *
                            x[static_cast<std::size_t>(eta_index_[static_cast<std::size_t>(p)])];
        }
        x[pivot] = accumulated / eta_pivot_value_[static_cast<std::size_t>(k)];
    }

    // B^T = Q U^T L^T P, so B^T y = c becomes U^T L^T (P y) = Q^T c.
    for (std::int32_t k = 0; k < m_; ++k) {
        permuted_[static_cast<std::size_t>(k)] =
            x[static_cast<std::size_t>(q_[static_cast<std::size_t>(k)])];
    }

    for (std::int32_t j = 0; j < m_; ++j) {
        const std::int32_t diagonal = Up_[static_cast<std::size_t>(j) + 1] - 1;
        double accumulated = permuted_[static_cast<std::size_t>(j)];
        for (std::int32_t p = Up_[static_cast<std::size_t>(j)]; p < diagonal; ++p) {
            accumulated -= Ux_[static_cast<std::size_t>(p)] *
                            permuted_[static_cast<std::size_t>(Ui_[static_cast<std::size_t>(p)])];
        }
        permuted_[static_cast<std::size_t>(j)] =
            accumulated / Ux_[static_cast<std::size_t>(diagonal)];
    }

    for (std::int32_t j = m_ - 1; j >= 0; --j) {
        double accumulated = permuted_[static_cast<std::size_t>(j)];
        const std::int32_t begin = Lp_[static_cast<std::size_t>(j)] + 1;
        const std::int32_t end = Lp_[static_cast<std::size_t>(j) + 1];
        for (std::int32_t p = begin; p < end; ++p) {
            accumulated -= Lx_[static_cast<std::size_t>(p)] *
                            permuted_[static_cast<std::size_t>(Li_[static_cast<std::size_t>(p)])];
        }
        permuted_[static_cast<std::size_t>(j)] = accumulated;
    }

    for (std::int32_t i = 0; i < m_; ++i) {
        x[static_cast<std::size_t>(i)] =
            permuted_[static_cast<std::size_t>(pinv_[static_cast<std::size_t>(i)])];
    }
}

bool BasisFactorization::update(std::int32_t leaving_pos, const std::vector<double>& ftran_col) {
    const double pivot = ftran_col[static_cast<std::size_t>(leaving_pos)];
    if (!std::isfinite(pivot) || std::fabs(pivot) < kPivotFloor) return false;

    eta_pivot_.push_back(leaving_pos);
    eta_pivot_value_.push_back(pivot);
    for (std::int32_t i = 0; i < m_; ++i) {
        if (i == leaving_pos) continue;
        const double value = ftran_col[static_cast<std::size_t>(i)];
        if (std::fabs(value) <= kEtaDropTol) continue;
        eta_index_.push_back(i);
        eta_value_.push_back(value);
    }
    eta_ptr_.push_back(static_cast<std::int32_t>(eta_value_.size()));
    return true;
}

} // namespace sihps
