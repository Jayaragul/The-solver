#include "Scaling.hpp"

#include "../sparse/CSCMatrix.hpp"
#include "../sparse/Convert.hpp"

#include <algorithm>
#include <cmath>

namespace sihps {

ScaleFactors compute_ruiz_scaling(const CSRMatrix& a, int max_iters, double tol) {
    const std::int32_t m = a.rows();
    const std::int32_t n = a.cols();
    ScaleFactors s = ScaleFactors::identity(m, n);
    if (a.nnz() == 0 || m == 0 || n == 0) return s;

    // Column norms need column-major access; built once, reused every
    // iteration (the sparsity pattern never changes, only the running
    // row_scale/col_scale weights applied on top of it).
    const CSCMatrix a_csc = csr_to_csc(a);

    bool converged = false;
    for (int iter = 0; iter < max_iters; ++iter) {
        // Row/column infinity norms of the matrix AS CURRENTLY SCALED,
        // recovered from the ORIGINAL values and the running R/C weights
        // (A'_ij = R_i * A_ij * C_j) rather than by materializing a scaled
        // copy of A every iteration -- keeps each iteration O(nnz).
        std::vector<double> row_norm(static_cast<std::size_t>(m), 0.0);
        for (std::int32_t i = 0; i < m; ++i) {
            const std::int32_t begin = a.row_ptr()[i];
            const std::int32_t end = a.row_ptr()[i + 1];
            double best = 0.0;
            for (std::int32_t k = begin; k < end; ++k) {
                const double v = std::fabs(a.values()[k]) *
                                  s.col_scale[static_cast<std::size_t>(a.col_idx()[k])];
                best = std::max(best, v);
            }
            row_norm[static_cast<std::size_t>(i)] = s.row_scale[static_cast<std::size_t>(i)] * best;
        }

        std::vector<double> col_norm(static_cast<std::size_t>(n), 0.0);
        for (std::int32_t j = 0; j < n; ++j) {
            const std::int32_t begin = a_csc.col_ptr()[j];
            const std::int32_t end = a_csc.col_ptr()[j + 1];
            double best = 0.0;
            for (std::int32_t k = begin; k < end; ++k) {
                const double v = std::fabs(a_csc.values()[k]) *
                                  s.row_scale[static_cast<std::size_t>(a_csc.row_idx()[k])];
                best = std::max(best, v);
            }
            col_norm[static_cast<std::size_t>(j)] = s.col_scale[static_cast<std::size_t>(j)] * best;
        }

        // Convergence check (docs/architecture/NUMERICS.md \S2): every row
        // AND column norm within [1-tol, 1+tol]. Empty rows/columns (norm
        // 0, no nonzeros to equilibrate) are excluded -- they carry no
        // convergence signal and must not block it.
        double max_dev = 0.0;
        for (double v : row_norm) {
            if (v > 0.0) max_dev = std::max(max_dev, std::fabs(v - 1.0));
        }
        for (double v : col_norm) {
            if (v > 0.0) max_dev = std::max(max_dev, std::fabs(v - 1.0));
        }
        if (max_dev < tol) {
            converged = true;
            break;
        }

        // Simultaneous update (both from the same pre-update snapshot) --
        // an IMPLEMENTATION DECISION: simpler than a Gauss-Seidel-style
        // alternation and still gives Ruiz's documented linear convergence
        // rate; a zero norm (empty row/column) leaves that factor
        // untouched rather than dividing by zero.
        for (std::int32_t i = 0; i < m; ++i) {
            const double rn = row_norm[static_cast<std::size_t>(i)];
            if (rn > 0.0) s.row_scale[static_cast<std::size_t>(i)] /= std::sqrt(rn);
        }
        for (std::int32_t j = 0; j < n; ++j) {
            const double cn = col_norm[static_cast<std::size_t>(j)];
            if (cn > 0.0) s.col_scale[static_cast<std::size_t>(j)] /= std::sqrt(cn);
        }
    }

    // Recovery per docs/architecture/NUMERICS.md \S2: a partially-converged
    // scaling is worse than none, because a caller that trusts it as
    // converged may under-estimate residual magnitude in scaled space.
    if (!converged) {
        return ScaleFactors::identity(m, n);
    }
    return s;
}

CSRMatrix apply_ruiz_scaling(const CSRMatrix& a, const ScaleFactors& factors) {
    std::vector<std::int32_t> row_ptr(a.row_ptr(), a.row_ptr() + a.rows() + 1);
    std::vector<std::int32_t> col_idx(a.col_idx(), a.col_idx() + a.nnz());
    std::vector<double> values(static_cast<std::size_t>(a.nnz()));

    for (std::int32_t i = 0; i < a.rows(); ++i) {
        const std::int32_t begin = a.row_ptr()[i];
        const std::int32_t end = a.row_ptr()[i + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            values[static_cast<std::size_t>(k)] =
                a.values()[k] * factors.row_scale[static_cast<std::size_t>(i)] *
                factors.col_scale[static_cast<std::size_t>(a.col_idx()[k])];
        }
    }
    return CSRMatrix(a.rows(), a.cols(), std::move(row_ptr), std::move(col_idx), std::move(values));
}

ScaleFactors compute_pock_chambolle_scaling(const CSRMatrix& a, double alpha) {
    const auto m = static_cast<std::size_t>(a.rows());
    const auto n = static_cast<std::size_t>(a.cols());
    ScaleFactors s = ScaleFactors::identity(a.rows(), a.cols());

    std::vector<double> row_acc(m, 0.0);
    std::vector<double> col_acc(n, 0.0);

    const double row_exp = 2.0 - alpha;
    const std::int32_t* row_ptr = a.row_ptr();
    const std::int32_t* col_idx = a.col_idx();
    const double* values = a.values();

    for (std::int32_t i = 0; i < a.rows(); ++i) {
        for (std::int32_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            const double v = std::fabs(values[k]);
            if (v == 0.0) continue;
            row_acc[static_cast<std::size_t>(i)] += std::pow(v, row_exp);
            col_acc[static_cast<std::size_t>(col_idx[k])] += std::pow(v, alpha);
        }
    }

    // An empty row or column contributes no constraint and no variable
    // coupling; leaving its factor at 1 keeps the transform invertible,
    // which is the invariant every scaling in this engine must preserve
    // (docs/architecture/NUMERICS.md 2) -- a zero or infinite factor would
    // make unscaling impossible.
    for (std::size_t i = 0; i < m; ++i) {
        if (row_acc[i] > 0.0) s.row_scale[i] = 1.0 / std::sqrt(row_acc[i]);
    }
    for (std::size_t j = 0; j < n; ++j) {
        if (col_acc[j] > 0.0) s.col_scale[j] = 1.0 / std::sqrt(col_acc[j]);
    }
    return s;
}

ScaleFactors compose_scaling(const ScaleFactors& first, const ScaleFactors& second) {
    ScaleFactors out = first;
    for (std::size_t i = 0; i < out.row_scale.size(); ++i) out.row_scale[i] *= second.row_scale[i];
    for (std::size_t j = 0; j < out.col_scale.size(); ++j) out.col_scale[j] *= second.col_scale[j];
    return out;
}

} // namespace sihps
