#pragma once

#include "../sparse/CSRMatrix.hpp"

#include <vector>

namespace sihps {

// Row/column scale factors from Ruiz equilibration
// (docs/architecture/NUMERICS.md \S2; docs/research/SOTA.md \S1.4.1;
// ESTABLISHED METHOD: Ruiz, "A scaling algorithm to equilibrate both rows
// and columns norms in matrices", RAL-TR-2001-034, 2001). Both vectors are
// strictly positive by construction, so scaling preserves the sign and
// ordering of every bound it is applied to -- only magnitude changes.
//
// The scaled matrix is A' = diag(row_scale) * A * diag(col_scale).
struct ScaleFactors {
    std::vector<double> row_scale; // R, size n_rows
    std::vector<double> col_scale; // C, size n_cols

    static ScaleFactors identity(std::int32_t n_rows, std::int32_t n_cols) {
        return ScaleFactors{std::vector<double>(static_cast<std::size_t>(n_rows), 1.0),
                             std::vector<double>(static_cast<std::size_t>(n_cols), 1.0)};
    }
};

// Computes Ruiz equilibration scale factors for `a`.
//
// Method: iteratively rescale each row and column by the reciprocal square
// root of its current infinity norm (Ruiz 2001). Because R and C are
// diagonal, the infinity norm of row i / column j of the CURRENTLY scaled
// matrix can be recovered as R_i * max_j(|A_ij| * C_j) / C_j-independent
// arrangement -- see Scaling.cpp -- without ever materializing a rescaled
// copy of A, keeping each iteration O(nnz).
//
// Stopping criterion (docs/architecture/NUMERICS.md \S2): iterate until
// every row and column infinity norm of the implied A' is within
// [1-tol, 1+tol], or `max_iters` is reached, whichever first.
//
// Failure condition + recovery (docs/architecture/NUMERICS.md \S2): if the
// convergence check has not passed within `max_iters` iterations, this
// function returns IDENTITY scale factors (R=C=1) rather than a
// partially-converged scaling -- proceeding with a partial equilibration
// that the caller believes is converged is explicitly documented there as
// worse than not scaling at all.
//
// An all-zero row or column (no nonzeros) has no informative norm to
// equilibrate against; its scale factor is left at its last value (1.0 if
// never touched) rather than dividing by zero.
ScaleFactors compute_ruiz_scaling(const CSRMatrix& a, int max_iters = 100, double tol = 1e-2);

// Pock-Chambolle diagonal preconditioner (Pock & Chambolle, "Diagonal
// preconditioning for first order primal-dual algorithms", ICCV 2011):
//
//     R_i = 1 / sqrt( sum_j |A_ij|^(2 - alpha) )
//     C_j = 1 / sqrt( sum_i |A_ij|^alpha )
//
// This is NOT an alternative to Ruiz equilibration -- it is applied AFTER
// it, which is what PDLP does (Applegate et al. 2021, App. B) and what this
// engine does in solve_lp's first-order path.
//
// The two do different jobs. Ruiz equalizes the magnitudes of rows and
// columns, which is what a simplex basis factorization cares about.
// Pock-Chambolle instead targets the quantity a first-order method's
// convergence rate actually depends on: with alpha = 1 the preconditioned
// operator satisfies ||R A C||_2 <= 1 by construction, so the step size no
// longer has to be shrunk to accommodate a few dominant rows. On a badly
// balanced model that is the difference between converging and stalling,
// which is why it is worth a second scaling pass rather than folding into
// the first.
//
// `alpha` in [0, 2]; 1.0 is the balanced choice and the default.
ScaleFactors compute_pock_chambolle_scaling(const CSRMatrix& a, double alpha = 1.0);

// Composes two scalings applied in sequence: `first` then `second`, where
// `second` was computed on the matrix `first` already produced. Returns the
// single (R, C) pair equivalent to both, so the caller stores one set of
// factors and unscales exactly once.
ScaleFactors compose_scaling(const ScaleFactors& first, const ScaleFactors& second);

// Returns A' = diag(factors.row_scale) * a * diag(factors.col_scale) as a
// new CSR matrix, same sparsity pattern as `a`.
CSRMatrix apply_ruiz_scaling(const CSRMatrix& a, const ScaleFactors& factors);

} // namespace sihps
