#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace sihps {

// A basis column in (matrix row index, value) form.
using SparseColumn = std::vector<std::pair<std::int32_t, double>>;

// Sparse LU factorization of the simplex basis matrix B, with product-form
// (eta) updates between refactorizations.
//
// WHY THIS EXISTS (the mathematical/computational bottleneck it solves):
// the previous implementation stored B^-1 explicitly as a dense m x m
// array. That is O(m^2) memory and O(m^2) work per pivot, with O(m^3)
// refactorization -- tractable to a few hundred rows and hopeless beyond.
// At Netlib scale that is already fatal (dfl001, m=6072: 295 MB and
// ~2.2e11 flops per refactorization; ken-18, m=105127: 88 TB), and
// refinery-scale models are larger still. Sparse LU makes both storage and
// solve cost proportional to the nonzeros in the factors instead of to m^2
// -- this is the single change that decides whether the engine scales at
// all.
//
// ESTABLISHED METHOD, two independent literature lineages:
//
//  1. Factorization: left-looking sparse LU with a depth-first-search
//     symbolic phase (Gilbert & Peierls, "Sparse partial pivoting in time
//     proportional to arithmetic operations", SIAM J. Sci. Stat. Comput.
//     9(5), 1988). Each column is computed by a sparse triangular solve
//     against the already-factored columns, at a cost proportional to the
//     arithmetic actually performed rather than to m.
//
//  2. Updates: product form of the inverse (Dantzig & Orchard-Hays, "The
//     product form for the inverse in the simplex method", Mathematical
//     Tables and Other Aids to Computation 8, 1954). A basis change is
//     recorded as one additional sparse eta vector rather than by redoing
//     the factorization.
//
// DEVIATION FROM docs/architecture/LP.md \S4, stated explicitly: LP.md
// specifies Bartels-Golub / Forrest-Tomlin updates, which modify the U
// factor in place and preserve sparsity better than PFI over long update
// sequences. PFI is implemented here instead because it is markedly
// simpler to get numerically right, and because its known weakness (eta
// file growth) is bounded by the same periodic-refactorization policy
// LP.md already mandates. Forrest-Tomlin remains the correct upgrade if
// benchmarking shows eta growth, not factorization, dominates -- that is a
// measurement to make, not an assumption to act on now.
//
// Numerical policy (docs/architecture/NUMERICS.md): FP64 throughout;
// threshold partial pivoting bounds multiplier growth; a pivot candidate
// is rejected outright below an absolute floor. Structurally or
// numerically singular columns are REPORTED, not silently accepted -- see
// FactorizeResult::singular.
class BasisFactorization {
public:
    struct FactorizeResult {
        bool ok = false;

        // Basis positions for which no numerically acceptable pivot
        // existed, each paired with a matrix row left unmatched. A
        // singular basis is a legitimate runtime event in simplex (a
        // degenerate pivot sequence can produce one), so it is surfaced
        // for repair rather than thrown: the caller substitutes that row's
        // logical column and refactorizes (NUMERICS.md \S5, basis repair).
        // When this is non-empty the factorization is still USABLE -- the
        // offending columns were replaced by unit columns internally -- but
        // it factorizes a DIFFERENT matrix than the one requested, so the
        // caller must repair the basis to match before trusting a solve.
        std::vector<std::pair<std::int32_t, std::int32_t>> singular;
    };

    // Factorizes the m x m basis whose column at position r is columns[r].
    // Discards any accumulated eta file. Row indices in `columns` are
    // matrix rows; positions are basis rows.
    FactorizeResult factorize(std::int32_t m, const std::vector<SparseColumn>& columns);

    // Solves B x = b. On entry `x` holds b indexed by MATRIX ROW; on exit
    // it holds x indexed by BASIS POSITION.
    void ftran(std::vector<double>& x) const;

    // Solves B^T y = c. On entry `x` holds c indexed by BASIS POSITION; on
    // exit it holds y indexed by MATRIX ROW.
    void btran(std::vector<double>& x) const;

    // Records a basis change in product form: the column at basis position
    // `leaving_pos` is replaced by the entering column. `ftran_col` must be
    // B^-1 a_entering computed against the CURRENT factorization (i.e. the
    // output of ftran, including all etas already applied) -- that is
    // exactly the PFI recursion B_new^-1 = E^-1 B_old^-1.
    //
    // Returns false if the pivot element is too small to invert safely; the
    // caller must then refactorize rather than proceed on a corrupted
    // representation.
    bool update(std::int32_t leaving_pos, const std::vector<double>& ftran_col);

    std::int32_t eta_count() const noexcept {
        return static_cast<std::int32_t>(eta_pivot_.size());
    }
    std::int32_t factor_nnz() const noexcept {
        return static_cast<std::int32_t>(Lx_.size() + Ux_.size());
    }
    std::int32_t eta_nnz() const noexcept { return static_cast<std::int32_t>(eta_value_.size()); }

private:
    // Sparse triangular solve x = L \ column, returning the start index of
    // the computed nonzero pattern within pattern_ (entries
    // pattern_[top..m-1], in topological order).
    std::int32_t sparse_lsolve(const SparseColumn& column, std::int32_t mark_value);

    std::int32_t m_ = 0;

    // P B Q = L U, with P given by pinv_ (matrix row -> pivot step) and Q
    // by q_ (pivot step -> basis position). After factorization L's row
    // indices are remapped through pinv_, so both factors are triangular in
    // pivot-step space and the solves need no further indirection.
    std::vector<std::int32_t> pinv_, q_;
    std::vector<std::int32_t> Lp_, Li_, Up_, Ui_;
    std::vector<double> Lx_, Ux_;

    // Eta file. Eta k pivots at basis position eta_pivot_[k] with pivot
    // value eta_pivot_value_[k]; its off-pivot entries are
    // eta_index_/eta_value_ over [eta_ptr_[k], eta_ptr_[k+1]).
    std::vector<std::int32_t> eta_pivot_, eta_index_, eta_ptr_;
    std::vector<double> eta_pivot_value_, eta_value_;

    // Factorization + solve scratch, sized once per factorize() so the hot
    // path performs no allocation (prompt.md \S3.1).
    mutable std::vector<double> work_, permuted_;
    std::vector<std::int32_t> pattern_, dfs_node_, dfs_next_, mark_, row_count_;
};

} // namespace sihps
