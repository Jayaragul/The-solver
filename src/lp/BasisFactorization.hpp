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
    // pattern_[top..m-1], in topological order). Used only during
    // factorize(), where Li_ is still matrix-row-indexed and pinv_ is
    // still being filled in incrementally -- NOT reusable post-
    // factorization, where both are in their final pivot-step-indexed
    // form. See reach() below for the post-factorization equivalent.
    std::int32_t sparse_lsolve(const SparseColumn& column, std::int32_t mark_value);

    // Builds a row-major (CSR) transpose of a CSC column structure that
    // stores its own diagonal as either the FIRST entry per column (L) or
    // the LAST (U), excluding that diagonal entry -- the transpose is
    // used only as a dependency graph for reach() below, and the diagonal
    // never represents a dependency edge. Called once per factorize(),
    // for both L and U, since neither changes again before the next
    // refactorization (only the eta file does).
    void build_transpose(const std::vector<std::int32_t>& col_ptr,
                          const std::vector<std::int32_t>& row_idx, bool diagonal_first,
                          std::vector<std::int32_t>& t_ptr, std::vector<std::int32_t>& t_idx);

    // DFS reachability, in topological order, over the graph given by
    // (t_ptr, t_idx) -- a row-major transpose of L's or U's off-diagonal
    // structure, entirely in pivot-step space (no pinv_ indirection
    // needed, unlike sparse_lsolve) -- seeded from `seeds`. Returns the
    // start index into pattern_ of the discovered pattern
    // (pattern_[top..m_-1]). Structurally identical to sparse_lsolve's
    // DFS, generalized to take the graph as a parameter since it is
    // reused for both L^T's and U^T's reachability in btran().
    std::int32_t reach(const std::vector<std::int32_t>& t_ptr, const std::vector<std::int32_t>& t_idx,
                        const std::vector<std::int32_t>& seeds, std::int32_t mark_value) const;

    std::int32_t m_ = 0;

    // P B Q = L U, with P given by pinv_ (matrix row -> pivot step) and Q
    // by q_ (pivot step -> basis position). After factorization L's row
    // indices are remapped through pinv_, so both factors are triangular in
    // pivot-step space and the solves need no further indirection.
    std::vector<std::int32_t> pinv_, q_;
    std::vector<std::int32_t> Lp_, Li_, Up_, Ui_;
    std::vector<double> Lx_, Ux_;

    // Row-major (CSR-style) transposes of L's and U's off-diagonal
    // structure -- no values, structure only. Built once per factorize()
    // (Section: build_transpose above) and consumed by btran()'s
    // hyper-sparse reachability DFS, which needs "which columns have a
    // nonzero in row i" -- the opposite query direction from Lp_/Li_/
    // Up_/Ui_'s column-major storage.
    std::vector<std::int32_t> Lt_p_, Lt_i_, Ut_p_, Ut_i_;

    // Eta file. Eta k pivots at basis position eta_pivot_[k] with pivot
    // value eta_pivot_value_[k]; its off-pivot entries are
    // eta_index_/eta_value_ over [eta_ptr_[k], eta_ptr_[k+1]).
    std::vector<std::int32_t> eta_pivot_, eta_index_, eta_ptr_;
    std::vector<double> eta_pivot_value_, eta_value_;

    // Factorization + solve scratch, sized once per factorize() so the hot
    // path performs no allocation (prompt.md \S3.1). mutable: reach() and
    // btran() are const (called from Simplex's const methods) but still
    // need this scratch space -- exactly the same reasoning that already
    // makes work_/permuted_ mutable.
    mutable std::vector<double> work_, permuted_;
    mutable std::vector<std::int32_t> pattern_, dfs_node_, dfs_next_, mark_;
    // btran()'s own scratch: the seed pattern from the post-gather scan,
    // and (when reach() runs rather than the density fallback) a copy of
    // the U^T phase's discovered pattern to seed the L^T phase with.
    // Members rather than locals for the same reason as everything else
    // in this list -- btran() runs every simplex iteration, and a fresh
    // heap allocation per call is exactly what prompt.md \S3.1 rules out.
    // MEASURED: on MIPLIB node relaxations (m in the 7-90 range, called
    // across hundreds of thousands of B&B nodes), the original per-call
    // std::vector locals cost 4-7% of node throughput -- small dense
    // solves are cheap enough that allocation overhead, not DFS work,
    // dominated.
    mutable std::vector<std::int32_t> seed_work_, ut_pattern_work_;
    std::vector<std::int32_t> row_count_;
    // Monotonic mark counter for reach()'s two per-btran() DFS calls
    // (U^T then L^T), independent of sparse_lsolve's own step-indexed
    // marks -- reset whenever factorize() re-initializes mark_, so it
    // never needs to worry about colliding with a previous factorization's
    // marks, only with earlier btran() calls since the last factorize().
    mutable std::int32_t solve_mark_ = 0;
};

} // namespace sihps
