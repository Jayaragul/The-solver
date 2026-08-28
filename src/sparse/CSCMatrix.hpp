#pragma once

#include "Triplet.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

// Compressed Sparse Column -- the mirror image of CSRMatrix. Kept as a
// distinct concrete type (not a template shared with CSRMatrix) since the
// two have different physical layouts and different roles in the LP
// engine (A vs A^T access patterns, docs/architecture/LP.md); sharing a
// template here would trade a small amount of duplication for a layer of
// abstraction neither type otherwise needs.
class CSCMatrix {
public:
    CSCMatrix() : rows_(0), cols_(0) {}

    CSCMatrix(std::int32_t rows, std::int32_t cols,
              std::vector<std::int32_t> col_ptr,
              std::vector<std::int32_t> row_idx,
              std::vector<double> values);

    // Duplicate (row, col) entries are summed -- same semantics as
    // CSRMatrix::from_triplets.
    static CSCMatrix from_triplets(std::int32_t rows, std::int32_t cols,
                                    const std::vector<Triplet>& triplets);

    std::int32_t rows() const noexcept { return rows_; }
    std::int32_t cols() const noexcept { return cols_; }
    std::int32_t nnz() const noexcept { return static_cast<std::int32_t>(values_.size()); }

    const std::int32_t* col_ptr() const noexcept { return col_ptr_.data(); }
    const std::int32_t* row_idx() const noexcept { return row_idx_.data(); }
    const double* values() const noexcept { return values_.data(); }

    void validate() const;

    // y = A * x, via column-oriented scatter-add: for each column j,
    // y[row_idx[k]] += values[k] * x[j]. multiply() zeroes y itself (size
    // rows()) before accumulating, so the contract matches
    // CSRMatrix::multiply's "y is overwritten" semantics exactly, despite
    // the different internal access pattern (scattered writes to y here,
    // vs. scattered reads from x in CSRMatrix::multiply).
    void multiply(const double* x, double* y) const;

private:
    std::int32_t rows_, cols_;
    std::vector<std::int32_t> col_ptr_;
    std::vector<std::int32_t> row_idx_;
    std::vector<double> values_;
};

} // namespace sihps
