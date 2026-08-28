#pragma once

#include "Triplet.hpp"

#include <cstdint>
#include <vector>

#include "../parallel/Parallel.hpp"

namespace sihps {

// Compressed Sparse Row.
//
// Index type is int32_t: refinery-scale models (docs/research/SOTA.md) are
// not expected to approach 2^31 rows/cols/nnz, and the narrower index
// halves matrix memory traffic relative to int64 -- directly relevant to
// SpMV's memory-bandwidth-bound cost model (docs/architecture/CPU_GPU.md
// \S2.1). This also matches cusparseIndexType_t's CUSPARSE_INDEX_32I,
// avoiding an index-width mismatch when this matrix is later mirrored to
// device memory.
class CSRMatrix {
public:
    CSRMatrix() : rows_(0), cols_(0) {}

    // Takes ownership of already-built CSR arrays. Validates on
    // construction -- a CSRMatrix that exists is a CSRMatrix that has
    // already passed structural validation.
    CSRMatrix(std::int32_t rows, std::int32_t cols,
              std::vector<std::int32_t> row_ptr,
              std::vector<std::int32_t> col_idx,
              std::vector<double> values);

    // Builds from an unordered triplet list. Duplicate (row, col) entries
    // are summed -- standard sparse-assembly semantics (IMPLEMENTATION
    // DECISION): a modeling layer that emits the same coefficient twice is
    // treated as accumulating a contribution, not as a silent
    // last-write-wins overwrite.
    static CSRMatrix from_triplets(std::int32_t rows, std::int32_t cols,
                                    const std::vector<Triplet>& triplets);

    std::int32_t rows() const noexcept { return rows_; }
    std::int32_t cols() const noexcept { return cols_; }
    std::int32_t nnz() const noexcept { return static_cast<std::int32_t>(values_.size()); }

    const std::int32_t* row_ptr() const noexcept { return row_ptr_.data(); }
    const std::int32_t* col_idx() const noexcept { return col_idx_.data(); }
    const double* values() const noexcept { return values_.data(); }

    // Throws std::invalid_argument on any structural violation: wrong
    // row_ptr length, non-monotonic row_ptr, size mismatches, or an
    // out-of-range column index.
    void validate() const;

    // y = A * x. Caller owns x (size cols()) and y (size rows()); y is
    // overwritten, not accumulated into. Row-parallel-friendly (each row's
    // work is independent) and reads values_/col_idx_ contiguously within
    // a row -- the cache-locality argument in docs/architecture/CPU_GPU.md
    // \S2.1 for why this loop shape is the right default before any
    // OpenMP/GPU offload is justified by measurement (prompt.md \S3.2: "do
    // not prematurely over-optimize without benchmark evidence").
    void multiply(const double* x, double* y,
                  ParallelMode parallel_mode = ParallelMode::AUTO) const;

private:
    std::int32_t rows_, cols_;
    std::vector<std::int32_t> row_ptr_;
    std::vector<std::int32_t> col_idx_;
    std::vector<double> values_;
};

} // namespace sihps
