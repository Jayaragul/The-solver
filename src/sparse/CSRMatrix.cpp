#include "CSRMatrix.hpp"

#include "../parallel/Parallel.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sihps {

CSRMatrix::CSRMatrix(std::int32_t rows, std::int32_t cols,
                      std::vector<std::int32_t> row_ptr,
                      std::vector<std::int32_t> col_idx,
                      std::vector<double> values)
    : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)), values_(std::move(values)) {
    validate();
}

CSRMatrix CSRMatrix::from_triplets(std::int32_t rows, std::int32_t cols,
                                    const std::vector<Triplet>& triplets) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("CSRMatrix::from_triplets: negative dimension");
    }
    for (const auto& t : triplets) {
        if (t.row < 0 || t.row >= rows || t.col < 0 || t.col >= cols) {
            throw std::invalid_argument("CSRMatrix::from_triplets: index out of range");
        }
    }

    // Bucket by row via counting sort: O(nnz + rows). One-time assembly
    // cost, not a hot-path operation.
    std::vector<std::int32_t> row_start(rows + 1, 0);
    for (const auto& t : triplets) {
        row_start[t.row + 1] += 1;
    }
    for (std::int32_t i = 0; i < rows; ++i) {
        row_start[i + 1] += row_start[i];
    }

    std::vector<std::int32_t> cursor = row_start;
    std::vector<std::int32_t> bucketed_col(triplets.size());
    std::vector<double> bucketed_val(triplets.size());
    for (const auto& t : triplets) {
        std::int32_t pos = cursor[t.row]++;
        bucketed_col[pos] = t.col;
        bucketed_val[pos] = t.value;
    }

    // Within each row bucket, sort by column and merge duplicates.
    std::vector<std::int32_t> final_row_ptr(rows + 1, 0);
    std::vector<std::int32_t> final_col_idx;
    std::vector<double> final_values;
    final_col_idx.reserve(triplets.size());
    final_values.reserve(triplets.size());

    std::vector<std::pair<std::int32_t, double>> row_buf;
    for (std::int32_t r = 0; r < rows; ++r) {
        const std::int32_t begin = row_start[r];
        const std::int32_t end = row_start[r + 1];
        row_buf.clear();
        row_buf.reserve(static_cast<std::size_t>(end - begin));
        for (std::int32_t k = begin; k < end; ++k) {
            row_buf.emplace_back(bucketed_col[k], bucketed_val[k]);
        }
        std::sort(row_buf.begin(), row_buf.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::size_t k = 0;
        while (k < row_buf.size()) {
            const std::int32_t c = row_buf[k].first;
            double sum = row_buf[k].second;
            ++k;
            while (k < row_buf.size() && row_buf[k].first == c) {
                sum += row_buf[k].second;
                ++k;
            }
            final_col_idx.push_back(c);
            final_values.push_back(sum);
        }
        final_row_ptr[r + 1] = static_cast<std::int32_t>(final_col_idx.size());
    }

    return CSRMatrix(rows, cols, std::move(final_row_ptr), std::move(final_col_idx),
                      std::move(final_values));
}

void CSRMatrix::validate() const {
    if (rows_ < 0 || cols_ < 0) {
        throw std::invalid_argument("CSRMatrix: negative dimension");
    }
    if (static_cast<std::int32_t>(row_ptr_.size()) != rows_ + 1) {
        throw std::invalid_argument("CSRMatrix: row_ptr size must be rows()+1");
    }
    if (row_ptr_.empty() || row_ptr_.front() != 0) {
        throw std::invalid_argument("CSRMatrix: row_ptr[0] must be 0");
    }
    for (std::int32_t i = 0; i < rows_; ++i) {
        if (row_ptr_[static_cast<std::size_t>(i) + 1] < row_ptr_[static_cast<std::size_t>(i)]) {
            throw std::invalid_argument("CSRMatrix: row_ptr must be non-decreasing");
        }
    }
    if (row_ptr_.back() != static_cast<std::int32_t>(col_idx_.size()) ||
        col_idx_.size() != values_.size()) {
        throw std::invalid_argument("CSRMatrix: row_ptr/col_idx/values size mismatch");
    }
    for (std::int32_t c : col_idx_) {
        if (c < 0 || c >= cols_) {
            throw std::invalid_argument("CSRMatrix: column index out of range");
        }
    }
}

void CSRMatrix::multiply(const double* x, double* y, ParallelMode parallel_mode) const {
    // Row-parallel and bit-reproducible: row i's accumulator is private to
    // one iteration and summed in ascending k, exactly as the serial loop
    // does, so the answer does not depend on the thread count (see
    // parallel/Parallel.hpp). A CSC-ordered product could not make the
    // same claim -- its scatter-add would let the summation order follow
    // the scheduler -- which is why CSCMatrix::multiply is left serial.
    SIHPS_OMP(omp parallel for schedule(static) if(parallel_mode == ParallelMode::PARALLEL ||
                                                   (parallel_mode == ParallelMode::AUTO &&
                                                    nnz() >= kParallelNnzThreshold)))
    for (std::int32_t i = 0; i < rows_; ++i) {
        double acc = 0.0;
        const std::int32_t begin = row_ptr_[static_cast<std::size_t>(i)];
        const std::int32_t end = row_ptr_[static_cast<std::size_t>(i) + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            acc += values_[static_cast<std::size_t>(k)] * x[col_idx_[static_cast<std::size_t>(k)]];
        }
        y[i] = acc;
    }
}

} // namespace sihps
