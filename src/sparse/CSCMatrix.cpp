#include "CSCMatrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sihps {

CSCMatrix::CSCMatrix(std::int32_t rows, std::int32_t cols,
                      std::vector<std::int32_t> col_ptr,
                      std::vector<std::int32_t> row_idx,
                      std::vector<double> values)
    : rows_(rows), cols_(cols), col_ptr_(std::move(col_ptr)),
      row_idx_(std::move(row_idx)), values_(std::move(values)) {
    validate();
}

CSCMatrix CSCMatrix::from_triplets(std::int32_t rows, std::int32_t cols,
                                    const std::vector<Triplet>& triplets) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("CSCMatrix::from_triplets: negative dimension");
    }
    for (const auto& t : triplets) {
        if (t.row < 0 || t.row >= rows || t.col < 0 || t.col >= cols) {
            throw std::invalid_argument("CSCMatrix::from_triplets: index out of range");
        }
    }

    std::vector<std::int32_t> col_start(cols + 1, 0);
    for (const auto& t : triplets) {
        col_start[t.col + 1] += 1;
    }
    for (std::int32_t j = 0; j < cols; ++j) {
        col_start[j + 1] += col_start[j];
    }

    std::vector<std::int32_t> cursor = col_start;
    std::vector<std::int32_t> bucketed_row(triplets.size());
    std::vector<double> bucketed_val(triplets.size());
    for (const auto& t : triplets) {
        std::int32_t pos = cursor[t.col]++;
        bucketed_row[pos] = t.row;
        bucketed_val[pos] = t.value;
    }

    std::vector<std::int32_t> final_col_ptr(cols + 1, 0);
    std::vector<std::int32_t> final_row_idx;
    std::vector<double> final_values;
    final_row_idx.reserve(triplets.size());
    final_values.reserve(triplets.size());

    std::vector<std::pair<std::int32_t, double>> col_buf;
    for (std::int32_t c = 0; c < cols; ++c) {
        const std::int32_t begin = col_start[c];
        const std::int32_t end = col_start[c + 1];
        col_buf.clear();
        col_buf.reserve(static_cast<std::size_t>(end - begin));
        for (std::int32_t k = begin; k < end; ++k) {
            col_buf.emplace_back(bucketed_row[k], bucketed_val[k]);
        }
        std::sort(col_buf.begin(), col_buf.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::size_t k = 0;
        while (k < col_buf.size()) {
            const std::int32_t r = col_buf[k].first;
            double sum = col_buf[k].second;
            ++k;
            while (k < col_buf.size() && col_buf[k].first == r) {
                sum += col_buf[k].second;
                ++k;
            }
            final_row_idx.push_back(r);
            final_values.push_back(sum);
        }
        final_col_ptr[c + 1] = static_cast<std::int32_t>(final_row_idx.size());
    }

    return CSCMatrix(rows, cols, std::move(final_col_ptr), std::move(final_row_idx),
                      std::move(final_values));
}

void CSCMatrix::validate() const {
    if (rows_ < 0 || cols_ < 0) {
        throw std::invalid_argument("CSCMatrix: negative dimension");
    }
    if (static_cast<std::int32_t>(col_ptr_.size()) != cols_ + 1) {
        throw std::invalid_argument("CSCMatrix: col_ptr size must be cols()+1");
    }
    if (col_ptr_.empty() || col_ptr_.front() != 0) {
        throw std::invalid_argument("CSCMatrix: col_ptr[0] must be 0");
    }
    for (std::int32_t j = 0; j < cols_; ++j) {
        if (col_ptr_[static_cast<std::size_t>(j) + 1] < col_ptr_[static_cast<std::size_t>(j)]) {
            throw std::invalid_argument("CSCMatrix: col_ptr must be non-decreasing");
        }
    }
    if (col_ptr_.back() != static_cast<std::int32_t>(row_idx_.size()) ||
        row_idx_.size() != values_.size()) {
        throw std::invalid_argument("CSCMatrix: col_ptr/row_idx/values size mismatch");
    }
    for (std::int32_t r : row_idx_) {
        if (r < 0 || r >= rows_) {
            throw std::invalid_argument("CSCMatrix: row index out of range");
        }
    }
}

void CSCMatrix::multiply(const double* x, double* y) const {
    for (std::int32_t i = 0; i < rows_; ++i) {
        y[i] = 0.0;
    }
    for (std::int32_t j = 0; j < cols_; ++j) {
        const double xj = x[j];
        const std::int32_t begin = col_ptr_[static_cast<std::size_t>(j)];
        const std::int32_t end = col_ptr_[static_cast<std::size_t>(j) + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            y[row_idx_[static_cast<std::size_t>(k)]] += values_[static_cast<std::size_t>(k)] * xj;
        }
    }
}

} // namespace sihps
