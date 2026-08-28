#include "Convert.hpp"

#include <vector>

namespace sihps {
namespace {

// Reinterprets a (ptr_from, idx_from, values_from) structure -- n_from
// outer buckets, n_to inner index range -- as its transpose: n_to outer
// buckets, n_from inner index range. Used for both CSR->CSC (outer=row,
// inner=col) and CSC->CSR (outer=col, inner=row) by swapping arguments.
void bucket_transpose(std::int32_t n_from, std::int32_t n_to,
                      const std::int32_t* ptr_from, const std::int32_t* idx_from,
                      const double* values_from, std::int32_t nnz,
                      std::vector<std::int32_t>& ptr_to,
                      std::vector<std::int32_t>& idx_to,
                      std::vector<double>& values_to) {
    ptr_to.assign(static_cast<std::size_t>(n_to) + 1, 0);
    for (std::int32_t k = 0; k < nnz; ++k) {
        ptr_to[static_cast<std::size_t>(idx_from[k]) + 1] += 1;
    }
    for (std::int32_t i = 0; i < n_to; ++i) {
        ptr_to[static_cast<std::size_t>(i) + 1] += ptr_to[static_cast<std::size_t>(i)];
    }

    idx_to.assign(static_cast<std::size_t>(nnz), 0);
    values_to.assign(static_cast<std::size_t>(nnz), 0.0);
    std::vector<std::int32_t> cursor(ptr_to.begin(), ptr_to.end());

    for (std::int32_t outer = 0; outer < n_from; ++outer) {
        const std::int32_t begin = ptr_from[outer];
        const std::int32_t end = ptr_from[static_cast<std::size_t>(outer) + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            const std::int32_t inner = idx_from[k];
            const std::int32_t pos = cursor[static_cast<std::size_t>(inner)]++;
            idx_to[static_cast<std::size_t>(pos)] = outer;
            values_to[static_cast<std::size_t>(pos)] = values_from[k];
        }
    }
}

} // namespace

CSCMatrix csr_to_csc(const CSRMatrix& a) {
    std::vector<std::int32_t> col_ptr, row_idx;
    std::vector<double> values;
    bucket_transpose(a.rows(), a.cols(), a.row_ptr(), a.col_idx(), a.values(), a.nnz(),
                      col_ptr, row_idx, values);
    return CSCMatrix(a.rows(), a.cols(), std::move(col_ptr), std::move(row_idx),
                      std::move(values));
}

CSRMatrix csc_to_csr(const CSCMatrix& a) {
    std::vector<std::int32_t> row_ptr, col_idx;
    std::vector<double> values;
    bucket_transpose(a.cols(), a.rows(), a.col_ptr(), a.row_idx(), a.values(), a.nnz(),
                      row_ptr, col_idx, values);
    return CSRMatrix(a.rows(), a.cols(), std::move(row_ptr), std::move(col_idx),
                      std::move(values));
}

} // namespace sihps
