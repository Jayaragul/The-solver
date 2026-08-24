#pragma once

#include "CSCMatrix.hpp"
#include "CSRMatrix.hpp"

namespace sihps {

// Both conversions are O(nnz + max(rows, cols)) bucket-sort "transpose"
// operations -- no comparison sort is needed, because traversing the
// source structure's outer index in increasing order and appending to each
// destination bucket in that same order leaves every destination bucket
// already sorted by its own outer index (see Convert.cpp).
CSCMatrix csr_to_csc(const CSRMatrix& a);
CSRMatrix csc_to_csr(const CSCMatrix& a);

} // namespace sihps
