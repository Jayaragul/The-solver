#pragma once

#include <cstdint>

namespace sihps {

// One (row, col, value) coefficient contribution. The unordered,
// possibly-duplicated input format that CSR/CSC matrices are assembled
// from -- see CSRMatrix::from_triplets / CSCMatrix::from_triplets.
struct Triplet {
    std::int32_t row;
    std::int32_t col;
    double value;
};

} // namespace sihps
