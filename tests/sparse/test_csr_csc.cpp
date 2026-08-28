#include "../test_framework.hpp"
#include "sparse/CSCMatrix.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Convert.hpp"

#include <random>
#include <vector>

using sihps::CSCMatrix;
using sihps::CSRMatrix;
using sihps::Triplet;

namespace {

// Dense reference multiply, for correctness checks independent of any
// sparse-specific code path.
std::vector<double> dense_multiply(std::int32_t rows, std::int32_t cols,
                                    const std::vector<double>& dense_row_major,
                                    const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(rows), 0.0);
    for (std::int32_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (std::int32_t j = 0; j < cols; ++j) {
            acc += dense_row_major[static_cast<std::size_t>(i) * cols + j] * x[j];
        }
        y[i] = acc;
    }
    return y;
}

std::vector<Triplet> triplets_from_dense(std::int32_t rows, std::int32_t cols,
                                          const std::vector<double>& dense_row_major) {
    std::vector<Triplet> t;
    for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t j = 0; j < cols; ++j) {
            double v = dense_row_major[static_cast<std::size_t>(i) * cols + j];
            if (v != 0.0) t.push_back({i, j, v});
        }
    }
    return t;
}

std::vector<Triplet> random_triplets(std::int32_t rows, std::int32_t cols, double density,
                                      unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);
    std::vector<Triplet> t;
    for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t j = 0; j < cols; ++j) {
            if (unit(rng) < density) {
                t.push_back({i, j, val(rng)});
            }
        }
    }
    return t;
}

} // namespace

SIHPS_TEST(csr_empty_matrix) {
    CSRMatrix a = CSRMatrix::from_triplets(0, 0, {});
    SIHPS_ASSERT_EQ(a.rows(), 0);
    SIHPS_ASSERT_EQ(a.cols(), 0);
    SIHPS_ASSERT_EQ(a.nnz(), 0);
}

SIHPS_TEST(csr_matrix_with_zero_nnz_but_positive_dimensions) {
    CSRMatrix a = CSRMatrix::from_triplets(5, 5, {});
    SIHPS_ASSERT_EQ(a.rows(), 5);
    SIHPS_ASSERT_EQ(a.nnz(), 0);
    std::vector<double> x(5, 1.0), y(5, -1.0);
    a.multiply(x.data(), y.data());
    for (int i = 0; i < 5; ++i) SIHPS_ASSERT_EQ(y[i], 0.0);
}

SIHPS_TEST(csr_diagonal_matrix) {
    std::vector<Triplet> t = {{0, 0, 2.0}, {1, 1, 3.0}, {2, 2, 4.0}};
    CSRMatrix a = CSRMatrix::from_triplets(3, 3, t);
    std::vector<double> x = {1.0, 1.0, 1.0}, y(3);
    a.multiply(x.data(), y.data());
    SIHPS_ASSERT_EQ(y[0], 2.0);
    SIHPS_ASSERT_EQ(y[1], 3.0);
    SIHPS_ASSERT_EQ(y[2], 4.0);
}

SIHPS_TEST(csr_dense_matrix_represented_sparsely) {
    std::int32_t rows = 4, cols = 4;
    std::vector<double> dense = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    auto t = triplets_from_dense(rows, cols, dense);
    CSRMatrix a = CSRMatrix::from_triplets(rows, cols, t);
    SIHPS_ASSERT_EQ(a.nnz(), 16);
    std::vector<double> x = {1.0, 0.5, -1.0, 2.0};
    std::vector<double> y(rows);
    a.multiply(x.data(), y.data());
    auto y_ref = dense_multiply(rows, cols, dense, x);
    for (std::int32_t i = 0; i < rows; ++i) SIHPS_ASSERT_NEAR(y[i], y_ref[i], 1e-12);
}

SIHPS_TEST(csr_rectangular_matrix) {
    // 2 rows, 5 cols
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 4, 2.0}, {1, 2, 3.0}};
    CSRMatrix a = CSRMatrix::from_triplets(2, 5, t);
    std::vector<double> x = {1, 1, 1, 1, 1};
    std::vector<double> y(2);
    a.multiply(x.data(), y.data());
    SIHPS_ASSERT_EQ(y[0], 3.0);
    SIHPS_ASSERT_EQ(y[1], 3.0);
}

SIHPS_TEST(csr_zero_rows_have_no_effect) {
    // row 1 is entirely empty
    std::vector<Triplet> t = {{0, 0, 5.0}, {2, 1, 7.0}};
    CSRMatrix a = CSRMatrix::from_triplets(3, 2, t);
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> y(3);
    a.multiply(x.data(), y.data());
    SIHPS_ASSERT_EQ(y[0], 5.0);
    SIHPS_ASSERT_EQ(y[1], 0.0);
    SIHPS_ASSERT_EQ(y[2], 7.0);
}

SIHPS_TEST(csr_zero_columns_matrix) {
    CSRMatrix a = CSRMatrix::from_triplets(3, 0, {});
    SIHPS_ASSERT_EQ(a.cols(), 0);
    SIHPS_ASSERT_EQ(a.nnz(), 0);
    std::vector<double> y(3, -1.0);
    a.multiply(nullptr, y.data()); // no columns to read from x
    for (int i = 0; i < 3; ++i) SIHPS_ASSERT_EQ(y[i], 0.0);
}

SIHPS_TEST(csr_duplicate_triplets_are_summed) {
    std::vector<Triplet> t = {{0, 0, 2.0}, {0, 0, 3.0}};
    CSRMatrix a = CSRMatrix::from_triplets(1, 1, t);
    SIHPS_ASSERT_EQ(a.nnz(), 1);
    std::vector<double> x = {1.0}, y(1);
    a.multiply(x.data(), y.data());
    SIHPS_ASSERT_EQ(y[0], 5.0);
}

SIHPS_TEST(csr_out_of_range_index_throws) {
    std::vector<Triplet> t = {{0, 5, 1.0}};
    SIHPS_ASSERT_THROWS(CSRMatrix::from_triplets(3, 3, t));
}

SIHPS_TEST(csr_random_sparse_matrix_matches_dense_reference) {
    std::int32_t n = 40;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> val(-5.0, 5.0);
    std::vector<double> dense(static_cast<std::size_t>(n) * n, 0.0);
    for (std::int32_t i = 0; i < n; ++i) {
        for (std::int32_t j = 0; j < n; ++j) {
            if (unit(rng) < 0.1) dense[static_cast<std::size_t>(i) * n + j] = val(rng);
        }
    }
    auto t = triplets_from_dense(n, n, dense);
    CSRMatrix a = CSRMatrix::from_triplets(n, n, t);

    std::vector<double> x(static_cast<std::size_t>(n));
    for (auto& v : x) v = val(rng);
    std::vector<double> y(static_cast<std::size_t>(n));
    a.multiply(x.data(), y.data());
    auto y_ref = dense_multiply(n, n, dense, x);
    for (std::int32_t i = 0; i < n; ++i) SIHPS_ASSERT_NEAR(y[i], y_ref[i], 1e-9);
}

SIHPS_TEST(csr_large_sparse_matrix_is_internally_consistent) {
    std::int32_t n = 2000;
    auto t = random_triplets(n, n, 0.001, 777);
    CSRMatrix a = CSRMatrix::from_triplets(n, n, t);
    std::vector<double> x(static_cast<std::size_t>(n), 1.0);
    std::vector<double> y(static_cast<std::size_t>(n));
    a.multiply(x.data(), y.data());
    // Row sum check: with x all-ones, y[i] must equal the sum of row i's
    // stored values -- an independent identity that doesn't rely on a
    // second dense implementation for a matrix this large.
    for (std::int32_t i = 0; i < n; ++i) {
        double expected = 0.0;
        for (std::int32_t k = a.row_ptr()[i]; k < a.row_ptr()[i + 1]; ++k) {
            expected += a.values()[k];
        }
        SIHPS_ASSERT_NEAR(y[i], expected, 1e-9);
    }
}

SIHPS_TEST(csr_csc_conversion_round_trip_preserves_values) {
    auto t = random_triplets(15, 12, 0.2, 42);
    CSRMatrix csr = CSRMatrix::from_triplets(15, 12, t);
    CSCMatrix csc = sihps::csr_to_csc(csr);
    CSRMatrix back = sihps::csc_to_csr(csc);

    SIHPS_ASSERT_EQ(csr.nnz(), csc.nnz());
    SIHPS_ASSERT_EQ(csr.nnz(), back.nnz());
    for (std::int32_t i = 0; i <= csr.rows(); ++i) {
        SIHPS_ASSERT_EQ(csr.row_ptr()[i], back.row_ptr()[i]);
    }
    for (std::int32_t k = 0; k < csr.nnz(); ++k) {
        SIHPS_ASSERT_EQ(csr.col_idx()[k], back.col_idx()[k]);
        SIHPS_ASSERT_NEAR(csr.values()[k], back.values()[k], 1e-15);
    }
}

SIHPS_TEST(csr_and_csc_multiply_agree) {
    auto t = random_triplets(25, 20, 0.15, 99);
    CSRMatrix csr = CSRMatrix::from_triplets(25, 20, t);
    CSCMatrix csc = sihps::csr_to_csc(csr);

    std::mt19937 rng(1);
    std::uniform_real_distribution<double> val(-3.0, 3.0);
    std::vector<double> x(20);
    for (auto& v : x) v = val(rng);

    std::vector<double> y_csr(25), y_csc(25);
    csr.multiply(x.data(), y_csr.data());
    csc.multiply(x.data(), y_csc.data());

    for (int i = 0; i < 25; ++i) SIHPS_ASSERT_NEAR(y_csr[i], y_csc[i], 1e-12);
}

SIHPS_TEST(csc_direct_construction_matches_csr_reference) {
    std::int32_t rows = 4, cols = 4;
    std::vector<double> dense = {
        1, 0, 0, 4,
        0, 6, 0, 0,
        9, 0, 11, 0,
        0, 14, 0, 16,
    };
    auto t = triplets_from_dense(rows, cols, dense);
    CSRMatrix csr = CSRMatrix::from_triplets(rows, cols, t);
    CSCMatrix csc = CSCMatrix::from_triplets(rows, cols, t);

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_csr(rows), y_csc(rows);
    csr.multiply(x.data(), y_csr.data());
    csc.multiply(x.data(), y_csc.data());
    auto y_ref = dense_multiply(rows, cols, dense, x);
    for (std::int32_t i = 0; i < rows; ++i) {
        SIHPS_ASSERT_NEAR(y_csr[i], y_ref[i], 1e-12);
        SIHPS_ASSERT_NEAR(y_csc[i], y_ref[i], 1e-12);
    }
}
