#include "../test_framework.hpp"
#include "lp/BasisFactorization.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using sihps::BasisFactorization;
using sihps::SparseColumn;

namespace {

// Dense reference: B[row][col].
using Dense = std::vector<std::vector<double>>;

std::vector<SparseColumn> to_columns(const Dense& b) {
    const auto m = static_cast<std::int32_t>(b.size());
    std::vector<SparseColumn> columns(static_cast<std::size_t>(m));
    for (std::int32_t col = 0; col < m; ++col) {
        for (std::int32_t row = 0; row < m; ++row) {
            const double v = b[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (v != 0.0) columns[static_cast<std::size_t>(col)].emplace_back(row, v);
        }
    }
    return columns;
}

// Verifies B x = rhs directly against the dense matrix, which is the only
// check that actually catches a permutation error -- comparing against
// another factorization would just reproduce the same bug.
void check_ftran(const Dense& b, const std::vector<double>& rhs, double tol) {
    const auto m = static_cast<std::int32_t>(b.size());
    BasisFactorization factor;
    auto result = factor.factorize(m, to_columns(b));
    SIHPS_ASSERT_TRUE(result.ok);
    SIHPS_ASSERT_TRUE(result.singular.empty());

    std::vector<double> x = rhs;
    factor.ftran(x);

    for (std::int32_t row = 0; row < m; ++row) {
        double accumulated = 0.0;
        for (std::int32_t col = 0; col < m; ++col) {
            accumulated += b[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                            x[static_cast<std::size_t>(col)];
        }
        SIHPS_ASSERT_NEAR(accumulated, rhs[static_cast<std::size_t>(row)], tol);
    }
}

// Verifies B^T y = c directly against the dense matrix.
void check_btran(const Dense& b, const std::vector<double>& c, double tol) {
    const auto m = static_cast<std::int32_t>(b.size());
    BasisFactorization factor;
    auto result = factor.factorize(m, to_columns(b));
    SIHPS_ASSERT_TRUE(result.ok);

    std::vector<double> y = c;
    factor.btran(y);

    for (std::int32_t col = 0; col < m; ++col) {
        double accumulated = 0.0;
        for (std::int32_t row = 0; row < m; ++row) {
            accumulated += b[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                            y[static_cast<std::size_t>(row)];
        }
        SIHPS_ASSERT_NEAR(accumulated, c[static_cast<std::size_t>(col)], tol);
    }
}

Dense random_nonsingular(std::int32_t m, double density, std::mt19937& rng) {
    std::uniform_real_distribution<double> value(-3.0, 3.0);
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    Dense b(static_cast<std::size_t>(m), std::vector<double>(static_cast<std::size_t>(m), 0.0));
    for (std::int32_t row = 0; row < m; ++row) {
        for (std::int32_t col = 0; col < m; ++col) {
            if (chance(rng) < density) {
                b[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = value(rng);
            }
        }
        // Strong diagonal guarantees nonsingularity without needing a
        // determinant check, so the test exercises the factorization rather
        // than a random-matrix rejection loop.
        b[static_cast<std::size_t>(row)][static_cast<std::size_t>(row)] = 10.0 + value(rng);
    }
    return b;
}

} // namespace

SIHPS_TEST(basis_factorization_solves_identity) {
    Dense b = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    check_ftran(b, {3.0, -1.0, 7.0}, 1e-12);
    check_btran(b, {3.0, -1.0, 7.0}, 1e-12);
}

SIHPS_TEST(basis_factorization_solves_permutation_matrix) {
    // A pure permutation is the sharpest possible test of the row/column
    // permutation bookkeeping: any mix-up between P and Q survives on
    // symmetric inputs but fails here.
    Dense b = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    check_ftran(b, {5.0, 6.0, 7.0}, 1e-12);
    check_btran(b, {5.0, 6.0, 7.0}, 1e-12);
}

SIHPS_TEST(basis_factorization_solves_lower_triangular) {
    Dense b = {{2, 0, 0}, {3, -1, 0}, {1, 4, 5}};
    check_ftran(b, {2.0, 1.0, -3.0}, 1e-12);
    check_btran(b, {2.0, 1.0, -3.0}, 1e-12);
}

SIHPS_TEST(basis_factorization_solves_upper_triangular) {
    Dense b = {{2, 3, 1}, {0, -1, 4}, {0, 0, 5}};
    check_ftran(b, {2.0, 1.0, -3.0}, 1e-12);
    check_btran(b, {2.0, 1.0, -3.0}, 1e-12);
}

SIHPS_TEST(basis_factorization_solves_dense_general_matrix) {
    Dense b = {{4, -2, 1, 3}, {-2, 5, 2, -1}, {1, 2, 6, 2}, {3, -1, 2, 7}};
    check_ftran(b, {1.0, -2.0, 3.0, 0.5}, 1e-10);
    check_btran(b, {1.0, -2.0, 3.0, 0.5}, 1e-10);
}

// Requires pivoting: the natural first pivot is zero, so a factorization
// that ignored ordering would divide by it.
SIHPS_TEST(basis_factorization_requires_pivoting) {
    Dense b = {{0, 2, 1}, {1, 0, 3}, {4, 5, 0}};
    check_ftran(b, {1.0, 1.0, 1.0}, 1e-10);
    check_btran(b, {1.0, 1.0, 1.0}, 1e-10);
}

SIHPS_TEST(basis_factorization_matches_dense_on_random_sparse_matrices) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 25; ++trial) {
        const std::int32_t m = 20 + (trial % 30);
        Dense b = random_nonsingular(m, 0.12, rng);
        std::vector<double> rhs(static_cast<std::size_t>(m));
        std::uniform_real_distribution<double> value(-5.0, 5.0);
        for (auto& v : rhs) v = value(rng);
        check_ftran(b, rhs, 1e-8);
        check_btran(b, rhs, 1e-8);
    }
}

// A singular basis must be REPORTED for repair, never silently accepted --
// column 2 here is an exact duplicate of column 0.
SIHPS_TEST(basis_factorization_reports_singular_columns) {
    Dense b = {{1, 0, 1}, {0, 1, 0}, {0, 0, 0}};
    BasisFactorization factor;
    auto result = factor.factorize(3, to_columns(b));
    SIHPS_ASSERT_TRUE(result.ok);
    SIHPS_ASSERT_TRUE(!result.singular.empty());
}

// The product-form update must reproduce a solve against the UPDATED basis
// exactly as if that basis had been factorized from scratch. This is the
// property the whole simplex loop rests on.
SIHPS_TEST(basis_factorization_product_form_update_matches_refactorization) {
    std::mt19937 rng(777);
    Dense b = random_nonsingular(24, 0.15, rng);

    BasisFactorization factor;
    SIHPS_ASSERT_TRUE(factor.factorize(24, to_columns(b)).ok);

    // Replace basis position 5 with a new column, via one PFI update.
    const std::int32_t leaving = 5;
    std::vector<double> entering_dense(24, 0.0);
    std::uniform_real_distribution<double> value(-4.0, 4.0);
    for (std::int32_t i = 0; i < 24; ++i) entering_dense[static_cast<std::size_t>(i)] = value(rng);
    entering_dense[static_cast<std::size_t>(leaving)] += 12.0; // keep the pivot healthy

    std::vector<double> direction = entering_dense;
    factor.ftran(direction);
    SIHPS_ASSERT_TRUE(factor.update(leaving, direction));

    // Same basis, built explicitly and factorized fresh.
    Dense updated = b;
    for (std::int32_t row = 0; row < 24; ++row) {
        updated[static_cast<std::size_t>(row)][static_cast<std::size_t>(leaving)] =
            entering_dense[static_cast<std::size_t>(row)];
    }

    std::vector<double> rhs(24);
    for (auto& v : rhs) v = value(rng);

    std::vector<double> x_updated = rhs;
    factor.ftran(x_updated);

    for (std::int32_t row = 0; row < 24; ++row) {
        double accumulated = 0.0;
        for (std::int32_t col = 0; col < 24; ++col) {
            accumulated += updated[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                            x_updated[static_cast<std::size_t>(col)];
        }
        SIHPS_ASSERT_NEAR(accumulated, rhs[static_cast<std::size_t>(row)], 1e-7);
    }

    std::vector<double> y_updated = rhs;
    factor.btran(y_updated);
    for (std::int32_t col = 0; col < 24; ++col) {
        double accumulated = 0.0;
        for (std::int32_t row = 0; row < 24; ++row) {
            accumulated += updated[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                            y_updated[static_cast<std::size_t>(row)];
        }
        SIHPS_ASSERT_NEAR(accumulated, rhs[static_cast<std::size_t>(col)], 1e-7);
    }
}

// Many successive updates: the eta file must stay correct as it grows, not
// merely for the first one.
SIHPS_TEST(basis_factorization_survives_many_successive_updates) {
    std::mt19937 rng(4242);
    const std::int32_t m = 18;
    Dense b = random_nonsingular(m, 0.2, rng);

    BasisFactorization factor;
    SIHPS_ASSERT_TRUE(factor.factorize(m, to_columns(b)).ok);

    std::uniform_real_distribution<double> value(-3.0, 3.0);
    for (std::int32_t update_index = 0; update_index < 12; ++update_index) {
        const std::int32_t leaving = update_index % m;
        std::vector<double> entering(static_cast<std::size_t>(m));
        for (auto& v : entering) v = value(rng);
        entering[static_cast<std::size_t>(leaving)] += 15.0;

        std::vector<double> direction = entering;
        factor.ftran(direction);
        SIHPS_ASSERT_TRUE(factor.update(leaving, direction));

        for (std::int32_t row = 0; row < m; ++row) {
            b[static_cast<std::size_t>(row)][static_cast<std::size_t>(leaving)] =
                entering[static_cast<std::size_t>(row)];
        }
    }
    SIHPS_ASSERT_EQ(factor.eta_count(), 12);

    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (auto& v : rhs) v = value(rng);
    std::vector<double> x = rhs;
    factor.ftran(x);
    for (std::int32_t row = 0; row < m; ++row) {
        double accumulated = 0.0;
        for (std::int32_t col = 0; col < m; ++col) {
            accumulated += b[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                            x[static_cast<std::size_t>(col)];
        }
        SIHPS_ASSERT_NEAR(accumulated, rhs[static_cast<std::size_t>(row)], 1e-6);
    }
}

// A zero pivot must be refused rather than producing infinities.
SIHPS_TEST(basis_factorization_update_rejects_tiny_pivot) {
    Dense b = {{1, 0}, {0, 1}};
    BasisFactorization factor;
    SIHPS_ASSERT_TRUE(factor.factorize(2, to_columns(b)).ok);
    std::vector<double> direction = {0.0, 1.0};
    SIHPS_ASSERT_TRUE(!factor.update(0, direction));
}
