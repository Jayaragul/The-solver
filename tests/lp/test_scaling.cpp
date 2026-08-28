#include "../test_framework.hpp"
#include "lp/LpProblem.hpp"
#include "lp/Scaling.hpp"
#include "lp/Simplex.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>

using sihps::apply_ruiz_scaling;
using sihps::compute_ruiz_scaling;
using sihps::CSRMatrix;
using sihps::kInfinity;
using sihps::LpProblem;
using sihps::LpStatus;
using sihps::PricingBackend;
using sihps::ScaleFactors;
using sihps::Simplex;
using sihps::Triplet;

namespace {

// Every nonzero row and column of `a` must have infinity norm within
// [1-tol, 1+tol] -- the convergence criterion Scaling.hpp documents.
void assert_equilibrated(const CSRMatrix& a, double tol) {
    for (std::int32_t i = 0; i < a.rows(); ++i) {
        double best = 0.0;
        for (std::int32_t k = a.row_ptr()[i]; k < a.row_ptr()[i + 1]; ++k) {
            best = std::max(best, std::fabs(a.values()[k]));
        }
        if (best > 0.0) SIHPS_ASSERT_TRUE(std::fabs(best - 1.0) < tol);
    }
    std::vector<double> col_best(static_cast<std::size_t>(a.cols()), 0.0);
    for (std::int32_t i = 0; i < a.rows(); ++i) {
        for (std::int32_t k = a.row_ptr()[i]; k < a.row_ptr()[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(a.col_idx()[k]);
            col_best[j] = std::max(col_best[j], std::fabs(a.values()[k]));
        }
    }
    for (double best : col_best) {
        if (best > 0.0) SIHPS_ASSERT_TRUE(std::fabs(best - 1.0) < tol);
    }
}

} // namespace

// A matrix with a row and a column each six orders of magnitude off from
// the rest -- exactly the mixed-physical-units pathology
// docs/research/SOTA.md \S1.4.1 hypothesizes for refinery models.
SIHPS_TEST(ruiz_scaling_equilibrates_badly_scaled_matrix) {
    std::vector<Triplet> t = {
        {0, 0, 1.0e6}, {0, 1, 2.0e6}, {0, 2, 5.0e5},
        {1, 0, 3.0},   {1, 1, 1.0e-3}, {1, 2, 4.0},
        {2, 0, 2.0},   {2, 1, 6.0e-3}, {2, 2, 1.0},
    };
    CSRMatrix a = CSRMatrix::from_triplets(3, 3, t);

    ScaleFactors s = compute_ruiz_scaling(a, 200, 1e-2);
    for (double r : s.row_scale) SIHPS_ASSERT_TRUE(r > 0.0);
    for (double c : s.col_scale) SIHPS_ASSERT_TRUE(c > 0.0);

    CSRMatrix scaled = apply_ruiz_scaling(a, s);
    assert_equilibrated(scaled, 0.05);
}

// An all-zero matrix has no informative row/column norm anywhere --
// compute_ruiz_scaling must return identity rather than divide by zero or
// loop forever.
SIHPS_TEST(ruiz_scaling_handles_zero_matrix) {
    std::vector<Triplet> t = {{0, 0, 0.0}, {1, 1, 0.0}};
    CSRMatrix a = CSRMatrix::from_triplets(2, 2, t);
    ScaleFactors s = compute_ruiz_scaling(a);
    SIHPS_ASSERT_NEAR(s.row_scale[0], 1.0, 1e-12);
    SIHPS_ASSERT_NEAR(s.row_scale[1], 1.0, 1e-12);
    SIHPS_ASSERT_NEAR(s.col_scale[0], 1.0, 1e-12);
    SIHPS_ASSERT_NEAR(s.col_scale[1], 1.0, 1e-12);
}

// An already well-scaled matrix (all entries near unit magnitude) should
// still satisfy the equilibration property after scaling -- scaling must
// be a no-regression operation on inputs that didn't need it.
SIHPS_TEST(ruiz_scaling_converges_on_well_scaled_matrix) {
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 0.5}, {1, 0, 2.0}, {1, 1, 1.0}};
    CSRMatrix a = CSRMatrix::from_triplets(2, 2, t);
    ScaleFactors s = compute_ruiz_scaling(a);
    CSRMatrix scaled = apply_ruiz_scaling(a, s);
    assert_equilibrated(scaled, 0.05);
}

// End-to-end: the SAME feasible region and optimum as
// test_simplex.cpp's tiny_lp (hand-derived optimum (1.6, 1.2), objective
// -2.8), but with the first row scaled up by 1e6 in both its coefficients
// and rhs (Ax <= b is invariant under multiplying a row by a positive
// constant) -- this must still solve to the identical answer, proving the
// scale/unscale round-trip introduced by Ruiz equilibration is correct,
// not just that it doesn't crash.
SIHPS_TEST(simplex_solves_ill_scaled_lp_correctly_with_ruiz_scaling) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0e6}, {0, 1, 2.0e6}, {1, 0, 3.0}, {1, 1, 1.0},
    };
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {4.0e6, 6.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    Simplex simplex(p, PricingBackend::CPU, /*use_ruiz_scaling=*/true);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 1.2, 1e-6);
    SIHPS_ASSERT_TRUE(result.primal_residual < 1e-6);
    SIHPS_ASSERT_TRUE(result.dual_residual < 1e-6);
}

// Same problem with scaling explicitly disabled -- both paths must agree,
// confirming use_ruiz_scaling=false is a genuine identity-scaling fallback
// and not a different code path with different semantics.
SIHPS_TEST(simplex_solves_ill_scaled_lp_correctly_without_ruiz_scaling) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0e6}, {0, 1, 2.0e6}, {1, 0, 3.0}, {1, 1, 1.0},
    };
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {4.0e6, 6.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    Simplex simplex(p, PricingBackend::CPU, /*use_ruiz_scaling=*/false);
    auto result = simplex.solve();
    SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(result.x[1], 1.2, 1e-6);
}
