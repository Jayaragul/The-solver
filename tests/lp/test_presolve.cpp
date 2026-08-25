#include "../test_framework.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "lp/Presolve.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <string>

using sihps::CSRMatrix;
using sihps::kInfinity;
using sihps::LpProblem;
using sihps::LpSolverOptions;
using sihps::LpStatus;
using sihps::presolve;
using sihps::PresolveStatus;
using sihps::read_mps_file;
using sihps::solve_lp;
using sihps::Triplet;

namespace {

std::string netlib(const std::string& name) {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/" + name + ".mps";
}

// minimize -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x,y >= 0.
// Hand-derived optimum (see tests/lp/test_simplex.cpp): (1.6, 1.2), -2.8.
LpProblem tiny_lp() {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {4.0, 6.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);
    return p;
}

} // namespace

// A row whose single coefficient bounds one variable must become a bound,
// not survive as a row: x <= 3 written as a row, with y otherwise free to
// two-variable row.
SIHPS_TEST(presolve_converts_singleton_row_to_bound) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {1.0, 1.0};
    p.rhs = {3.0, 10.0};
    p.row_types = {'L', 'E'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.removed_rows() >= 1);
    // x's upper bound must have absorbed the singleton row. The comparison
    // allows Presolve.cpp's deliberate outward safety margin
    // (kBoundRelax * (1 + |bound|), so ~4e-9 at a bound of 3): presolve is
    // required to produce a RELAXATION of the true bound, never an
    // over-tightening, so a bound landing marginally above 3 is the
    // specified behaviour rather than an error.
    constexpr double kMargin = 1e-8;
    bool found = false;
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 0) {
            SIHPS_ASSERT_TRUE(r.reduced.upper[k] <= 3.0 + kMargin);
            found = true;
        }
    }
    SIHPS_ASSERT_TRUE(found);
}

// A row that its variables' bounds already imply carries no information and
// must be dropped: x in [0,1], y in [0,1], row x + y <= 5.
SIHPS_TEST(presolve_drops_redundant_row) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {5.0};
    p.row_types = {'L'};
    p.lower = {0.0, 0.0};
    p.upper = {1.0, 1.0};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_EQ(r.removed_rows(), 1);
}

// A fixed column must be substituted out and its value recoverable.
SIHPS_TEST(presolve_removes_fixed_column_and_postsolve_restores_it) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, 1.0};
    p.rhs = {10.0};
    p.row_types = {'E'};
    p.lower = {4.0, 0.0};
    p.upper = {4.0, kInfinity}; // x fixed at 4
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 4.0, 1e-12);

    // Whatever the reduced problem reports for the surviving column,
    // postsolve must put x back at 4. Presolve here cascades: fixing x
    // leaves the row as the singleton y = 6, so y is fixed too and NO
    // column survives -- postsolve must still reconstruct both. The 1e-8
    // tolerance covers the outward safety margin documented above.
    std::vector<double> reduced_x(r.kept_columns.size(), 6.0);
    auto full = sihps::postsolve(r, reduced_x);
    SIHPS_ASSERT_EQ(static_cast<int>(full.size()), 2);
    SIHPS_ASSERT_NEAR(full[0], 4.0, 1e-8);
    SIHPS_ASSERT_NEAR(full[1], 6.0, 1e-8);
}

// Contradictory bounds must be caught by presolve rather than passed on.
SIHPS_TEST(presolve_detects_infeasible_bounds) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 1, t);
    p.obj = {0.0};
    p.rhs = {3.0, 5.0};
    p.row_types = {'L', 'G'}; // x <= 3 and x >= 5
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::INFEASIBLE);
}

// An empty column with a cost that improves without limit is unbounded.
SIHPS_TEST(presolve_detects_unbounded_empty_column) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, -1.0}; // column 1 is in no row and improves forever
    p.rhs = {1.0};
    p.row_types = {'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::UNBOUNDED);
}

SIHPS_TEST(solve_lp_matches_hand_verified_optimum_with_presolve) {
    LpProblem p = tiny_lp();
    LpSolverOptions options;
    options.use_presolve = true;
    auto s = solve_lp(p, options);
    SIHPS_ASSERT_TRUE(s.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(s.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(s.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(s.x[1], 1.2, 1e-6);
    SIHPS_ASSERT_TRUE(s.primal_residual < 1e-6);
}

// The sharpest available check that the reductions are sound: presolve must
// not change the answer. Any reduction that is subtly wrong shows up here as
// a differing optimum on a real model, which no amount of synthetic unit
// testing would catch.
SIHPS_TEST(presolve_does_not_change_the_optimum_on_netlib_instances) {
    const char* names[] = {"afiro", "adlittle", "share2b", "blend", "sc205", "scagr7", "bore3d"};
    for (const char* name : names) {
        auto model = read_mps_file(netlib(name));
        LpProblem p = sihps::lp_problem_from_mps(model);

        LpSolverOptions with;
        with.use_presolve = true;
        LpSolverOptions without;
        without.use_presolve = false;

        auto a = solve_lp(p, with);
        auto b = solve_lp(p, without);

        SIHPS_ASSERT_TRUE(a.status == LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(b.status == LpStatus::OPTIMAL);
        const double scale = 1.0 + std::fabs(b.objective_value);
        SIHPS_ASSERT_TRUE(std::fabs(a.objective_value - b.objective_value) / scale < 1e-7);
        SIHPS_ASSERT_TRUE(a.primal_residual < 1e-6);
    }
}

// Postsolve must always return a vector in ORIGINAL column space, whatever
// presolve removed.
SIHPS_TEST(solve_lp_returns_original_space_solution_dimensions) {
    auto model = read_mps_file(netlib("afiro"));
    LpProblem p = sihps::lp_problem_from_mps(model);
    LpSolverOptions options;
    options.use_presolve = true;
    auto s = solve_lp(p, options);
    SIHPS_ASSERT_TRUE(s.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_EQ(static_cast<int>(s.x.size()), p.n_cols());
}
