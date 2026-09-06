#include "../test_framework.hpp"

#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <random>
#include <vector>

using sihps::CSRMatrix;
using sihps::LpProblem;
using sihps::LpStatus;
using sihps::LpSolverOptions;
using sihps::Triplet;

SIHPS_TEST(generated_scaled_sparse_lps_remain_feasible_and_bounded) {
    std::mt19937 generator(0x51A5u);
    std::uniform_real_distribution<double> point_distribution(0.1, 9.9);
    std::uniform_real_distribution<double> coefficient_distribution(-1.0, 1.0);
    std::uniform_int_distribution<int> exponent_distribution(-6, 6);

    for (int trial = 0; trial < 12; ++trial) {
        constexpr std::int32_t rows = 8;
        constexpr std::int32_t cols = 12;
        std::vector<double> known_point(static_cast<std::size_t>(cols));
        for (double& value : known_point) value = point_distribution(generator);

        std::vector<Triplet> entries;
        entries.reserve(static_cast<std::size_t>(rows * cols / 2));
        std::vector<double> rhs(static_cast<std::size_t>(rows), 0.0);
        for (std::int32_t row = 0; row < rows; ++row) {
            double activity = 0.0;
            for (std::int32_t col = 0; col < cols; ++col) {
                if (((row * 17 + col * 13 + trial) % 5) == 0) continue;
                const double coefficient = coefficient_distribution(generator) *
                                            std::pow(10.0, exponent_distribution(generator));
                entries.push_back({row, col, coefficient});
                activity += coefficient * known_point[static_cast<std::size_t>(col)];
            }
            // The known point is feasible for every generated upper row.
            rhs[static_cast<std::size_t>(row)] = activity +
                                                 0.25 * (1.0 + std::fabs(activity));
        }

        LpProblem problem;
        problem.A = CSRMatrix::from_triplets(rows, cols, entries);
        problem.obj.resize(static_cast<std::size_t>(cols));
        for (double& coefficient : problem.obj) {
            coefficient = coefficient_distribution(generator);
        }
        problem.rhs = std::move(rhs);
        problem.row_types.assign(static_cast<std::size_t>(rows), 'L');
        problem.lower.assign(static_cast<std::size_t>(cols), 0.0);
        problem.upper.assign(static_cast<std::size_t>(cols), 10.0);
        sihps::apply_default_row_bounds(problem);

        LpSolverOptions options;
        options.use_presolve = true;
        options.use_ruiz_scaling = true;
        const auto result = sihps::solve_lp(problem, options);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(result.x.size() == static_cast<std::size_t>(cols));
        SIHPS_ASSERT_TRUE(std::isfinite(result.objective_value));
        SIHPS_ASSERT_TRUE(result.primal_residual <= 1e-6);
    }
}
