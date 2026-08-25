// Thread-count invariance.
//
// docs/architecture/NUMERICS.md \S1 requires bit-reproducible results, and
// src/parallel/Parallel.hpp claims the OpenMP loops in this engine preserve
// that: one writer per output element, serial summation order inside each
// element, schedule(static). That is a claim about how the loops are
// written, so it is checked rather than trusted -- and checked with EXACT
// equality, because a tolerance would still pass if the property broke and
// only the last bits moved.
//
// This also covers the build-without-OpenMP configuration: with the
// pragmas compiled out, set_threads() is a no-op and the test degenerates
// to comparing a computation with itself, which is exactly the guarantee
// being asserted (a serial build must produce the same numbers).

#include "test_framework.hpp"

#include "parallel/Parallel.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "sparse/CSRMatrix.hpp"
#include "sparse/Triplet.hpp"

#include "io/MpsReader.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

using sihps::CSRMatrix;
using sihps::LpSolution;
using sihps::LpSolverOptions;
using sihps::LpStatus;
using sihps::ParallelMode;
using sihps::read_mps_file;
using sihps::Triplet;

namespace {

void set_threads(int n) {
#if defined(_OPENMP)
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Restores the process-wide thread count on scope exit. Without this a
// failing assertion would leave every later test running single-threaded,
// turning one failure into a misleading cascade.
struct ThreadCountGuard {
    int saved;
    ThreadCountGuard() : saved(max_threads()) {}
    ~ThreadCountGuard() { set_threads(saved); }
};

CSRMatrix random_csr(std::int32_t rows, std::int32_t cols, std::int32_t per_row, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> col_pick(0, cols - 1);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::vector<Triplet> t;
    t.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(per_row));
    for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t k = 0; k < per_row; ++k) {
            t.push_back(Triplet{i, col_pick(rng), val(rng)});
        }
    }
    return CSRMatrix::from_triplets(rows, cols, t);
}

} // namespace

SIHPS_TEST(csr_multiply_is_bit_identical_across_thread_counts) {
    ThreadCountGuard guard;

    // Comfortably above kParallelNnzThreshold, so the parallel branch is
    // the one actually being compared. A matrix below the threshold would
    // run serially in both passes and the test would prove nothing.
    const CSRMatrix a = random_csr(30000, 30000, 9, 4242u);
    SIHPS_ASSERT_TRUE(a.nnz() >= sihps::kParallelNnzThreshold);

    std::vector<double> x(static_cast<std::size_t>(a.cols()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        // Values spanning several orders of magnitude: reassociation shows
        // up soonest when the addends are badly scaled relative to one
        // another, so a uniform vector would be a weak probe.
        x[i] = ((i % 2) ? 1.0 : -1.0) * (1e-6 + static_cast<double>(i % 1000));
    }

    std::vector<double> y_one(static_cast<std::size_t>(a.rows()), 0.0);
    std::vector<double> y_many(static_cast<std::size_t>(a.rows()), 0.0);

    set_threads(1);
    a.multiply(x.data(), y_one.data());
    set_threads(std::max(2, guard.saved));
    a.multiply(x.data(), y_many.data());

    SIHPS_ASSERT_TRUE(std::equal(y_one.begin(), y_one.end(), y_many.begin()));
}

SIHPS_TEST(lp_solve_is_bit_identical_across_thread_counts) {
    ThreadCountGuard guard;

    // fit2d has 129018 nonzeros -- far above the threshold, so its pricing
    // passes genuinely run on a thread team. Solving it twice at different
    // thread counts must give the same objective AND the same number of
    // iterations: a different iteration count would mean the thread count
    // had perturbed a pivot decision, which is the failure this guards.
    const auto model =
        read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/fit2d.mps");
    const auto p = sihps::lp_problem_from_mps(model);

    LpSolverOptions opts; // CPU backend: this is a CPU-threading property

    set_threads(1);
    const LpSolution one = sihps::solve_lp(p, opts);
    set_threads(std::max(2, guard.saved));
    const LpSolution many = sihps::solve_lp(p, opts);

    SIHPS_ASSERT_TRUE(one.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(many.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_EQ(one.iterations, many.iterations);
    SIHPS_ASSERT_NEAR(one.objective_value, many.objective_value, 0.0);
}

SIHPS_TEST(lp_parallel_policy_is_explicit_and_bit_identical) {
    ThreadCountGuard guard;

    const auto model =
        read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/fit2d.mps");
    const auto p = sihps::lp_problem_from_mps(model);

    LpSolverOptions serial_opts;
    serial_opts.parallel_mode = ParallelMode::SERIAL;
    LpSolverOptions parallel_opts;
    parallel_opts.parallel_mode = ParallelMode::PARALLEL;

    set_threads(1);
    const LpSolution serial = sihps::solve_lp(p, serial_opts);
    set_threads(std::max(2, guard.saved));
    const LpSolution parallel = sihps::solve_lp(p, parallel_opts);

    SIHPS_ASSERT_TRUE(serial.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(parallel.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_EQ(serial.iterations, parallel.iterations);
    SIHPS_ASSERT_NEAR(serial.objective_value, parallel.objective_value, 0.0);
    SIHPS_ASSERT_NEAR(serial.primal_residual, parallel.primal_residual, 0.0);
}
