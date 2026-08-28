// Tests for the custom pricing kernels (src/cuda/PricingKernels.cu) and
// the device-resident pricer built on them (src/cuda/GpuPricer.hpp).
//
// These sit at prompt.md \S3.7's Level 4 -- CPU/GPU equivalence -- but with
// a stricter bar than the SpMV tests use. An SpMV is checked against a
// tolerance because two summation orders legitimately differ in the last
// bits. Pricing produces a DECISION (which column enters), and a decision
// is either the same or it is not; where a quantity here can be compared
// exactly, it is.
//
// Three properties are under test, and the third is the one that caught a
// real bug:
//
//   1. The GPU's reduced-cost vector matches the CPU's.
//   2. The GPU's entering-variable choice matches the CPU's, including the
//      tie-break, and both backends reach the same optimum.
//   3. Repeated GPU solves of the same model are IDENTICAL -- same
//      iteration count, same objective bit for bit. This is not a
//      formality: the first version of the asynchronous Devex path
//      overwrote a pinned staging buffer while its DMA was still in
//      flight, and the only symptom was that iteration counts wandered
//      between runs and one instance intermittently failed verification.
//      A tolerance-based check would have passed throughout.

#include "test_framework.hpp"

#include "cuda/GpuPricer.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "sparse/Convert.hpp"
#include "sparse/CSRMatrix.hpp"

#include <cmath>
#include <string>
#include <vector>

using sihps::CSCMatrix;
using sihps::CSRMatrix;
using sihps::csr_to_csc;
using sihps::GpuPricer;
using sihps::LpAlgorithm;
using sihps::LpProblem;
using sihps::LpSolution;
using sihps::LpSolverOptions;
using sihps::LpStatus;
using sihps::PricingBackend;
using sihps::PricingRule;
using sihps::read_mps_file;
using sihps::Triplet;

namespace {

std::string netlib(const char* stem) {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/" + stem + ".mps";
}

// The reduced costs the CPU path computes, in the same augmented layout
// GpuPricer uses: [structural | slack | artificial].
std::vector<double> reference_reduced_costs(const CSCMatrix& a, const std::vector<double>& y,
                                             const std::vector<double>& cost,
                                             const std::vector<double>& art_sign,
                                             std::int32_t n_rows, std::int32_t n_struct) {
    const std::int32_t n_total = n_struct + 2 * n_rows;
    std::vector<double> rc(static_cast<std::size_t>(n_total), 0.0);
    for (std::int32_t j = 0; j < n_struct; ++j) {
        double dot = 0.0;
        for (std::int32_t k = a.col_ptr()[j]; k < a.col_ptr()[j + 1]; ++k) {
            dot += a.values()[k] * y[static_cast<std::size_t>(a.row_idx()[k])];
        }
        rc[static_cast<std::size_t>(j)] = cost[static_cast<std::size_t>(j)] - dot;
    }
    for (std::int32_t i = 0; i < n_rows; ++i) {
        rc[static_cast<std::size_t>(n_struct + i)] =
            cost[static_cast<std::size_t>(n_struct + i)] - y[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = 0; i < n_rows; ++i) {
        const auto ai = static_cast<std::size_t>(n_struct + n_rows + i);
        rc[ai] = cost[ai] - art_sign[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
    }
    return rc;
}

} // namespace

SIHPS_TEST(gpu_pricer_reduced_costs_match_cpu_on_a_hand_built_matrix) {
    // 3 x 4 with a deliberately irregular sparsity pattern, including an
    // empty column -- an empty column's reduced cost is just its cost, and
    // getting the CSR row-pointer arithmetic wrong for A^T is the obvious
    // way to break exactly that case.
    const std::vector<Triplet> t = {{0, 0, 2.0},  {0, 2, -1.5}, {1, 0, 3.0},
                                    {1, 3, 4.25}, {2, 2, 0.5},  {2, 3, -2.0}};
    const CSRMatrix a_csr = CSRMatrix::from_triplets(3, 4, t);
    const CSCMatrix a_csc = csr_to_csc(a_csr);

    const std::int32_t n_rows = 3, n_struct = 4, n_total = n_struct + 2 * n_rows;
    std::vector<double> cost(static_cast<std::size_t>(n_total));
    for (std::int32_t j = 0; j < n_total; ++j) {
        cost[static_cast<std::size_t>(j)] = 0.25 * (j + 1);
    }
    const std::vector<double> lower(static_cast<std::size_t>(n_total), 0.0);
    const std::vector<double> upper(static_cast<std::size_t>(n_total), 10.0);
    const std::vector<double> art_sign = {1.0, -1.0, 1.0};
    const std::vector<double> y = {0.75, -1.25, 2.5};

    GpuPricer pricer(a_csc, n_rows, n_struct, n_rows, n_rows);
    pricer.sync_phase(cost.data(), lower.data(), upper.data(), art_sign.data());

    std::vector<double> gpu_rc(static_cast<std::size_t>(n_total), 0.0);
    pricer.price_to_host(y.data(), gpu_rc.data());

    const std::vector<double> cpu_rc =
        reference_reduced_costs(a_csc, y, cost, art_sign, n_rows, n_struct);

    // Every entry here is a sum of at most two products, so both backends
    // perform identical arithmetic and the comparison can be exact.
    for (std::int32_t j = 0; j < n_total; ++j) {
        SIHPS_ASSERT_NEAR(gpu_rc[static_cast<std::size_t>(j)], cpu_rc[static_cast<std::size_t>(j)],
                           0.0);
    }
}

SIHPS_TEST(gpu_pricer_selects_the_same_entering_column_as_the_cpu_rule) {
    // Only column 1 is eligible: column 0 sits at its lower bound with a
    // POSITIVE reduced cost (increasing it would worsen the objective),
    // column 2 is fixed (lower == upper), and column 3 is basic.
    const std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}, {1, 2, 1.0}, {1, 3, 1.0}};
    const CSCMatrix a_csc = csr_to_csc(CSRMatrix::from_triplets(2, 4, t));

    const std::int32_t n_rows = 2, n_struct = 4, n_total = n_struct + 2 * n_rows;
    std::vector<double> cost(static_cast<std::size_t>(n_total), 0.0);
    cost[0] = 5.0;   // at lower, positive rc -> not eligible
    cost[1] = -7.0;  // at lower, negative rc -> eligible, and the winner
    cost[2] = -99.0; // fixed column -> must be ignored despite the best rc
    cost[3] = -50.0; // basic -> must be ignored

    std::vector<double> lower(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(n_total), 10.0);
    upper[2] = 0.0; // fixed
    const std::vector<double> art_sign(static_cast<std::size_t>(n_rows), 1.0);
    const std::vector<double> y(static_cast<std::size_t>(n_rows), 0.0);

    std::vector<std::uint8_t> status(static_cast<std::size_t>(n_total), 0 /* AT_LOWER */);
    status[3] = 3; // BASIC

    GpuPricer pricer(a_csc, n_rows, n_struct, n_rows, n_rows);
    pricer.sync_phase(cost.data(), lower.data(), upper.data(), art_sign.data());

    const auto cand = pricer.price_and_select(y.data(), status.data(), /*devex=*/false, 1e-7);
    SIHPS_ASSERT_EQ(cand.index, 1);
    SIHPS_ASSERT_EQ(cand.direction, 1);
    SIHPS_ASSERT_NEAR(cand.reduced_cost, -7.0, 1e-12);
}

SIHPS_TEST(gpu_pricer_reports_no_candidate_when_every_column_is_priced_out) {
    const std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 1.0}};
    const CSCMatrix a_csc = csr_to_csc(CSRMatrix::from_triplets(2, 2, t));

    const std::int32_t n_rows = 2, n_struct = 2, n_total = n_struct + 2 * n_rows;
    // All costs positive with every variable resting at its lower bound:
    // nothing can improve, which is the optimality condition the primal
    // loop reads off the sentinel index.
    const std::vector<double> cost(static_cast<std::size_t>(n_total), 3.0);
    const std::vector<double> lower(static_cast<std::size_t>(n_total), 0.0);
    const std::vector<double> upper(static_cast<std::size_t>(n_total), 1.0);
    const std::vector<double> art_sign(static_cast<std::size_t>(n_rows), 1.0);
    const std::vector<double> y(static_cast<std::size_t>(n_rows), 0.0);
    const std::vector<std::uint8_t> status(static_cast<std::size_t>(n_total), 0);

    GpuPricer pricer(a_csc, n_rows, n_struct, n_rows, n_rows);
    pricer.sync_phase(cost.data(), lower.data(), upper.data(), art_sign.data());

    const auto cand = pricer.price_and_select(y.data(), status.data(), /*devex=*/false, 1e-7);
    SIHPS_ASSERT_TRUE(cand.index < 0);
}

SIHPS_TEST(gpu_pricer_breaks_score_ties_toward_the_lowest_index) {
    // Two columns with identical reduced costs. The CPU scans ascending
    // with a strict `>`, so it keeps the first; the device reduction must
    // agree, or the two backends silently diverge onto different pivot
    // sequences on every degenerate model.
    const std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    const CSCMatrix a_csc = csr_to_csc(CSRMatrix::from_triplets(1, 2, t));

    const std::int32_t n_rows = 1, n_struct = 2, n_total = n_struct + 2 * n_rows;
    std::vector<double> cost(static_cast<std::size_t>(n_total), 0.0);
    cost[0] = -4.0;
    cost[1] = -4.0;
    const std::vector<double> lower(static_cast<std::size_t>(n_total), 0.0);
    const std::vector<double> upper(static_cast<std::size_t>(n_total), 10.0);
    const std::vector<double> art_sign(static_cast<std::size_t>(n_rows), 1.0);
    const std::vector<double> y(static_cast<std::size_t>(n_rows), 0.0);
    const std::vector<std::uint8_t> status(static_cast<std::size_t>(n_total), 0);

    GpuPricer pricer(a_csc, n_rows, n_struct, n_rows, n_rows);
    pricer.sync_phase(cost.data(), lower.data(), upper.data(), art_sign.data());
    const auto cand = pricer.price_and_select(y.data(), status.data(), /*devex=*/false, 1e-7);
    SIHPS_ASSERT_EQ(cand.index, 0);
}

SIHPS_TEST(gpu_pricer_devex_weights_start_at_one_and_survive_a_reset) {
    const std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 1.0}};
    const CSCMatrix a_csc = csr_to_csc(CSRMatrix::from_triplets(2, 2, t));
    const std::int32_t n_rows = 2, n_struct = 2, n_total = n_struct + 2 * n_rows;

    GpuPricer pricer(a_csc, n_rows, n_struct, n_rows, n_rows);
    std::vector<double> w(static_cast<std::size_t>(n_total), 0.0);
    pricer.download_devex_weights(w.data());
    for (double v : w) SIHPS_ASSERT_NEAR(v, 1.0, 0.0);

    pricer.reset_devex_weights();
    pricer.download_devex_weights(w.data());
    for (double v : w) SIHPS_ASSERT_NEAR(v, 1.0, 0.0);
}

SIHPS_TEST(gpu_and_cpu_pricing_backends_agree_on_a_midsize_netlib_model) {
    // afiro (27 rows) is already covered elsewhere and is small enough to
    // hide most of what can go wrong. sctap1 has 300 rows and 480 columns,
    // so the fused pricing kernel runs a real grid-stride loop and the
    // Devex weight update actually accumulates over hundreds of pivots.
    const auto model = read_mps_file(netlib("sctap1"));
    const LpProblem p = sihps::lp_problem_from_mps(model);

    LpSolverOptions cpu_opts;
    cpu_opts.backend = PricingBackend::CPU;
    LpSolverOptions gpu_opts;
    gpu_opts.backend = PricingBackend::GPU;

    const LpSolution cpu = sihps::solve_lp(p, cpu_opts);
    const LpSolution gpu = sihps::solve_lp(p, gpu_opts);

    SIHPS_ASSERT_TRUE(cpu.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(gpu.status == LpStatus::OPTIMAL);
    // The optimum is a property of the model; only the route to it may
    // differ, and only then because cuSPARSE sums in a different order.
    SIHPS_ASSERT_NEAR(cpu.objective_value, gpu.objective_value,
                       1e-6 * (1.0 + std::fabs(cpu.objective_value)));
}

SIHPS_TEST(gpu_pricing_is_bit_reproducible_across_repeated_solves) {
    // See the header comment: this is the property whose absence was the
    // ONLY symptom of an in-flight-DMA race in the asynchronous Devex
    // path. Exact equality, not a tolerance -- a tolerance passed while
    // the bug was present.
    const auto model = read_mps_file(netlib("sctap1"));
    const LpProblem p = sihps::lp_problem_from_mps(model);

    LpSolverOptions opts;
    opts.backend = PricingBackend::GPU;
    opts.pricing_rule = PricingRule::DEVEX;

    const LpSolution first = sihps::solve_lp(p, opts);
    SIHPS_ASSERT_TRUE(first.status == LpStatus::OPTIMAL);

    for (int rep = 0; rep < 3; ++rep) {
        const LpSolution again = sihps::solve_lp(p, opts);
        SIHPS_ASSERT_TRUE(again.status == first.status);
        SIHPS_ASSERT_EQ(again.iterations, first.iterations);
        SIHPS_ASSERT_NEAR(again.objective_value, first.objective_value, 0.0);
    }
}

SIHPS_TEST(gpu_pricing_agrees_with_cpu_under_dantzig_too) {
    // The two pricing rules take different branches inside the fused
    // kernel (Devex divides by a device-resident weight, Dantzig does
    // not), so backend equivalence has to be established for both.
    const auto model = read_mps_file(netlib("share2b"));
    const LpProblem p = sihps::lp_problem_from_mps(model);

    LpSolverOptions cpu_opts;
    cpu_opts.backend = PricingBackend::CPU;
    cpu_opts.pricing_rule = PricingRule::DANTZIG;
    LpSolverOptions gpu_opts;
    gpu_opts.backend = PricingBackend::GPU;
    gpu_opts.pricing_rule = PricingRule::DANTZIG;

    const LpSolution cpu = sihps::solve_lp(p, cpu_opts);
    const LpSolution gpu = sihps::solve_lp(p, gpu_opts);

    SIHPS_ASSERT_TRUE(cpu.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(gpu.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(cpu.objective_value, gpu.objective_value,
                       1e-6 * (1.0 + std::fabs(cpu.objective_value)));
}

SIHPS_TEST(gpu_pricing_handles_the_dual_algorithm_path) {
    // The dual simplex uses price_to_host (it needs the whole reduced-cost
    // vector for its ratio test), which is a different GpuPricer entry
    // point from the primal loop's fused select. Both must work, and both
    // must land on the same optimum as the CPU.
    const auto model = read_mps_file(netlib("sctap1"));
    const LpProblem p = sihps::lp_problem_from_mps(model);

    LpSolverOptions cpu_opts;
    cpu_opts.backend = PricingBackend::CPU;
    cpu_opts.algorithm = LpAlgorithm::DUAL;
    LpSolverOptions gpu_opts;
    gpu_opts.backend = PricingBackend::GPU;
    gpu_opts.algorithm = LpAlgorithm::DUAL;

    const LpSolution cpu = sihps::solve_lp(p, cpu_opts);
    const LpSolution gpu = sihps::solve_lp(p, gpu_opts);

    SIHPS_ASSERT_TRUE(cpu.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(gpu.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(cpu.objective_value, gpu.objective_value,
                       1e-6 * (1.0 + std::fabs(cpu.objective_value)));
}
