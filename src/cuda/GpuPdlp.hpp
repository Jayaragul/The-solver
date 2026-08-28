#pragma once

#include "CudaBuffer.hpp"
#include "CudaEvent.hpp"
#include "CudaStream.hpp"
#include "CusparseHandle.hpp"
#include "GpuSpMV.hpp"
#include "PdlpKernels.cuh"
#include "PinnedBuffer.hpp"
#include "../sparse/CSRMatrix.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace sihps {

struct PdlpParams {
    // Relative KKT tolerance. All three of primal residual, dual residual
    // and duality gap must be within it. 1e-8 is a demanding target for a
    // first-order method and is deliberately the default: the point of this
    // solver is to be fast AND verifiable, and a loose default would make
    // the speed meaningless.
    double eps_optimal = 1e-8;

    std::int32_t max_iterations = 500000;

    // Iterations between convergence checks. This is the parameter that
    // makes the whole approach work: the inner loop issues NO host
    // synchronization, so the per-sync cost that makes GPU simplex
    // uncompetitive (docs/architecture/CPU_GPU.md 4) is divided by this
    // number. Too small and the syncs dominate; too large and the restart
    // scheme reacts slowly. Measured, not guessed --
    // benchmarks/bench_pdlp.cpp sweeps it.
    std::int32_t restart_period = 64;

    double time_limit_seconds = 0.0; // 0 disables

    // Power-iteration count for the ||A||_2 estimate that sets the step
    // size. Underestimating it breaks the convergence condition
    // tau*sigma*||A||^2 < 1, so the estimate is deliberately inflated by
    // kStepSafety in the implementation rather than trusted exactly.
    std::int32_t power_iterations = 30;

    // Applegate et al. 2021 3.1 adaptive step size, evaluated entirely on
    // device (PdlpKernels.cuh). Costs two extra launches per iteration and
    // buys a large reduction in iteration count. Set false to fall back to
    // the fixed eta = 0.9/||A||_2, which is what the A/B comparison in
    // docs/architecture/PDLP.md 6 measures against.
    bool adaptive_step = true;

    // Guard rails on the adaptive step, as multiples of the global bound
    // 1/||A||_2. The lower one stops a pathological instance from grinding
    // the step to zero; the upper one bounds how far the local estimate is
    // allowed to run ahead of the global bound.
    double eta_min_factor = 1e-6;
    double eta_max_factor = 1e4;

    // Cooperative cancellation, polled once per restart window -- i.e. at
    // the one point in the solve where the host is already synchronizing,
    // so it costs nothing and cannot perturb the sync-free inner loop.
    //
    // LpMethod::HYBRID uses this to stop the loser of the simplex/PDLP
    // race. Null means never cancel.
    const std::atomic<bool>* cancel = nullptr;
};

struct PdlpStats {
    bool converged = false;
    bool cancelled = false;
    std::int32_t iterations = 0;
    std::int32_t restarts = 0;
    double primal_objective = 0.0;
    double dual_objective = 0.0;
    double relative_primal_residual = 1.0;
    double relative_dual_residual = 1.0;
    double relative_gap = 1.0;
    double matrix_norm = 0.0;
    double seconds = 0.0;

    // Adaptive-step diagnostics. rejected_steps counts iterations whose
    // candidate violated the local bound and was discarded; a high ratio
    // means the step size is oscillating and is worth looking at.
    std::int32_t rejected_steps = 0;
    double final_step_size = 0.0;
    double step_size_ratio = 1.0; // final eta / (1/||A||_2)
    // Host synchronizations performed during the whole solve. Reported
    // because it is the quantity this design exists to minimize, and a
    // regression in it would silently undo the entire advantage.
    std::int32_t host_syncs = 0;
};

// GPU PDLP: restarted average primal-dual hybrid gradient for
//
//     min c'x   subject to   rl <= A x <= ru,   l <= x <= u
//
// WHY THIS IS THE GPU ALGORITHM AND SIMPLEX IS NOT
// ------------------------------------------------
// A simplex iteration cannot continue until the host knows which column
// entered, so it pays one device synchronize per iteration. Measured on
// this machine, an SpMV costs 33-52 us with a sync behind it and 15-26 us
// without (benchmarks/bench_spmv_algorithm.cpp), and a whole GPU pricing
// iteration costs ~250 us against the CPU's ~90 us. That gap is imposed by
// the algorithm's sequential decision chain, not by the kernels, and no
// amount of kernel work closes it -- which is what
// docs/architecture/CPU_GPU.md 4 concluded and why PricingBackend::CPU
// remains the simplex default.
//
// PDHG has no such decision. Its iteration is two SpMVs and two elementwise
// updates with no data-dependent branch the host must resolve, so
// `restart_period` iterations are queued back to back and synchronized
// ONCE. The per-sync cost is divided by the restart period instead of being
// paid every iteration, and the SpMVs run in their cheaper queued regime.
// That is the entire argument, and PdlpStats::host_syncs is reported so it
// stays checkable rather than becoming folklore.
//
// WHAT IT DOES NOT DO
// -------------------
// First-order methods converge fast to moderate accuracy and slowly to
// high accuracy, and they return an interior-ish point rather than a
// vertex. This class therefore reports a VERIFIED KKT error and never
// asserts optimality on its own authority; `LpSolver` decides whether to
// accept the point or hand it to the simplex for exact vertex
// certification. Claiming a first-order point is optimal because the
// iteration stopped would violate NUMERICS.md 6's rule that a result is
// only OPTIMAL after residuals are checked.
class GpuPdlp {
public:
    // `a` is the (already scaled) constraint matrix; every vector is in the
    // same scaled space. Sizes: cost/lower/upper have a.cols() entries,
    // row_lower/row_upper have a.rows().
    GpuPdlp(const CSRMatrix& a, const std::vector<double>& cost, const std::vector<double>& lower,
            const std::vector<double>& upper, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper);

    PdlpStats solve(const PdlpParams& params, std::vector<double>& x_out,
                    std::vector<double>& y_out);

private:
    // Estimates ||A||_2 by power iteration on A^T A, entirely on device --
    // one synchronize at the end, not one per iteration.
    double estimate_matrix_norm(std::int32_t iterations);

    // Queues the two SpMVs and the residual kernel for the iterate held in
    // (d_x, d_y), leaving the result in d_residuals_. Does not synchronize.
    void queue_residual_evaluation(double* d_x, double* d_y, double* d_ax_scratch);

    // Reads the last queued residual evaluation back and turns it into
    // relative measures. Synchronizes once.
    struct Kkt {
        double primal = 1.0, dual = 1.0, gap = 1.0;
        double primal_obj = 0.0, dual_obj = 0.0;
        double primal_move = 0.0, dual_move = 0.0;
        double worst() const {
            const double a = primal > dual ? primal : dual;
            return a > gap ? a : gap;
        }
    };
    Kkt read_residuals();

    std::int32_t n_, m_;
    std::int32_t grid_n_, grid_m_, grid_max_;

    CusparseHandle handle_;
    CudaStream stream_;
    CudaEvent done_;

    // A and A^T both device-resident. Storing the transpose explicitly
    // costs a second copy of the nonzeros and buys two things that matter
    // more: cusparseSpMV's transpose mode is markedly slower on CSR, and
    // binding each operator to fixed vectors lets the whole iteration run
    // without rebinding anything.
    std::unique_ptr<GpuCsrSpMV> a_;  // x-space -> row-space
    std::unique_ptr<GpuCsrSpMV> at_; // row-space -> x-space

    CudaBuffer<double> d_x_, d_x_bar_, d_x_sum_, d_x_restart_, d_x_avg_, d_aty_;
    CudaBuffer<double> d_y_, d_y_sum_, d_y_restart_, d_y_avg_, d_ax_bar_, d_ax_;
    // Adaptive path: candidate iterate, its matrix product, and the previous
    // A x needed to form A xbar without a third SpMV.
    CudaBuffer<double> d_x_cand_, d_y_cand_, d_ax_cand_, d_ax_prev_;
    CudaBuffer<gpu::PdlpStepState> d_step_;
    PinnedBuffer<gpu::PdlpStepState> h_step_;
    CudaBuffer<double> d_cost_, d_lower_, d_upper_, d_row_lower_, d_row_upper_;
    CudaBuffer<double> d_pow_v_, d_pow_w_, d_pow_v2_;
    CudaBuffer<double> d_block_scratch_, d_scalar_;
    CudaBuffer<unsigned int> d_retire_;
    CudaBuffer<gpu::PdlpResiduals> d_residuals_;

    PinnedBuffer<gpu::PdlpResiduals> h_residuals_;
    PinnedBuffer<double> h_scalar_;
    PinnedBuffer<double> h_vec_n_, h_vec_m_;

    double cost_norm_ = 0.0, bound_norm_ = 0.0;
    std::int32_t sync_count_ = 0;
};

} // namespace sihps
