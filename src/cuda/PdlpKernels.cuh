#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace sihps {
namespace gpu {

// ---------------------------------------------------------------------
// Kernels for PDLP -- restarted average primal-dual hybrid gradient.
// ---------------------------------------------------------------------
//
// WHY THIS EXISTS AT ALL
// ----------------------
// docs/architecture/CPU_GPU.md \S4 measured GPU pricing losing to CPU
// pricing by 3-5x and traced the cause precisely: a simplex iteration
// cannot proceed until the host knows which column enters, so every
// iteration pays a device synchronize. On this machine an SpMV costs
// 33-52 us when a sync follows it and 15-26 us when it does not
// (benchmarks/bench_spmv_algorithm.cpp). The sync is the cost, and no
// kernel tuning removes it, because it is imposed by the ALGORITHM's
// sequential decision chain rather than by the implementation.
//
// A first-order method has no such chain. PDHG's iteration is
//
//     y <- prox( y + sigma * A xbar )
//     x <- proj( x - tau * (c + A^T y) )
//
// -- pure arithmetic on fixed-size vectors, with no branch the host has to
// resolve. So hundreds of iterations can be queued back to back and
// synchronized ONCE, at a convergence check. That is the structural reason
// GPUs win on first-order LP and lose on simplex, and it is why prompt.md
// \S3.2 lists "first-order LP iterations" among the GPU candidates while
// SOTA.md \S1.3b warns off GPU simplex.
//
// Per iteration this file costs exactly two cuSPARSE SpMVs and two custom
// kernels, with no host round trip between them.
//
// THE FORMULATION
// ---------------
// The LP is put in the two-sided form
//
//     min c'x   subject to   rl <= A x <= ru,   l <= x <= u
//
// which LpProblem's (A x + s = rhs, slack bounds) convention maps onto
// directly: rl = rhs - slack_upper, ru = rhs - slack_lower.
//
// As a saddle-point problem,  min_x max_y  c'x + y'A x - sigma_C(y),
// where C = [rl, ru] and sigma_C is its support function. Chambolle-Pock
// applied to that gives the two updates below. The dual prox comes from
// Moreau's identity -- prox_{s.sigma_C}(v) = v - s * proj_C(v/s) -- which
// is what turns an abstract support function into a clamp, and is the
// reason a dual step costs one elementwise pass rather than a solve.
//
// ACCURACY: this is a first-order method, so it converges quickly to
// moderate accuracy and slowly to high accuracy. That is a property of the
// method, not a defect of this implementation, and it is why PdlpSolver
// reports a verified KKT error rather than asserting optimality -- the
// caller decides whether to accept it or hand the point to the simplex for
// exact vertex certification.

// One-shot reduction outputs, sized to be copied to the host in a single
// small transfer at a convergence check (never inside the iteration loop).
struct PdlpResiduals {
    double primal_residual_sq;  // ||Ax - proj_C(Ax)||^2
    double dual_residual_sq;    // ||violation of reduced-cost sign conditions||^2
    double primal_objective;    // c'x
    double dual_objective;      // g(y), the Lagrangian dual value
    double primal_norm_sq;      // ||x||^2, for relative measures
    double dual_norm_sq;        // ||y||^2
    double primal_move_sq;      // ||x - x_restart||^2, drives the primal weight
    double dual_move_sq;        // ||y - y_restart||^2
};

// y <- v - sigma * proj_[rl,ru](v / sigma),  where v = y + sigma * (A xbar)
//
// Also accumulates the running average used by the restart scheme. Folding
// the accumulation in costs one add and saves a launch; at four operations
// per iteration, a fifth would be a 25% increase in queue depth.
void launch_pdlp_dual_update(double* d_y, const double* d_ax_bar, const double* d_row_lower,
                              const double* d_row_upper, double sigma, std::int32_t m,
                              double* d_y_sum, double weight, cudaStream_t stream);

// x_new <- proj_[l,u]( x - tau * (c + A^T y) );  xbar <- 2 x_new - x
//
// The extrapolated point xbar is what the NEXT dual step multiplies by A,
// so it is written into the buffer cuSPARSE is already bound to -- no copy
// and no rebinding per iteration.
void launch_pdlp_primal_update(double* d_x, double* d_x_bar, const double* d_aty,
                                const double* d_cost, const double* d_lower, const double* d_upper,
                                double tau, std::int32_t n, double* d_x_sum, double weight,
                                cudaStream_t stream);

// Computes every quantity the termination test and the restart heuristic
// need, in one launch, reducing to a single PdlpResiduals the host reads
// with one 64-byte transfer.
//
// `d_ax` must hold A x for the iterate being evaluated (NOT A xbar), and
// `d_aty` must hold A^T y for the same iterate -- the caller issues those
// two SpMVs before this kernel. Evaluating a candidate therefore costs two
// SpMVs plus this kernel, which is why it happens once per restart window
// rather than once per iteration.
void launch_pdlp_residuals(const double* d_x, const double* d_y, const double* d_ax,
                            const double* d_aty, const double* d_cost, const double* d_lower,
                            const double* d_upper, const double* d_row_lower,
                            const double* d_row_upper, const double* d_x_restart,
                            const double* d_y_restart, std::int32_t n, std::int32_t m,
                            double* d_block_scratch, std::int32_t block_capacity,
                            unsigned int* d_retire_count, PdlpResiduals* d_out,
                            cudaStream_t stream);

// out[i] = scale * sum[i]  -- turns a running sum into the average iterate.
void launch_pdlp_scale_into(double* d_out, const double* d_sum, double scale, std::int32_t n,
                             cudaStream_t stream);

// Copies src into dst on the device (restarting the iterate from the
// average, without a host round trip).
void launch_pdlp_copy(double* d_dst, const double* d_src, std::int32_t n, cudaStream_t stream);

void launch_pdlp_fill(double* d_out, double value, std::int32_t n, cudaStream_t stream);

// Sum of squares of a device vector, reduced to one double. Used only by
// the power iteration that estimates ||A||_2 at setup.
void launch_pdlp_norm_sq(const double* d_v, std::int32_t n, double* d_block_scratch,
                          std::int32_t block_capacity, unsigned int* d_retire_count,
                          double* d_out, cudaStream_t stream);

// v <- v / ||v||, for the same power iteration. Reads the norm from device
// memory so the host never has to see it.
void launch_pdlp_normalize(double* d_v, const double* d_norm_sq, std::int32_t n,
                            cudaStream_t stream);

// ---------------------------------------------------------------------
// ADAPTIVE STEP SIZE -- Applegate et al. 2021 3.1, implemented sync-free
// ---------------------------------------------------------------------
//
// An earlier revision of this file rejected the adaptive rule on the
// grounds that its accept/reject test "needs a host decision every
// iteration, and a host-visible decision per iteration is what makes GPU
// simplex uncompetitive". That reasoning was wrong, and it cost the solver
// the single most valuable feature in the algorithm.
//
// The test needs a DEVICE-visible decision, not a host-visible one. eta
// lives in device memory, every kernel reads it from there, and the
// accept/reject test is a reduction plus a small kernel -- all queued. The
// host never learns whether any individual step was accepted.
//
// Nor does the retry need a loop. A rejected step simply leaves (x, y)
// unchanged and shrinks eta, so the NEXT queued iteration is the retry.
// The whole rule becomes a predicated commit with no control flow, which
// is why it costs two extra launches per iteration and no synchronization
// at all.
//
// WHY IT IS WORTH TWO EXTRA LAUNCHES: eta = 0.9/||A||_2 is a global bound
// that must hold for every direction A can act in. The iterates do not
// explore every direction, so the LOCAL bound
//
//     eta_bar = movement / |dy' A dx|,
//     movement = 0.5 * (omega*||dx||^2 + ||dy||^2/omega)
//
// is routinely far larger. Accepting a step whenever eta <= eta_bar is
// exactly the condition PDHG's convergence proof requires, evaluated
// against the step actually taken rather than against the worst step
// possible.
//
// THE STRUCTURAL TRICK that makes dy' A dx free: the previous formulation
// computed A xbar directly, so A dx was not available without a third
// SpMV. Tracking A x instead of A xbar makes both quantities fall out of
// the same two SpMVs, because
//
//     A xbar^k = 2 A x^k - A x^{k-1}      (an inline AXPY, no SpMV)
//     A dx     = A x^{k+1} - A x^k        (a subtraction, no SpMV)
//
// so the extrapolation the dual step needs and the interaction term the
// adaptive rule needs are the same two products. SpMV count per iteration
// is unchanged at two.

// Device-resident step-size state. The host reads this once per restart
// window (never inside the loop) to learn how many steps were accepted,
// which it needs to weight the running average correctly.
struct PdlpStepState {
    double eta;          // current step size, updated every iteration
    double eta_bar;      // last local bound computed
    double dx_sq;        // ||x_cand - x||^2
    double dy_sq;        // ||y_cand - y||^2
    double interaction;  // |dy' A dx|
    double accepted;     // 1.0 if the last step was accepted, else 0.0
    double updates;      // number of step-size updates (Applegate's k)
    double accept_count; // total accepted steps this solve
};

// Initializes the state with a starting step size (the global bound).
void launch_pdlp_init_step_state(PdlpStepState* d_state, double eta0, cudaStream_t stream);

// y_cand <- prox( y + sigma * A xbar ),  A xbar = 2*ax - ax_prev computed inline.
// sigma = eta*omega with eta read from device memory.
void launch_pdlp_dual_update_adaptive(double* d_y_cand, const double* d_y, const double* d_ax,
                                       const double* d_ax_prev, const double* d_row_lower,
                                       const double* d_row_upper, const PdlpStepState* d_state,
                                       double omega, std::int32_t m, cudaStream_t stream);

// x_cand <- proj_[l,u]( x - tau * (c + A^T y_cand) ),  tau = eta/omega.
void launch_pdlp_primal_update_adaptive(double* d_x_cand, const double* d_x, const double* d_aty,
                                         const double* d_cost, const double* d_lower,
                                         const double* d_upper, const PdlpStepState* d_state,
                                         double omega, std::int32_t n, cudaStream_t stream);

// Reduces ||dx||^2, ||dy||^2 and dy'(A dx), then computes eta_bar, the
// accept flag and the next eta -- all in one launch, all on device.
void launch_pdlp_step_probe(const double* d_x, const double* d_x_cand, const double* d_y,
                             const double* d_y_cand, const double* d_ax, const double* d_ax_cand,
                             std::int32_t n, std::int32_t m, double omega, double eta_min,
                             double eta_max, double* d_block_scratch, std::int32_t block_capacity,
                             unsigned int* d_retire_count, PdlpStepState* d_state,
                             cudaStream_t stream);

// Predicated commit: on accept, advances (x, y, ax, ax_prev) and folds the
// new iterate into the running averages. On reject, does nothing -- the
// next iteration retries from the same point with the smaller eta.
void launch_pdlp_commit(double* d_x, double* d_y, double* d_ax, double* d_ax_prev,
                         double* d_x_sum, double* d_y_sum, const double* d_x_cand,
                         const double* d_y_cand, const double* d_ax_cand,
                         const PdlpStepState* d_state, std::int32_t n, std::int32_t m,
                         cudaStream_t stream);

std::int32_t pdlp_grid_size(std::int32_t n);

} // namespace gpu
} // namespace sihps
