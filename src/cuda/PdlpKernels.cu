// PDLP kernels -- see PdlpKernels.cuh for why a first-order method is the
// one place a GPU actually beats the CPU on this workload.
//
// Everything here is elementwise or a reduction, both memory-bound. That is
// fine and expected: the win is not arithmetic intensity, it is that these
// four operations per iteration need NO host synchronization, so hundreds
// of iterations queue back to back and the per-sync cost that sinks GPU
// simplex is amortized to nothing.
//
// DETERMINISM (docs/architecture/NUMERICS.md \S1)
// -----------------------------------------------
// The elementwise updates are trivially reproducible: one thread per
// element, no cross-thread arithmetic.
//
// The reductions are the interesting case, because floating-point addition
// is not associative and a tree reduction's shape must therefore be fixed.
// It is: each block reduces a grid-strided slice in a fixed order, writes
// one partial, and the last block to retire combines the partials in
// ASCENDING BLOCK INDEX order. Which block retires last varies run to run;
// the order in which their partials are summed does not. Same grid, same
// answer, bit for bit.
//
// That is weaker than it sounds unless the grid is also fixed -- so
// pdlp_grid_size is a pure function of n, and every reduction in a given
// solve uses the same grid for the same vector length.

#include "PdlpKernels.cuh"

#include "CudaCheck.hpp"

#include <cfloat>
#include <cmath>

namespace sihps {
namespace gpu {

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 512;
constexpr int kReduceSlots = 8; // one per PdlpResiduals field

__device__ __forceinline__ double clamp_to(double v, double lo, double hi) {
    // fmin/fmax propagate infinite bounds correctly: an infinite bound
    // simply never binds, which is exactly the intended semantics for a
    // free row or a free variable.
    return fmin(fmax(v, lo), hi);
}

__device__ void reduce_sum(double* shared, int tid, int block_size) {
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
}

// True in exactly one block: the last to arrive. atomicInc wraps the
// counter to zero as it issues the final ticket, so it is self-clearing
// and needs no reset launch between calls.
__device__ __forceinline__ bool is_last_block(unsigned int* retire_count, bool& flag, int tid) {
    if (tid == 0) {
        __threadfence();
        flag = (atomicInc(retire_count, gridDim.x - 1) == gridDim.x - 1);
    }
    __syncthreads();
    return flag;
}

// ---------------------------------------------------------------- updates

__global__ void k_dual_update(double* __restrict__ y, const double* __restrict__ ax_bar,
                               const double* __restrict__ row_lower,
                               const double* __restrict__ row_upper, double sigma, int m,
                               double* __restrict__ y_sum, double weight) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        const double v = y[i] + sigma * ax_bar[i];
        // Moreau: prox of the support function of [rl, ru] is
        // v - sigma * proj_[rl,ru](v / sigma).
        const double yi = v - sigma * clamp_to(v / sigma, row_lower[i], row_upper[i]);
        y[i] = yi;
        y_sum[i] += weight * yi;
    }
}

__global__ void k_primal_update(double* __restrict__ x, double* __restrict__ x_bar,
                                 const double* __restrict__ aty, const double* __restrict__ cost,
                                 const double* __restrict__ lower, const double* __restrict__ upper,
                                 double tau, int n, double* __restrict__ x_sum, double weight) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        const double xj = x[j];
        const double xn = clamp_to(xj - tau * (cost[j] + aty[j]), lower[j], upper[j]);
        // The extrapolated point the next dual step multiplies by A.
        x_bar[j] = 2.0 * xn - xj;
        x[j] = xn;
        x_sum[j] += weight * xn;
    }
}

// ---------------------------------------------------------------- residuals

__global__ void k_residuals(const double* __restrict__ x, const double* __restrict__ y,
                             const double* __restrict__ ax, const double* __restrict__ aty,
                             const double* __restrict__ cost, const double* __restrict__ lower,
                             const double* __restrict__ upper,
                             const double* __restrict__ row_lower,
                             const double* __restrict__ row_upper,
                             const double* __restrict__ x_restart,
                             const double* __restrict__ y_restart, int n, int m,
                             double* __restrict__ block_scratch, unsigned int* retire_count,
                             PdlpResiduals* __restrict__ out) {
    __shared__ double shared[kBlockSize];
    __shared__ bool last_block;
    const int tid = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    const int base = static_cast<int>(blockIdx.x * blockDim.x) + tid;

    double acc[kReduceSlots];
    for (int s = 0; s < kReduceSlots; ++s) acc[s] = 0.0;

    // ---- column space ----
    for (int j = base; j < n; j += stride) {
        const double xj = x[j];
        const double lo = lower[j];
        const double hi = upper[j];
        // Reduced cost r = c + A^T y.
        const double r = cost[j] + aty[j];

        acc[2] += cost[j] * xj;   // primal objective
        acc[4] += xj * xj;        // ||x||^2
        const double dx = xj - x_restart[j];
        acc[6] += dx * dx;        // primal movement since the last restart

        // Dual feasibility, and the dual objective's column term. With both
        // bounds finite any sign of r is admissible and the contribution is
        // min(r*lo, r*hi). With one bound infinite, r must point away from
        // it or the dual is unbounded -- the shortfall is the residual.
        const bool lo_finite = isfinite(lo);
        const bool hi_finite = isfinite(hi);
        double viol = 0.0;
        double dual_term = 0.0;
        if (lo_finite && hi_finite) {
            dual_term = fmin(r * lo, r * hi);
        } else if (lo_finite) {
            viol = fmax(0.0, -r); // need r >= 0
            dual_term = fmax(0.0, r) * lo;
        } else if (hi_finite) {
            viol = fmax(0.0, r); // need r <= 0
            dual_term = fmin(0.0, r) * hi;
        } else {
            viol = fabs(r); // free variable: need r == 0
        }
        acc[1] += viol * viol;
        acc[3] += dual_term;
    }

    // ---- row space ----
    for (int i = base; i < m; i += stride) {
        const double axi = ax[i];
        const double rl = row_lower[i];
        const double ru = row_upper[i];
        const double pr = axi - clamp_to(axi, rl, ru);
        acc[0] += pr * pr; // primal residual

        const double yi = y[i];
        acc[5] += yi * yi;
        const double dy = yi - y_restart[i];
        acc[7] += dy * dy;

        // -sigma_C(y): the support function's row term, subtracted from the
        // dual objective. An infinite bound on the side y points to would
        // make this unbounded; the sign conditions above keep y off it, and
        // the term is simply dropped when the bound is infinite (the dual
        // residual already records the violation).
        double support = 0.0;
        if (yi > 0.0) {
            support = isfinite(ru) ? yi * ru : 0.0;
        } else if (yi < 0.0) {
            support = isfinite(rl) ? yi * rl : 0.0;
        }
        acc[3] -= support;
    }

    // ---- per-block reduction, one slot at a time ----
    for (int s = 0; s < kReduceSlots; ++s) {
        __syncthreads();
        shared[tid] = acc[s];
        __syncthreads();
        reduce_sum(shared, tid, kBlockSize);
        if (tid == 0) block_scratch[blockIdx.x * kReduceSlots + s] = shared[0];
    }

    if (gridDim.x > 1 && !is_last_block(retire_count, last_block, tid)) return;

    // ---- cross-block combine, in ascending block index ----
    for (int s = 0; s < kReduceSlots; ++s) {
        double v = 0.0;
        for (int b = tid; b < static_cast<int>(gridDim.x); b += kBlockSize) {
            v += block_scratch[b * kReduceSlots + s];
        }
        __syncthreads();
        shared[tid] = v;
        __syncthreads();
        reduce_sum(shared, tid, kBlockSize);
        if (tid == 0) {
            double* fields = reinterpret_cast<double*>(out);
            fields[s] = shared[0];
        }
    }
}

// ---------------------------------------------------------------- helpers

__global__ void k_scale_into(double* __restrict__ out, const double* __restrict__ sum, double scale,
                              int n) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        out[j] = scale * sum[j];
    }
}

__global__ void k_copy(double* __restrict__ dst, const double* __restrict__ src, int n) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        dst[j] = src[j];
    }
}

__global__ void k_fill(double* __restrict__ out, double value, int n) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        out[j] = value;
    }
}

__global__ void k_norm_sq(const double* __restrict__ v, int n, double* __restrict__ block_scratch,
                           unsigned int* retire_count, double* __restrict__ out) {
    __shared__ double shared[kBlockSize];
    __shared__ bool last_block;
    const int tid = static_cast<int>(threadIdx.x);

    double acc = 0.0;
    for (int j = blockIdx.x * blockDim.x + tid; j < n; j += blockDim.x * gridDim.x) {
        acc += v[j] * v[j];
    }
    shared[tid] = acc;
    __syncthreads();
    reduce_sum(shared, tid, kBlockSize);
    if (tid == 0) block_scratch[blockIdx.x] = shared[0];

    if (gridDim.x == 1) {
        if (tid == 0) *out = shared[0];
        return;
    }
    if (!is_last_block(retire_count, last_block, tid)) return;

    double v2 = 0.0;
    for (int b = tid; b < static_cast<int>(gridDim.x); b += kBlockSize) v2 += block_scratch[b];
    __syncthreads();
    shared[tid] = v2;
    __syncthreads();
    reduce_sum(shared, tid, kBlockSize);
    if (tid == 0) *out = shared[0];
}

__global__ void k_normalize(double* __restrict__ v, const double* __restrict__ norm_sq, int n) {
    const double nrm = sqrt(*norm_sq);
    if (!(nrm > 0.0)) return;
    const double inv = 1.0 / nrm;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        v[j] *= inv;
    }
}

} // namespace

std::int32_t pdlp_grid_size(std::int32_t n) {
    if (n <= 0) return 1;
    const std::int32_t blocks = (n + kBlockSize - 1) / kBlockSize;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

void launch_pdlp_dual_update(double* d_y, const double* d_ax_bar, const double* d_row_lower,
                              const double* d_row_upper, double sigma, std::int32_t m,
                              double* d_y_sum, double weight, cudaStream_t stream) {
    if (m <= 0) return;
    k_dual_update<<<pdlp_grid_size(m), kBlockSize, 0, stream>>>(d_y, d_ax_bar, d_row_lower,
                                                                 d_row_upper, sigma, m, d_y_sum,
                                                                 weight);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_primal_update(double* d_x, double* d_x_bar, const double* d_aty,
                                const double* d_cost, const double* d_lower, const double* d_upper,
                                double tau, std::int32_t n, double* d_x_sum, double weight,
                                cudaStream_t stream) {
    if (n <= 0) return;
    k_primal_update<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(
        d_x, d_x_bar, d_aty, d_cost, d_lower, d_upper, tau, n, d_x_sum, weight);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_residuals(const double* d_x, const double* d_y, const double* d_ax,
                            const double* d_aty, const double* d_cost, const double* d_lower,
                            const double* d_upper, const double* d_row_lower,
                            const double* d_row_upper, const double* d_x_restart,
                            const double* d_y_restart, std::int32_t n, std::int32_t m,
                            double* d_block_scratch, std::int32_t block_capacity,
                            unsigned int* d_retire_count, PdlpResiduals* d_out,
                            cudaStream_t stream) {
    std::int32_t grid = pdlp_grid_size(n > m ? n : m);
    if (grid > block_capacity) grid = block_capacity;
    k_residuals<<<grid, kBlockSize, 0, stream>>>(d_x, d_y, d_ax, d_aty, d_cost, d_lower, d_upper,
                                                  d_row_lower, d_row_upper, d_x_restart,
                                                  d_y_restart, n, m, d_block_scratch,
                                                  d_retire_count, d_out);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_scale_into(double* d_out, const double* d_sum, double scale, std::int32_t n,
                             cudaStream_t stream) {
    if (n <= 0) return;
    k_scale_into<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(d_out, d_sum, scale, n);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_copy(double* d_dst, const double* d_src, std::int32_t n, cudaStream_t stream) {
    if (n <= 0) return;
    k_copy<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(d_dst, d_src, n);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_fill(double* d_out, double value, std::int32_t n, cudaStream_t stream) {
    if (n <= 0) return;
    k_fill<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(d_out, value, n);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_norm_sq(const double* d_v, std::int32_t n, double* d_block_scratch,
                          std::int32_t block_capacity, unsigned int* d_retire_count, double* d_out,
                          cudaStream_t stream) {
    if (n <= 0) return;
    std::int32_t grid = pdlp_grid_size(n);
    if (grid > block_capacity) grid = block_capacity;
    k_norm_sq<<<grid, kBlockSize, 0, stream>>>(d_v, n, d_block_scratch, d_retire_count, d_out);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_normalize(double* d_v, const double* d_norm_sq, std::int32_t n,
                            cudaStream_t stream) {
    if (n <= 0) return;
    k_normalize<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(d_v, d_norm_sq, n);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------
// Adaptive step size -- see PdlpKernels.cuh for why this is sync-free.
// ---------------------------------------------------------------------

namespace {

constexpr int kStepSlots = 3; // dx_sq, dy_sq, interaction

__global__ void k_init_step_state(PdlpStepState* state, double eta0) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    state->eta = eta0;
    state->eta_bar = eta0;
    state->dx_sq = 0.0;
    state->dy_sq = 0.0;
    state->interaction = 0.0;
    state->accepted = 1.0;
    state->updates = 0.0;
    state->accept_count = 0.0;
}

// y_cand = prox( y + sigma * A xbar ),  A xbar = 2*ax - ax_prev.
//
// The extrapolation is done here on the already-computed products rather
// than by a separate SpMV -- that substitution is what makes the
// interaction term in the adaptive rule free.
__global__ void k_dual_update_adaptive(double* y_cand, const double* y, const double* ax,
                                       const double* ax_prev, const double* row_lower,
                                       const double* row_upper, const PdlpStepState* state,
                                       double omega, std::int32_t m) {
    const double sigma = state->eta * omega;
    for (std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        const double ax_bar = 2.0 * ax[i] - ax_prev[i];
        const double v = y[i] + sigma * ax_bar;
        // Moreau: prox_{sigma . support_C}(v) = v - sigma * proj_C(v/sigma)
        y_cand[i] = v - sigma * clamp_to(v / sigma, row_lower[i], row_upper[i]);
    }
}

__global__ void k_primal_update_adaptive(double* x_cand, const double* x, const double* aty,
                                         const double* cost, const double* lower,
                                         const double* upper, const PdlpStepState* state,
                                         double omega, std::int32_t n) {
    const double tau = state->eta / omega;
    for (std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        x_cand[i] = clamp_to(x[i] - tau * (cost[i] + aty[i]), lower[i], upper[i]);
    }
}

// Reduces the three quantities the adaptive rule needs, then -- in the last
// block to retire -- computes the local bound, the accept flag and the next
// step size. One launch, no host involvement.
__global__ void k_step_probe(const double* x, const double* x_cand, const double* y,
                             const double* y_cand, const double* ax, const double* ax_cand,
                             std::int32_t n, std::int32_t m, double omega, double eta_min,
                             double eta_max, double* block_scratch, unsigned int* retire_count,
                             PdlpStepState* state) {
    __shared__ double shared[kBlockSize];
    __shared__ bool last;

    const int tid = threadIdx.x;
    const std::int32_t stride = gridDim.x * blockDim.x;
    const std::int32_t start = blockIdx.x * blockDim.x + tid;

    double acc[kStepSlots] = {0.0, 0.0, 0.0};
    for (std::int32_t i = start; i < n; i += stride) {
        const double dx = x_cand[i] - x[i];
        acc[0] += dx * dx;
    }
    for (std::int32_t i = start; i < m; i += stride) {
        const double dy = y_cand[i] - y[i];
        const double adx = ax_cand[i] - ax[i];
        acc[1] += dy * dy;
        acc[2] += dy * adx;
    }

    for (int s = 0; s < kStepSlots; ++s) {
        shared[tid] = acc[s];
        __syncthreads();
        reduce_sum(shared, tid, kBlockSize);
        if (tid == 0) block_scratch[s * gridDim.x + blockIdx.x] = shared[0];
        __syncthreads();
    }

    if (!is_last_block(retire_count, last, tid)) return;

    // Fixed ascending-block-index combine keeps this bit-reproducible even
    // though which block arrives last does not.
    double total[kStepSlots];
    for (int s = 0; s < kStepSlots; ++s) {
        shared[tid] = 0.0;
        for (std::int32_t b = tid; b < gridDim.x; b += kBlockSize) {
            shared[tid] += block_scratch[s * gridDim.x + b];
        }
        __syncthreads();
        reduce_sum(shared, tid, kBlockSize);
        total[s] = shared[0];
        __syncthreads();
    }
    if (tid != 0) return;

    const double dx_sq = total[0];
    const double dy_sq = total[1];
    const double interaction = fabs(total[2]);

    // movement is the weighted distance travelled; interaction is how much
    // of that distance A actually couples. Their ratio is the largest step
    // for which the descent inequality still holds along THIS direction.
    const double movement = 0.5 * (omega * dx_sq + dy_sq / omega);
    const double eta_used = state->eta;

    double eta_bar;
    if (interaction > 0.0 && movement > 0.0) {
        eta_bar = movement / interaction;
    } else {
        // No coupling (or no movement): the step cannot violate the bound.
        eta_bar = eta_max;
    }

    const bool accept = (eta_used <= eta_bar);
    const double k = state->updates + 1.0;

    // Applegate's two-sided update: never exceed the freshly measured local
    // bound by more than a shrinking margin, and never grow faster than a
    // shrinking factor. Both exponents decay so the sequence is summable,
    // which is what makes the whole scheme convergent rather than merely
    // heuristic.
    const double shrink = 1.0 - pow(k + 1.0, -0.3);
    const double grow = 1.0 + pow(k + 1.0, -0.6);
    double eta_next = fmin(shrink * eta_bar, grow * eta_used);
    if (!(eta_next > 0.0) || !isfinite(eta_next)) eta_next = eta_used;
    eta_next = fmin(fmax(eta_next, eta_min), eta_max);

    state->eta = eta_next;
    state->eta_bar = eta_bar;
    state->dx_sq = dx_sq;
    state->dy_sq = dy_sq;
    state->interaction = interaction;
    state->accepted = accept ? 1.0 : 0.0;
    state->updates = k;
    if (accept) state->accept_count += 1.0;
}

// Predicated commit. On reject this kernel writes nothing and the next
// iteration retries from the same point with the smaller eta -- which is
// how backtracking is expressed without any control flow.
__global__ void k_commit(double* x, double* y, double* ax, double* ax_prev, double* x_sum,
                         double* y_sum, const double* x_cand, const double* y_cand,
                         const double* ax_cand, const PdlpStepState* state, std::int32_t n,
                         std::int32_t m) {
    if (state->accepted == 0.0) return;
    const std::int32_t stride = gridDim.x * blockDim.x;
    const std::int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
    for (std::int32_t i = start; i < n; i += stride) {
        const double xv = x_cand[i];
        x[i] = xv;
        x_sum[i] += xv;
    }
    for (std::int32_t i = start; i < m; i += stride) {
        const double yv = y_cand[i];
        y[i] = yv;
        y_sum[i] += yv;
        ax_prev[i] = ax[i];
        ax[i] = ax_cand[i];
    }
}

} // namespace

void launch_pdlp_init_step_state(PdlpStepState* d_state, double eta0, cudaStream_t stream) {
    k_init_step_state<<<1, 1, 0, stream>>>(d_state, eta0);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_dual_update_adaptive(double* d_y_cand, const double* d_y, const double* d_ax,
                                       const double* d_ax_prev, const double* d_row_lower,
                                       const double* d_row_upper, const PdlpStepState* d_state,
                                       double omega, std::int32_t m, cudaStream_t stream) {
    if (m <= 0) return;
    k_dual_update_adaptive<<<pdlp_grid_size(m), kBlockSize, 0, stream>>>(
        d_y_cand, d_y, d_ax, d_ax_prev, d_row_lower, d_row_upper, d_state, omega, m);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_primal_update_adaptive(double* d_x_cand, const double* d_x, const double* d_aty,
                                         const double* d_cost, const double* d_lower,
                                         const double* d_upper, const PdlpStepState* d_state,
                                         double omega, std::int32_t n, cudaStream_t stream) {
    if (n <= 0) return;
    k_primal_update_adaptive<<<pdlp_grid_size(n), kBlockSize, 0, stream>>>(
        d_x_cand, d_x, d_aty, d_cost, d_lower, d_upper, d_state, omega, n);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_step_probe(const double* d_x, const double* d_x_cand, const double* d_y,
                             const double* d_y_cand, const double* d_ax, const double* d_ax_cand,
                             std::int32_t n, std::int32_t m, double omega, double eta_min,
                             double eta_max, double* d_block_scratch, std::int32_t block_capacity,
                             unsigned int* d_retire_count, PdlpStepState* d_state,
                             cudaStream_t stream) {
    const std::int32_t len = n > m ? n : m;
    if (len <= 0) return;
    std::int32_t grid = pdlp_grid_size(len);
    if (grid > block_capacity) grid = block_capacity;
    k_step_probe<<<grid, kBlockSize, 0, stream>>>(d_x, d_x_cand, d_y, d_y_cand, d_ax, d_ax_cand, n,
                                                   m, omega, eta_min, eta_max, d_block_scratch,
                                                   d_retire_count, d_state);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_pdlp_commit(double* d_x, double* d_y, double* d_ax, double* d_ax_prev, double* d_x_sum,
                         double* d_y_sum, const double* d_x_cand, const double* d_y_cand,
                         const double* d_ax_cand, const PdlpStepState* d_state, std::int32_t n,
                         std::int32_t m, cudaStream_t stream) {
    const std::int32_t len = n > m ? n : m;
    if (len <= 0) return;
    k_commit<<<pdlp_grid_size(len), kBlockSize, 0, stream>>>(
        d_x, d_y, d_ax, d_ax_prev, d_x_sum, d_y_sum, d_x_cand, d_y_cand, d_ax_cand, d_state, n, m);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace sihps
