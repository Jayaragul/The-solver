// Custom CUDA kernels for the pricing step of the revised simplex.
//
// WHY THESE EXIST AND THE SpMV KERNEL DOES NOT
// --------------------------------------------
// prompt.md \S3.5 forbids a hand-written SpMV kernel ("use cuSPARSE
// correctly"), and this project obeys that: A^T v is still cusparseSpMV
// (GpuSpMV.cpp). \S3.2's GPU-candidate list separately names "pricing-
// related kernels where appropriate", "vector arithmetic" and "sparse
// reductions", which is what this file implements. cuSPARSE's SpMV is a
// tuned library primitive no reimplementation here would beat; the pricing
// scan is a problem-specific fused elementwise-plus-argmax pass no library
// exposes.
//
// WHAT THE KERNELS BUY, IN PCIe TERMS
// -----------------------------------
// The naive GPU pricing placement runs cusparseSpMV and ships the whole
// structural result back, leaving the reduced-cost assembly and the
// entering-variable argmax as host loops. Per iteration that is an O(n)
// device-to-host transfer plus O(n) host work -- the GPU does the one cheap
// part and the host still pays for everything else, plus the round trip.
// With the argmax on device, one 24-byte PricingCandidate crosses back per
// iteration, independent of n.
//
// WHY EVERY PASS IS A SINGLE LAUNCH
// ---------------------------------
// This is the design constraint that shaped the whole file, and it was
// MEASURED rather than assumed (benchmarks/bench_gpu_latency.cpp). A GPU
// pricing iteration on this machine costs ~230 us for a 1,692-nonzero model
// and ~256 us for a 129,018-nonzero one: 76x the arithmetic for 11% more
// time. The cost is not bandwidth and it is not FLOPs -- it is the NUMBER
// OF QUEUED OPERATIONS, each of which costs host submission time (~7 us)
// and device-side dispatch time (~10 us) whether it does any work or not.
//
// So the optimization target here is op count, not throughput, and that
// inverts several textbook choices:
//
//   - The argmax uses a THREADFENCE reduction (the last block to retire
//     performs the final combine) instead of the conventional two-kernel
//     scheme. Two kernels is the tidier structure and would be the right
//     call on a throughput-bound problem; here the second launch costs more
//     than the reduction it performs.
//   - The Devex update folds its weight update, its reference-framework
//     maximum, and the conditional restart into one launch for the same
//     reason -- three kernels became one.
//   - The Devex update reads NO status array of its own. It reuses the one
//     pricing already uploaded this iteration and applies the two entries
//     that changed (the entering and leaving variables) as scalar
//     parameters. That removes an entire H2D per iteration at zero
//     correctness risk, because pricing re-uploads the full array next
//     iteration regardless.
//
// Net: 11 queued operations per Devex iteration became 8, with a further
// collapse of the submission half handled by CUDA graph capture in
// GpuPricer.
//
// DETERMINISM SURVIVES ALL OF IT
// ------------------------------
// The threadfence pattern makes WHICH block finishes last nondeterministic,
// which would be fatal if the combine were order-sensitive. It is not: the
// final block reduces the per-block results in fixed index order using a
// comparator that is a total order on (score descending, index ascending).
// Any block can be last and the answer is the same. NUMERICS.md \S1 holds.
//
// ARITHMETIC INTENSITY (prompt.md \S3.2 requires this stated, not assumed)
// -----------------------------------------------------------------------
// Every kernel here is memory-bound: the fused pricing pass does roughly 5
// flops per 25 bytes moved, and will never approach peak FP64 throughput.
// No amount of tuning changes that, and at the sizes measured above it does
// not matter -- these kernels spend most of their wall time being dispatched
// rather than executing. Occupancy is bounded by 256-thread blocks with 6 KB
// of shared memory each. Branch divergence inside the eligibility test is
// real but the branches are short, and a branchless form would cost every
// thread the full arithmetic instead of only the eligible ones.

#include "PricingKernels.cuh"

#include "CudaCheck.hpp"

#include <cfloat>
#include <cmath>

namespace sihps {
namespace gpu {

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

// The augmented-space source vector, assembled on the fly. Every consumer
// reads each entry exactly once, so materializing it would cost a launch
// and a full write-then-read for no reuse -- see the header comment.
//
//   src[j] = z[j]                      structural j   (z = A^T v)
//          = v[j - n_struct]           slack j        (slack column = e_i)
//          = art_sign[i] * v[i]        artificial j   (artificial column = +-e_i)
__device__ __forceinline__ double augmented_src(int j, const double* __restrict__ spmv_out,
                                                 const double* __restrict__ v,
                                                 const double* __restrict__ art_sign, int n_struct,
                                                 int n_slack) {
    if (j < n_struct) return spmv_out[j];
    if (j < n_struct + n_slack) return v[j - n_struct];
    const int i = j - n_struct - n_slack;
    return art_sign[i] * v[i];
}

// Strict "a is a better candidate than b": a total order on (score
// descending, index ascending) with index < 0 as the identity. Total order
// plus associativity is what lets the reduction be combined in any order --
// including the nondeterministic block-retirement order below -- and still
// give one answer.
__device__ __forceinline__ bool better(const PricingCandidate& a, const PricingCandidate& b) {
    if (b.index < 0) return a.index >= 0;
    if (a.index < 0) return false;
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
}

__device__ __forceinline__ PricingCandidate none_candidate() {
    PricingCandidate c;
    c.score = 0.0;
    c.reduced_cost = 0.0;
    c.index = -1;
    c.direction = 0;
    return c;
}

__device__ void reduce_candidates(PricingCandidate* shared, int tid, int block_size) {
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && better(shared[tid + stride], shared[tid])) {
            shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }
}

__device__ void reduce_max(double* shared, int tid, int block_size) {
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmax(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
}

// Returns true in exactly one block: the last one to arrive. atomicInc with
// a wrap value of gridDim.x - 1 resets the counter to zero as it hands out
// the final ticket, so the counter is self-clearing and the next launch
// needs no separate memset (which would be another queued operation --
// exactly what this file is trying to avoid).
__device__ __forceinline__ bool is_last_block(unsigned int* retire_count, bool& shared_flag,
                                               int tid) {
    if (tid == 0) {
        __threadfence(); // make this block's global writes visible first
        const unsigned int ticket = atomicInc(retire_count, gridDim.x - 1);
        shared_flag = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    return shared_flag;
}

__global__ void k_assemble_augmented(const double* __restrict__ spmv_out,
                                      const double* __restrict__ v,
                                      const double* __restrict__ cost,
                                      const double* __restrict__ art_sign, double* __restrict__ out,
                                      int n_struct, int n_slack, int n_art, double alpha,
                                      double beta) {
    const int n_total = n_struct + n_slack + n_art;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_total; j += blockDim.x * gridDim.x) {
        const double src = augmented_src(j, spmv_out, v, art_sign, n_struct, n_slack);
        out[j] = (alpha != 0.0 ? alpha * cost[j] : 0.0) + beta * src;
    }
}

// Reduced cost, eligibility, score and argmax across the whole augmented
// column set -- assembled, scored and fully reduced in ONE launch.
__global__ void k_price_fused(const double* __restrict__ spmv_out, const double* __restrict__ v,
                               const double* __restrict__ cost,
                               const double* __restrict__ art_sign,
                               const double* __restrict__ weight,
                               const unsigned char* __restrict__ status,
                               const double* __restrict__ lower, const double* __restrict__ upper,
                               int n_struct, int n_slack, int n_art, int devex, double opt_tol,
                               PricingCandidate* __restrict__ block_out,
                               unsigned int* __restrict__ retire_count,
                               PricingCandidate* __restrict__ result) {
    __shared__ PricingCandidate shared[kBlockSize];
    __shared__ bool last_block;
    const int tid = static_cast<int>(threadIdx.x);
    const int n_total = n_struct + n_slack + n_art;

    PricingCandidate best = none_candidate();

    for (int j = blockIdx.x * blockDim.x + tid; j < n_total; j += blockDim.x * gridDim.x) {
        const unsigned char st = status[j];
        if (st == kBasic) continue;
        if (upper[j] - lower[j] < 1e-12) continue; // fixed column: can never move

        const double d = cost[j] - augmented_src(j, spmv_out, v, art_sign, n_struct, n_slack);
        int dq = 0;
        if (st == kAtLower) {
            if (d < -opt_tol) dq = 1;
        } else if (st == kAtUpper) {
            if (d > opt_tol) dq = -1;
        } else { // AT_ZERO: a free variable may improve in either direction
            if (fabs(d) > opt_tol) dq = (d < 0.0) ? 1 : -1;
        }
        if (dq == 0) continue;

        PricingCandidate cand;
        cand.score = devex ? (d * d) / weight[j] : fabs(d);
        cand.reduced_cost = d;
        cand.index = j;
        cand.direction = dq;
        if (better(cand, best)) best = cand;
    }

    shared[tid] = best;
    __syncthreads();
    reduce_candidates(shared, tid, kBlockSize);
    if (tid == 0) block_out[blockIdx.x] = shared[0];

    // Single-block launches are the common case at Netlib scale and skip
    // the cross-block combine entirely.
    if (gridDim.x == 1) {
        if (tid == 0) *result = shared[0];
        return;
    }

    if (!is_last_block(retire_count, last_block, tid)) return;

    // Fixed index order over the per-block winners: which block gets here
    // varies, the answer does not.
    PricingCandidate final_best = none_candidate();
    for (int i = tid; i < static_cast<int>(gridDim.x); i += kBlockSize) {
        if (better(block_out[i], final_best)) final_best = block_out[i];
    }
    __syncthreads(); // shared[] is being reused for a second reduction
    shared[tid] = final_best;
    __syncthreads();
    reduce_candidates(shared, tid, kBlockSize);
    if (tid == 0) *result = shared[0];
}

// Devex reference-weight update (Harris 1973) with the pivot row assembled
// inline, the reference maximum reduced, the leaving variable's new weight
// installed, and the framework restart applied -- all in ONE launch.
//
// `status` is PRICING's upload from earlier this iteration, deliberately
// not re-sent. Exactly two entries have changed since: `entering` became
// basic (skipped anyway) and `leaving_var` became nonbasic (applied as an
// overlay below). Everything else is untouched between the two calls.
__global__ void k_devex_fused(double* __restrict__ weight, const double* __restrict__ spmv_out,
                               const double* __restrict__ v, const double* __restrict__ art_sign,
                               const unsigned char* __restrict__ status, int n_struct, int n_slack,
                               int n_art, int entering, int leaving_var, double pivot_sq,
                               double restart_threshold, double* __restrict__ block_max,
                               unsigned int* __restrict__ retire_count) {
    __shared__ double shared[kBlockSize];
    __shared__ bool last_block;
    __shared__ double max_weight;
    const int tid = static_cast<int>(threadIdx.x);
    const int n_total = n_struct + n_slack + n_art;

    // Broadcast read: no thread writes weight[entering] (the entering
    // column is skipped below), so this is race-free by construction.
    const double ratio_scale = weight[entering] / pivot_sq;

    double local_max = 1.0; // matches the CPU path's max_weight seed
    for (int j = blockIdx.x * blockDim.x + tid; j < n_total; j += blockDim.x * gridDim.x) {
        if (j == entering) continue;
        if (status[j] == kBasic && j != leaving_var) continue;
        const double r = augmented_src(j, spmv_out, v, art_sign, n_struct, n_slack);
        if (r != 0.0) {
            const double candidate = r * r * ratio_scale;
            if (candidate > weight[j]) weight[j] = candidate;
        }
        local_max = fmax(local_max, weight[j]);
    }

    shared[tid] = local_max;
    __syncthreads();
    reduce_max(shared, tid, kBlockSize);
    if (tid == 0) block_max[blockIdx.x] = shared[0];

    const bool single = (gridDim.x == 1);
    if (!single && !is_last_block(retire_count, last_block, tid)) return;

    if (!single) {
        double acc = 1.0;
        for (int i = tid; i < static_cast<int>(gridDim.x); i += kBlockSize) {
            acc = fmax(acc, block_max[i]);
        }
        __syncthreads();
        shared[tid] = acc;
        __syncthreads();
        reduce_max(shared, tid, kBlockSize);
    }

    if (tid == 0) {
        // The reduction above already saw whatever the update loop left in
        // the leaving variable's slot -- exactly what the CPU path's
        // max_weight observes before overwriting it.
        const double w_leaving = fmax(1.0, ratio_scale);
        weight[leaving_var] = w_leaving;
        max_weight = fmax(shared[0], w_leaving);
    }
    __syncthreads();

    // Weights are approximations whose error compounds; past a few orders
    // of magnitude Harris's scheme discards them rather than letting them
    // drift. Folded in here so the conditional never leaves the device --
    // testing it on the host would force a synchronize in the hot path
    // (prompt.md \S3.6).
    if (max_weight > restart_threshold) {
        for (int j = tid; j < n_total; j += kBlockSize) weight[j] = 1.0;
    }
}

__global__ void k_fill(double* __restrict__ out, int n, double value) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        out[j] = value;
    }
}

__global__ void k_noop() {}

} // namespace

std::int32_t pricing_grid_size(std::int32_t n) {
    if (n <= 0) return 1;
    const std::int32_t blocks = (n + kBlockSize - 1) / kBlockSize;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

void launch_assemble_augmented(const double* d_spmv_out, const double* d_v, const double* d_cost,
                                const double* d_art_sign, double* d_out, std::int32_t n_struct,
                                std::int32_t n_slack, std::int32_t n_art, double alpha, double beta,
                                cudaStream_t stream) {
    const std::int32_t n_total = n_struct + n_slack + n_art;
    if (n_total <= 0) return;
    k_assemble_augmented<<<pricing_grid_size(n_total), kBlockSize, 0, stream>>>(
        d_spmv_out, d_v, d_cost, d_art_sign, d_out, n_struct, n_slack, n_art, alpha, beta);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_price_argmax(const double* d_spmv_out, const double* d_v, const double* d_cost,
                          const double* d_art_sign, const double* d_weight,
                          const std::uint8_t* d_status, const double* d_lower,
                          const double* d_upper, std::int32_t n_struct, std::int32_t n_slack,
                          std::int32_t n_art, bool devex, double opt_tol,
                          PricingCandidate* d_block_scratch, std::int32_t block_scratch_capacity,
                          unsigned int* d_retire_count, PricingCandidate* d_result,
                          cudaStream_t stream) {
    const std::int32_t n_total = n_struct + n_slack + n_art;
    if (n_total <= 0) return;
    std::int32_t grid = pricing_grid_size(n_total);
    if (grid > block_scratch_capacity) grid = block_scratch_capacity;

    k_price_fused<<<grid, kBlockSize, 0, stream>>>(
        d_spmv_out, d_v, d_cost, d_art_sign, d_weight, d_status, d_lower, d_upper, n_struct,
        n_slack, n_art, devex ? 1 : 0, opt_tol, d_block_scratch, d_retire_count, d_result);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_devex_update(double* d_weight, const double* d_spmv_out, const double* d_v,
                          const double* d_art_sign, const std::uint8_t* d_status,
                          std::int32_t n_struct, std::int32_t n_slack, std::int32_t n_art,
                          std::int32_t entering, std::int32_t leaving_var, double pivot_sq,
                          double restart_threshold, double* d_block_max,
                          std::int32_t block_max_capacity, unsigned int* d_retire_count,
                          cudaStream_t stream) {
    const std::int32_t n_total = n_struct + n_slack + n_art;
    if (n_total <= 0) return;
    std::int32_t grid = pricing_grid_size(n_total);
    if (grid > block_max_capacity) grid = block_max_capacity;

    k_devex_fused<<<grid, kBlockSize, 0, stream>>>(
        d_weight, d_spmv_out, d_v, d_art_sign, d_status, n_struct, n_slack, n_art, entering,
        leaving_var, pivot_sq, restart_threshold, d_block_max, d_retire_count);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_fill(double* d_out, std::int32_t n, double value, cudaStream_t stream) {
    if (n <= 0) return;
    k_fill<<<pricing_grid_size(n), kBlockSize, 0, stream>>>(d_out, n, value);
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

void launch_noop(cudaStream_t stream) {
    k_noop<<<1, 1, 0, stream>>>();
    SIHPS_CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace sihps
