#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace sihps {
namespace gpu {

// Device mirror of Simplex::VarStatus. The enum values are asserted equal
// to the host enum's in Simplex.cpp -- a silent divergence here would make
// the GPU price a different LP than the CPU, which is exactly the class of
// bug the CPU/GPU equivalence tests (prompt.md \S3.7) exist to catch, so it
// is pinned by a static_assert rather than a comment.
enum : std::uint8_t { kAtLower = 0, kAtUpper = 1, kAtZero = 2, kBasic = 3 };

// Result of the entering-variable search: the ONLY thing that crosses PCIe
// per pricing call on the fast path. That is the whole point of doing the
// argmax on device -- see GpuPricer.hpp.
//
// `index < 0` is the "no eligible column" sentinel, which the primal loop
// reads as OPTIMAL.
struct PricingCandidate {
    double score;
    double reduced_cost;
    std::int32_t index;
    std::int32_t direction; // +1 entering increases, -1 entering decreases
};

// ---------------------------------------------------------------------
// The augmented-space vector both pricing and Devex are built on
// ---------------------------------------------------------------------
// Every kernel below reads the same conceptual vector, assembled from an
// SpMV result over the structural columns plus the analytically-known
// slack and artificial tails:
//
//   src[j] = z[j]                      for structural j   (z = A^T v)
//          = v[j - n_struct]           for slack j        (slack column = e_i)
//          = art_sign[i] * v[i]        for artificial j   (artificial column = +-e_i)
//
// Reduced costs are  d   = c - src  (with v = the duals y), and the Devex
// pivot row is       rho =     src  (with v = B^-T e_r). One assembly rule,
// two coefficient choices.
//
// The kernels FUSE that assembly into whatever consumes it rather than
// materializing the vector, because every consumer reads each entry
// exactly once. Materializing it would cost an extra kernel launch plus a
// full write-then-read of n_total doubles for no reuse whatsoever -- and
// on this workload launch latency, not bandwidth, is what dominates
// (measured: see bench_pricing_backend.cpp's latency probe). Only
// price_to_host, which genuinely has to hand the whole vector to the host,
// uses the standalone assembly kernel.

// Standalone assembly -- for the host-visible reduced-cost vector only.
void launch_assemble_augmented(const double* d_spmv_out, const double* d_v, const double* d_cost,
                                const double* d_art_sign, double* d_out, std::int32_t n_struct,
                                std::int32_t n_slack, std::int32_t n_art, double alpha, double beta,
                                cudaStream_t stream);

// Fused reduced-cost + eligibility + score + argmax -- ONE launch,
// whatever the grid size.
//
// Why one and not the conventional two-stage reduction: on this workload
// the binding cost is the number of queued operations, not throughput
// (measured -- benchmarks/bench_gpu_latency.cpp, and the reasoning in
// PricingKernels.cu's header). A second launch costs more than the combine
// it would perform, so the last block to retire does the combine instead.
//
// `d_retire_count` is a single-element device counter used for that
// hand-off. It is SELF-CLEARING (atomicInc wraps it to zero as it issues
// the final ticket), so no reset operation is needed between launches --
// which would have reintroduced exactly the launch this design removes.
// It must be zero-initialized once, at construction.
//
// DETERMINISM (docs/architecture/NUMERICS.md \S1): which block retires
// last is genuinely nondeterministic, and that is harmless here. The
// combine runs over the per-block winners in fixed index order under a
// comparator that is a total order on (score descending, index ascending),
// so every possible retirement order yields the same winner. Ties resolve
// to the lowest index, matching the CPU's ascending strict-greater scan --
// so the two backends agree on the entering column whenever their scores
// agree bit for bit.
void launch_price_argmax(const double* d_spmv_out, const double* d_v, const double* d_cost,
                          const double* d_art_sign, const double* d_weight,
                          const std::uint8_t* d_status, const double* d_lower,
                          const double* d_upper, std::int32_t n_struct, std::int32_t n_slack,
                          std::int32_t n_art, bool devex, double opt_tol,
                          PricingCandidate* d_block_scratch, std::int32_t block_scratch_capacity,
                          unsigned int* d_retire_count, PricingCandidate* d_result,
                          cudaStream_t stream);

// Fused Devex reference-weight update (Harris 1973), device-resident:
// weights live on the GPU across iterations and are never shipped to the
// host. ONE launch, covering the pivot-row assembly, the weight update, the
// reference-framework maximum, the leaving variable's new weight, and the
// conditional restart. The restart condition never leaves the device --
// testing it on the host would force a synchronize in the hot path
// (prompt.md \S3.6).
//
// `d_status` is PRICING's upload from earlier in the same iteration, NOT a
// fresh one. Exactly two entries changed in between -- `entering` became
// basic (which this kernel skips regardless) and `leaving_var` became
// nonbasic (applied as a scalar overlay) -- so re-sending the array would
// buy nothing and cost a transfer. Pricing re-uploads it in full next
// iteration, so no staleness can accumulate.
//
// `pivot_sq` must be > 0; the caller (which holds the pivot on the host)
// skips the update entirely otherwise, matching the CPU path.
void launch_devex_update(double* d_weight, const double* d_spmv_out, const double* d_v,
                          const double* d_art_sign, const std::uint8_t* d_status,
                          std::int32_t n_struct, std::int32_t n_slack, std::int32_t n_art,
                          std::int32_t entering, std::int32_t leaving_var, double pivot_sq,
                          double restart_threshold, double* d_block_max,
                          std::int32_t block_max_capacity, unsigned int* d_retire_count,
                          cudaStream_t stream);

// Fills a device array with a constant. Used to reset Devex weights to all
// ones at a phase boundary without a host round trip.
void launch_fill(double* d_out, std::int32_t n, double value, cudaStream_t stream);

// An empty kernel, for measuring this platform's launch and
// launch-plus-synchronize latency. That latency is the quantity that
// decides whether GPU pricing can pay for itself on small models, so the
// benchmark measures it rather than citing a vendor figure.
void launch_noop(cudaStream_t stream);

// Grid size these launches will use for a given problem size, so the
// caller can allocate the per-block scratch ONCE at construction
// (prompt.md \S3.1: no allocation after solve_start).
std::int32_t pricing_grid_size(std::int32_t n);

} // namespace gpu
} // namespace sihps
