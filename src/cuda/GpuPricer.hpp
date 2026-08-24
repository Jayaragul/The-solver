#pragma once

#include "CudaBuffer.hpp"
#include "CudaEvent.hpp"
#include "CudaStream.hpp"
#include "CusparseHandle.hpp"
#include "GpuSpMV.hpp"
#include "PinnedBuffer.hpp"
#include "PricingKernels.cuh"
#include "../sparse/CSCMatrix.hpp"

#include <cstdint>
#include <memory>

namespace sihps {

// Host-side cost breakdown of the GPU pricing path.
//
// "GPU pricing is slower because of overhead" is not a diagnosis anyone can
// act on -- overhead of WHAT? These counters split each call into the three
// things the host actually does, so the answer is a measurement rather than
// an argument:
//
//   stage   -- copying caller data into the pinned staging buffers
//   submit  -- queueing the transfers, the cuSPARSE SpMV and the kernels
//   wait    -- blocking on the event that signals the result has landed
//
// Anything not in these three is GPU execution genuinely overlapped with
// nothing, which is itself the finding if submit + wait turns out to
// account for the whole call.
//
// The instrumentation is always compiled in: two steady_clock reads cost
// tens of nanoseconds against a call that measures in the hundreds of
// microseconds, so gating it behind a build flag would cost more in
// divergence between the measured and shipped code than it saves.
struct GpuPricerProfile {
    double stage_seconds = 0.0;
    double submit_seconds = 0.0;
    double wait_seconds = 0.0;
    double devex_stage_seconds = 0.0;
    double devex_submit_seconds = 0.0;
    long calls = 0;
    long devex_calls = 0;

    double total_seconds() const {
        return stage_seconds + submit_seconds + wait_seconds + devex_stage_seconds +
               devex_submit_seconds;
    }
};

// Device-resident pricing state for the revised simplex: the reduced-cost
// vector, the Devex reference weights, the augmented-space bounds/costs,
// and the per-block reduction scratch all live on the GPU across
// iterations and are allocated exactly once, here, at construction
// (prompt.md \S3.1 -- no cudaMalloc after solve_start).
//
// WHY THIS TYPE EXISTS
// --------------------
// The naive GPU pricing placement -- run cusparseSpMV for A^T y, copy the
// result back, and do the reduced-cost assembly and entering-variable
// search on the host -- moves the cheapest part of pricing to the device
// and pays an O(n) PCIe round trip for the privilege. GpuPricer keeps the
// entire pricing chain resident:
//
//     host duals y  --(pinned, async H2D, m doubles)-->
//     cusparseSpMV  ->  A^T y                        (library, \S3.5)
//     assemble      ->  d = c - [A^T y ; y ; s.*y]   (custom kernel)
//     argmax        ->  one PricingCandidate         (custom kernel)
//                   --(async D2H, 24 bytes)-->  host
//
// so the return trip is O(1) in the number of columns rather than O(n),
// and the host's per-iteration pricing loop disappears entirely. The Devex
// weights never leave the device at all: their update also consumes an
// A^T-times-vector product (the pivot row is A^T B^-T e_r), so it reuses
// the same SpMV object and the same assembly kernel with different
// coefficients.
//
// Whether this is a net win against a well-optimized CPU pricing loop is
// deliberately NOT asserted here -- it is measured by
// benchmarks/bench_pricing_backend.cpp, which is the evidence
// docs/research/SOTA.md's H1/H5 hypotheses call for.
class GpuPricer {
public:
    // `a_csc` is the scaled constraint matrix in CSC, whose (col_ptr,
    // row_idx, values) arrays ARE the CSR representation of A^T -- so no
    // transpose is computed or stored.
    GpuPricer(const CSCMatrix& a_csc, std::int32_t n_rows, std::int32_t n_struct,
              std::int32_t n_slack, std::int32_t n_art);

    // Uploads the quantities that change only at a phase boundary (phase 1
    // installs its own costs and re-opens the artificial bounds). Cheap and
    // rare; deliberately separate from the per-iteration path so the hot
    // path never re-sends them.
    void sync_phase(const double* cost, const double* lower, const double* upper,
                    const double* art_sign);

    // Restarts the Devex reference framework device-side (all weights to
    // one), without a host round trip.
    void reset_devex_weights();

    // FAST PATH. Computes reduced costs for every augmented column and
    // returns the winning entering candidate. `status` is the host status
    // array (one byte per augmented column); it changes every iteration
    // and is the only O(n) host-to-device traffic here, at one byte per
    // column against the eight bytes per column a reduced-cost readback
    // would cost.
    //
    // Synchronizes on an event tied to the 24-byte result copy -- the host
    // genuinely cannot choose a pivot without this value, so the wait is
    // necessary rather than incidental (prompt.md \S3.6 forbids the
    // gratuitous kind, not this one).
    gpu::PricingCandidate price_and_select(const double* y, const std::uint8_t* status, bool devex,
                                            double opt_tol);

    // COMPATIBILITY PATH. Full reduced-cost vector back to the host, for
    // the two callers that genuinely need all of it: the dual simplex
    // (which scans reduced costs in its own ratio test) and the final
    // dual-residual verification. Costs the O(n) D2H that the fast path
    // exists to avoid, and is not on the primal iteration path.
    void price_to_host(const double* y, double* rc);

    // Devex weight update against device-resident weights. `binv_row` is
    // the host-computed row of B^-1 (a BTRAN result); the pivot row it
    // implies is formed on the device by the same SpMV + assembly pair
    // pricing uses, so the O(n) pivot row is never materialized on the
    // host at all.
    //
    // Fully asynchronous: no host synchronization anywhere in this call.
    // `status` is no longer transferred here -- see launch_devex_update's
    // note in PricingKernels.cuh. The parameter is gone rather than
    // ignored, so no caller can believe it is being honoured.
    void devex_update(const double* binv_row, std::int32_t entering, std::int32_t leaving_var,
                      double pivot);

    // Copies the device-resident Devex weights back to the host. Test and
    // diagnostic use only -- never called from the iteration loop.
    void download_devex_weights(double* out);

    const GpuPricerProfile& profile() const noexcept { return profile_; }
    void reset_profile() noexcept { profile_ = GpuPricerProfile{}; }

private:
    // spmv_from_host_vector takes the pinned staging buffer to route
    // through. THE BUFFER CHOICE IS A CORRECTNESS REQUIREMENT, not a
    // tuning knob.
    //
    // devex_update() deliberately queues its transfers and returns without
    // synchronizing -- that asynchrony is the point of doing the weight
    // update on the device. But it means the DMA out of its staging buffer
    // may still be in flight when the host returns to the iteration loop.
    // If the next price_and_select() then wrote its own inputs into that
    // same buffer, it would be overwriting memory the copy engine is still
    // reading, and the device would price against a mixture of two
    // iterations' data. MEASURED before this was separated: an instance
    // that otherwise verifies cleanly flipped between OPTIMAL and
    // NUMERICAL_FAILURE from run to run.
    //
    // Giving pricing and the Devex update their own staging makes the
    // hazard impossible rather than unlikely: every Devex copy is followed
    // by price_and_select()'s stream synchronize before that buffer can be
    // touched again, and every pricing copy is followed by the synchronize
    // in its own call.
    void upload_status(const std::uint8_t* status, std::uint8_t* staging);
    void spmv_from_host_vector(const double* v, double* staging);
    void submit_spmv_from(const double* staged);
    void submit_spmv();

    // The device-resident vector the slack/artificial tails read, and the
    // A^T-product the structural head reads. Both have a degenerate form
    // when the model has no structural nonzeros for cuSPARSE to describe.
    double* device_v();
    const double* device_spmv_out();

    std::int32_t n_rows_, n_struct_, n_slack_, n_art_, n_total_;
    std::int32_t grid_;

    CusparseHandle handle_;
    CudaStream stream_;
    CudaEvent done_;
    std::unique_ptr<GpuCsrSpMV> at_; // A^T, or null when the model has no structural nonzeros

    // d_rc_ backs price_to_host only: the fast path never materializes a
    // reduced-cost vector, on the device or anywhere else.
    CudaBuffer<double> d_rc_, d_weight_, d_lower_, d_upper_, d_cost_, d_art_sign_;
    CudaBuffer<double> d_spmv_zero_;  // stands in for A^T v when A has no nonzeros
    CudaBuffer<double> d_v_fallback_; // holds the duals when there is no SpMV object to hold them
    CudaBuffer<std::uint8_t> d_status_;
    CudaBuffer<gpu::PricingCandidate> d_block_scratch_, d_result_;
    CudaBuffer<double> d_block_max_;
    // Self-clearing block-retirement counters for the single-launch
    // reductions (PricingKernels.cuh). Zeroed once, here; never reset
    // again, because a reset would be another queued operation.
    CudaBuffer<unsigned int> d_price_retire_, d_devex_retire_;

    PinnedBuffer<double> h_v_, h_rc_;
    PinnedBuffer<std::uint8_t> h_status_;
    PinnedBuffer<gpu::PricingCandidate> h_result_;

    // Second staging set, used only by devex_update -- see the note on
    // upload_status/spmv_from_host_vector above.
    PinnedBuffer<double> h_v_devex_;

    GpuPricerProfile profile_;
};

} // namespace sihps
