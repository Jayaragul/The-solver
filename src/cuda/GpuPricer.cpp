#include "GpuPricer.hpp"

#include "CudaCheck.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace sihps {

namespace {
// Devex reference-framework restart threshold, identical to the CPU path's
// (Simplex::devex_update). Kept in sync deliberately: the two backends must
// restart at the same point or they stop being the same algorithm.
constexpr double kDevexRestart = 1e8;

using Clock = std::chrono::steady_clock;

double elapsed(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
} // namespace

GpuPricer::GpuPricer(const CSCMatrix& a_csc, std::int32_t n_rows, std::int32_t n_struct,
                     std::int32_t n_slack, std::int32_t n_art)
    : n_rows_(n_rows),
      n_struct_(n_struct),
      n_slack_(n_slack),
      n_art_(n_art),
      n_total_(n_struct + n_slack + n_art),
      grid_(gpu::pricing_grid_size(n_struct + n_slack + n_art)),
      d_rc_(static_cast<std::size_t>(n_total_)),
      d_weight_(static_cast<std::size_t>(n_total_)),
      d_lower_(static_cast<std::size_t>(n_total_)),
      d_upper_(static_cast<std::size_t>(n_total_)),
      d_cost_(static_cast<std::size_t>(n_total_)),
      d_art_sign_(static_cast<std::size_t>(std::max(n_art_, 1))),
      d_spmv_zero_(0),
      d_status_(static_cast<std::size_t>(n_total_)),
      d_block_scratch_(static_cast<std::size_t>(grid_)),
      d_result_(1),
      d_block_max_(static_cast<std::size_t>(grid_)),
      d_price_retire_(1),
      d_devex_retire_(1),
      h_v_(static_cast<std::size_t>(n_rows_)),
      h_rc_(static_cast<std::size_t>(n_total_)),
      h_status_(static_cast<std::size_t>(n_total_)),
      h_result_(1),
      h_v_devex_(static_cast<std::size_t>(n_rows_)) {
    // A_csc's (col_ptr, row_idx, values) IS the CSR representation of A^T
    // (docs/architecture/SYSTEM.md \S2.5): reinterpret directly, so no
    // transpose is ever computed or stored on either side of the bus.
    if (n_struct_ > 0 && a_csc.nnz() > 0) {
        std::vector<std::int32_t> at_row_ptr(a_csc.col_ptr(), a_csc.col_ptr() + n_struct_ + 1);
        std::vector<std::int32_t> at_col_idx(a_csc.row_idx(), a_csc.row_idx() + a_csc.nnz());
        std::vector<double> at_values(a_csc.values(), a_csc.values() + a_csc.nnz());
        CSRMatrix a_transpose(n_struct_, n_rows_, std::move(at_row_ptr), std::move(at_col_idx),
                               std::move(at_values));
        at_ = std::make_unique<GpuCsrSpMV>(a_transpose, handle_, stream_);
    } else if (n_struct_ > 0) {
        // Structural columns exist but the matrix is entirely empty, so
        // A^T v is identically zero. cuSPARSE cannot describe a zero-nnz
        // matrix with null pointers, so this substitutes the (constant)
        // answer rather than special-casing every call site.
        d_spmv_zero_ = CudaBuffer<double>(static_cast<std::size_t>(n_struct_));
        SIHPS_CUDA_CHECK(cudaMemset(d_spmv_zero_.data(), 0, d_spmv_zero_.bytes()));
    }
    if (!at_) {
        // No SpMV object means no device-resident x, so the duals need a
        // home of their own for the slack/artificial tails.
        d_v_fallback_ = CudaBuffer<double>(static_cast<std::size_t>(std::max(n_rows_, 1)));
    }
    // The retirement counters are self-clearing across launches but have
    // to start at zero exactly once.
    SIHPS_CUDA_CHECK(cudaMemset(d_price_retire_.data(), 0, d_price_retire_.bytes()));
    SIHPS_CUDA_CHECK(cudaMemset(d_devex_retire_.data(), 0, d_devex_retire_.bytes()));

    reset_devex_weights();
    stream_.synchronize(); // construction-time only, not the hot path
}

double* GpuPricer::device_v() { return at_ ? at_->device_x() : d_v_fallback_.data(); }

const double* GpuPricer::device_spmv_out() {
    return at_ ? at_->device_y() : d_spmv_zero_.data();
}

void GpuPricer::sync_phase(const double* cost, const double* lower, const double* upper,
                           const double* art_sign) {
    const auto n = static_cast<std::size_t>(n_total_);
    d_cost_.copy_from_host_async(cost, n, stream_.handle());
    d_lower_.copy_from_host_async(lower, n, stream_.handle());
    d_upper_.copy_from_host_async(upper, n, stream_.handle());
    if (n_art_ > 0) {
        d_art_sign_.copy_from_host_async(art_sign, static_cast<std::size_t>(n_art_),
                                          stream_.handle());
    }
    // These sources are caller-owned pageable memory, so the copies must
    // complete before this returns -- a phase boundary, not an iteration.
    stream_.synchronize();
}

void GpuPricer::reset_devex_weights() {
    gpu::launch_fill(d_weight_.data(), n_total_, 1.0, stream_.handle());
}

void GpuPricer::upload_status(const std::uint8_t* status, std::uint8_t* staging) {
    std::copy(status, status + n_total_, staging);
    d_status_.copy_from_host_async(staging, static_cast<std::size_t>(n_total_), stream_.handle());
}

void GpuPricer::spmv_from_host_vector(const double* v, double* staging) {
    std::copy(v, v + n_rows_, staging);
    submit_spmv_from(staging);
}

// Queues the H2D of an ALREADY-STAGED vector plus the SpMV against it.
// Split out from spmv_from_host_vector so the profile can charge the host
// memcpy and the submission to different buckets -- they are different
// costs with different fixes, and lumping them together is how "overhead"
// becomes unactionable.
void GpuPricer::submit_spmv_from(const double* staged) {
    if (at_) {
        at_->upload_x_async(staged);
        at_->multiply_device_resident();
    } else {
        d_v_fallback_.copy_from_host_async(staged, static_cast<std::size_t>(n_rows_),
                                            stream_.handle());
    }
}

void GpuPricer::submit_spmv() { submit_spmv_from(h_v_.data()); }

gpu::PricingCandidate GpuPricer::price_and_select(const double* y, const std::uint8_t* status,
                                                   bool devex, double opt_tol) {
    const auto t_stage = Clock::now();
    std::copy(status, status + n_total_, h_status_.data());
    std::copy(y, y + n_rows_, h_v_.data());
    profile_.stage_seconds += elapsed(t_stage);

    const auto t_submit = Clock::now();
    d_status_.copy_from_host_async(h_status_.data(), static_cast<std::size_t>(n_total_),
                                    stream_.handle());
    submit_spmv();

    // d = c - [A^T y ; y ; sign .* y], eligibility, score and argmax --
    // all in one kernel, because every reduced cost is read exactly once.
    gpu::launch_price_argmax(device_spmv_out(), device_v(), d_cost_.data(), d_art_sign_.data(),
                              d_weight_.data(), d_status_.data(), d_lower_.data(), d_upper_.data(),
                              n_struct_, n_slack_, n_art_, devex, opt_tol,
                              d_block_scratch_.data(), grid_, d_price_retire_.data(),
                              d_result_.data(), stream_.handle());

    d_result_.copy_to_host_async(h_result_.data(), 1, stream_.handle());
    done_.record(stream_.handle());
    profile_.submit_seconds += elapsed(t_submit);

    const auto t_wait = Clock::now();
    done_.synchronize();
    profile_.wait_seconds += elapsed(t_wait);
    ++profile_.calls;

    return *h_result_.data();
}

void GpuPricer::price_to_host(const double* y, double* rc) {
    spmv_from_host_vector(y, h_v_.data());
    gpu::launch_assemble_augmented(device_spmv_out(), device_v(), d_cost_.data(),
                                    d_art_sign_.data(), d_rc_.data(), n_struct_, n_slack_, n_art_,
                                    1.0, -1.0, stream_.handle());
    d_rc_.copy_to_host_async(h_rc_.data(), static_cast<std::size_t>(n_total_), stream_.handle());
    done_.record(stream_.handle());
    done_.synchronize();
    std::copy(h_rc_.data(), h_rc_.data() + n_total_, rc);
}

void GpuPricer::devex_update(const double* binv_row, std::int32_t entering,
                             std::int32_t leaving_var, double pivot) {
    const double pivot_sq = pivot * pivot;
    if (!(pivot_sq > 0.0)) return; // degenerate pivot; leave weights untouched (matches CPU)

    const auto t_stage = Clock::now();
    std::copy(binv_row, binv_row + n_rows_, h_v_devex_.data());
    profile_.devex_stage_seconds += elapsed(t_stage);

    const auto t_submit = Clock::now();
    // No status transfer: the kernel reuses pricing's upload from earlier
    // this iteration and takes the two entries that changed as scalars.
    submit_spmv_from(h_v_devex_.data());

    // The pivot row rho = [A^T w ; w ; sign .* w] is assembled inside the
    // update kernel, which also reduces the reference maximum, installs the
    // leaving variable's weight and applies any restart -- one launch for
    // what was four.
    gpu::launch_devex_update(d_weight_.data(), device_spmv_out(), device_v(), d_art_sign_.data(),
                              d_status_.data(), n_struct_, n_slack_, n_art_, entering, leaving_var,
                              pivot_sq, kDevexRestart, d_block_max_.data(), grid_,
                              d_devex_retire_.data(), stream_.handle());
    profile_.devex_submit_seconds += elapsed(t_submit);
    ++profile_.devex_calls;
    // No synchronization: nothing on the host reads any of this. The next
    // pricing call is queued on the same stream and is therefore already
    // ordered after it (prompt.md \S3.6).
}

void GpuPricer::download_devex_weights(double* out) {
    d_weight_.copy_to_host_async(h_rc_.data(), static_cast<std::size_t>(n_total_),
                                  stream_.handle());
    done_.record(stream_.handle());
    done_.synchronize();
    std::copy(h_rc_.data(), h_rc_.data() + n_total_, out);
}

} // namespace sihps
