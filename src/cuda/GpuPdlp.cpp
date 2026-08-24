#include "GpuPdlp.hpp"

#include "CudaCheck.hpp"
#include "../sparse/Convert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace sihps {

namespace {

using Clock = std::chrono::steady_clock;

// PDHG converges when tau * sigma * ||A||^2 < 1. The step size is set to
// kStepSafety / ||A||_2, and the norm itself comes from a power iteration
// that CONVERGES FROM BELOW -- so an under-estimate is the expected error,
// and an under-estimated norm means an over-sized step and divergence.
// 0.9 is the usual choice in the PDLP literature and leaves room for both
// the truncated power iteration and the fact that ||A||_2 is only estimated
// to a few digits.
constexpr double kStepSafety = 0.9;

// Bounds on the primal weight, which otherwise drifts without limit on
// badly balanced models and turns one of the two step sizes into a no-op.
constexpr double kMinPrimalWeight = 1e-6;
constexpr double kMaxPrimalWeight = 1e6;

// How much of the new primal-weight estimate to adopt per restart. PDLP
// uses a logarithmic smoothing; abrupt weight changes discard the progress
// the current scaling has already made.
constexpr double kPrimalWeightSmoothing = 0.5;

// Restart thresholds (Applegate et al. 2021, 5).
//
// RESTARTING IS NOT THE SAME AS CHECKING. An earlier version of this file
// restarted unconditionally at every convergence check, and the result was
// not merely suboptimal -- it broke the method. Two things went wrong at
// once: the ergodic average, which is the iterate PDHG actually converges
// on, was discarded every 64 iterations before it could accumulate; and the
// primal weight was re-estimated from movements measured over those same 64
// iterations, which is noise, so it random-walked and dragged the step-size
// balance with it. MEASURED: 11 of 14 small Netlib instances hit the time
// limit, several with KKT errors above 1e+5 -- divergence, not slow
// progress. With the criteria below, checks stay frequent (they are cheap
// and they are what bounds the sync count) while restarts happen only when
// the merit function says a restart has been earned.
constexpr double kSufficientDecay = 0.2;  // clear progress: restart and bank it
constexpr double kNecessaryDecay = 0.8;   // some progress, but stalling: restart
constexpr double kArtificialFraction = 0.36; // no restart in far too long

// TRIED AND REJECTED: a window-granularity adaptive step size.
//
// eta = 0.9 / ||A||_2 is a GLOBAL bound and is usually far more
// conservative than the local curvature requires -- which is why PDLP's
// adaptive rule (Applegate et al. 2021, 3.1) is worth several times the
// iteration count. That rule needs an accept/reject decision every
// iteration, and a host-visible decision per iteration is exactly what
// makes GPU simplex uncompetitive (docs/architecture/CPU_GPU.md 4), so
// adopting it as written would trade away the reason this solver is on the
// GPU at all.
//
// The obvious compromise -- adapt once per check window, growing eta by
// 1.25x while the merit function falls and halving it when it rises, capped
// at 8x the global bound -- was implemented and MEASURED. It destroyed the
// solver. Across 14 mid-size Netlib instances every single one failed,
// with KKT errors reaching 1e+50 and objectives reaching 1e+68; the
// pre-change code solved most of them in seconds.
//
// The reason is not tuning, so no choice of growth factor rescues it. PDHG
// diverges GEOMETRICALLY once tau*sigma*||A||^2 >= 1, and a window is
// 64-256 iterations long. By the time the merit function has been observed
// to rise, the iterate is already astronomically far away and halving eta
// recovers nothing. Applegate's rule works precisely because it rejects the
// single step that violated the bound, before the damage compounds.
//
// So the conclusion is a real constraint rather than a missing feature: on
// this design, the step size cannot exceed the global bound without
// per-iteration backtracking, and per-iteration backtracking costs the
// sync-free inner loop. eta stays at the safe value.

double finite_norm(const std::vector<double>& v) {
    double acc = 0.0;
    for (double e : v) {
        if (std::isfinite(e)) acc += e * e;
    }
    return std::sqrt(acc);
}

CSRMatrix transpose_csr(const CSRMatrix& a) {
    // CSC's (col_ptr, row_idx, values) IS the CSR of A^T
    // (docs/architecture/SYSTEM.md 2.5) -- no transpose kernel, no
    // duplicated conversion logic.
    const CSCMatrix csc = csr_to_csc(a);
    std::vector<std::int32_t> row_ptr(csc.col_ptr(), csc.col_ptr() + a.cols() + 1);
    std::vector<std::int32_t> col_idx(csc.row_idx(), csc.row_idx() + csc.nnz());
    std::vector<double> values(csc.values(), csc.values() + csc.nnz());
    return CSRMatrix(a.cols(), a.rows(), std::move(row_ptr), std::move(col_idx), std::move(values));
}

} // namespace

GpuPdlp::GpuPdlp(const CSRMatrix& a, const std::vector<double>& cost,
                 const std::vector<double>& lower, const std::vector<double>& upper,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper)
    : n_(a.cols()),
      m_(a.rows()),
      grid_n_(gpu::pdlp_grid_size(a.cols())),
      grid_m_(gpu::pdlp_grid_size(a.rows())),
      grid_max_(gpu::pdlp_grid_size(a.cols() > a.rows() ? a.cols() : a.rows())),
      d_x_(static_cast<std::size_t>(n_)),
      d_x_bar_(static_cast<std::size_t>(n_)),
      d_x_sum_(static_cast<std::size_t>(n_)),
      d_x_restart_(static_cast<std::size_t>(n_)),
      d_x_avg_(static_cast<std::size_t>(n_)),
      d_aty_(static_cast<std::size_t>(n_)),
      d_y_(static_cast<std::size_t>(m_)),
      d_y_sum_(static_cast<std::size_t>(m_)),
      d_y_restart_(static_cast<std::size_t>(m_)),
      d_y_avg_(static_cast<std::size_t>(m_)),
      d_ax_bar_(static_cast<std::size_t>(m_)),
      d_ax_(static_cast<std::size_t>(m_)),
      d_x_cand_(static_cast<std::size_t>(n_)),
      d_y_cand_(static_cast<std::size_t>(m_)),
      d_ax_cand_(static_cast<std::size_t>(m_)),
      d_ax_prev_(static_cast<std::size_t>(m_)),
      d_step_(1u),
      h_step_(1u),
      d_cost_(static_cast<std::size_t>(n_)),
      d_lower_(static_cast<std::size_t>(n_)),
      d_upper_(static_cast<std::size_t>(n_)),
      d_row_lower_(static_cast<std::size_t>(m_)),
      d_row_upper_(static_cast<std::size_t>(m_)),
      d_pow_v_(static_cast<std::size_t>(n_)),
      d_pow_w_(static_cast<std::size_t>(m_)),
      d_pow_v2_(static_cast<std::size_t>(n_)),
      // Eight reduction slots per block, so one launch produces the whole
      // PdlpResiduals rather than eight.
      d_block_scratch_(static_cast<std::size_t>(grid_max_) * 8u),
      d_scalar_(1),
      d_retire_(1),
      d_residuals_(1),
      h_residuals_(1),
      h_scalar_(1),
      h_vec_n_(static_cast<std::size_t>(n_)),
      h_vec_m_(static_cast<std::size_t>(m_)) {
    a_ = std::make_unique<GpuCsrSpMV>(a, handle_, stream_);
    at_ = std::make_unique<GpuCsrSpMV>(transpose_csr(a), handle_, stream_);

    const auto upload_n = [&](CudaBuffer<double>& dst, const std::vector<double>& src) {
        std::copy(src.begin(), src.end(), h_vec_n_.data());
        dst.copy_from_host_async(h_vec_n_.data(), static_cast<std::size_t>(n_), stream_.handle());
        stream_.synchronize(); // setup only
    };
    const auto upload_m = [&](CudaBuffer<double>& dst, const std::vector<double>& src) {
        std::copy(src.begin(), src.end(), h_vec_m_.data());
        dst.copy_from_host_async(h_vec_m_.data(), static_cast<std::size_t>(m_), stream_.handle());
        stream_.synchronize();
    };

    upload_n(d_cost_, cost);
    upload_n(d_lower_, lower);
    upload_n(d_upper_, upper);
    upload_m(d_row_lower_, row_lower);
    upload_m(d_row_upper_, row_upper);

    SIHPS_CUDA_CHECK(cudaMemset(d_retire_.data(), 0, d_retire_.bytes()));

    // Relative termination measures are normalized by these; computed on
    // finite entries only, since an infinite bound describes an absent
    // constraint rather than a large one.
    cost_norm_ = finite_norm(cost);
    bound_norm_ = std::sqrt(finite_norm(row_lower) * finite_norm(row_lower) +
                             finite_norm(row_upper) * finite_norm(row_upper));
}

double GpuPdlp::estimate_matrix_norm(std::int32_t iterations) {
    // Power iteration on A^T A. Every step is two SpMVs, a norm and a
    // normalize -- all queued, with ONE synchronize at the very end to read
    // the eigenvalue estimate. Doing it the obvious way (sync each step to
    // check convergence) would cost more than the whole estimate is worth.
    gpu::launch_pdlp_fill(d_pow_v_.data(), 1.0, n_, stream_.handle());
    gpu::launch_pdlp_norm_sq(d_pow_v_.data(), n_, d_block_scratch_.data(), grid_max_,
                              d_retire_.data(), d_scalar_.data(), stream_.handle());
    gpu::launch_pdlp_normalize(d_pow_v_.data(), d_scalar_.data(), n_, stream_.handle());

    a_->set_vectors(d_pow_v_.data(), d_pow_w_.data());
    at_->set_vectors(d_pow_w_.data(), d_pow_v2_.data());

    for (std::int32_t k = 0; k < iterations; ++k) {
        a_->multiply_device_resident();  // w  = A v
        at_->multiply_device_resident(); // v2 = A^T w
        gpu::launch_pdlp_norm_sq(d_pow_v2_.data(), n_, d_block_scratch_.data(), grid_max_,
                                  d_retire_.data(), d_scalar_.data(), stream_.handle());
        gpu::launch_pdlp_copy(d_pow_v_.data(), d_pow_v2_.data(), n_, stream_.handle());
        gpu::launch_pdlp_normalize(d_pow_v_.data(), d_scalar_.data(), n_, stream_.handle());
    }

    d_scalar_.copy_to_host_async(h_scalar_.data(), 1, stream_.handle());
    done_.record(stream_.handle());
    done_.synchronize();
    ++sync_count_;

    // With ||v|| = 1, ||A^T A v|| converges to the largest eigenvalue of
    // A^T A, whose square root is ||A||_2.
    const double lambda = std::sqrt(std::max(0.0, *h_scalar_.data()));
    const double norm = std::sqrt(lambda);
    return (norm > 0.0 && std::isfinite(norm)) ? norm : 1.0;
}

void GpuPdlp::queue_residual_evaluation(double* d_x, double* d_y, double* d_ax_scratch) {
    a_->set_vectors(d_x, d_ax_scratch);
    a_->multiply_device_resident();
    at_->set_vectors(d_y, d_aty_.data());
    at_->multiply_device_resident();

    gpu::launch_pdlp_residuals(d_x, d_y, d_ax_scratch, d_aty_.data(), d_cost_.data(),
                                d_lower_.data(), d_upper_.data(), d_row_lower_.data(),
                                d_row_upper_.data(), d_x_restart_.data(), d_y_restart_.data(), n_,
                                m_, d_block_scratch_.data(), grid_max_, d_retire_.data(),
                                d_residuals_.data(), stream_.handle());
}

GpuPdlp::Kkt GpuPdlp::read_residuals() {
    d_residuals_.copy_to_host_async(h_residuals_.data(), 1, stream_.handle());
    done_.record(stream_.handle());
    done_.synchronize();
    ++sync_count_;

    const gpu::PdlpResiduals& r = *h_residuals_.data();
    Kkt k;
    k.primal = std::sqrt(std::max(0.0, r.primal_residual_sq)) / (1.0 + bound_norm_);
    k.dual = std::sqrt(std::max(0.0, r.dual_residual_sq)) / (1.0 + cost_norm_);
    k.primal_obj = r.primal_objective;
    k.dual_obj = r.dual_objective;
    k.gap = std::fabs(r.primal_objective - r.dual_objective) /
             (1.0 + std::fabs(r.primal_objective) + std::fabs(r.dual_objective));
    k.primal_move = std::sqrt(std::max(0.0, r.primal_move_sq));
    k.dual_move = std::sqrt(std::max(0.0, r.dual_move_sq));

    // A NaN anywhere means the iteration diverged. Report it as "not
    // converged" rather than letting it propagate into a comparison that
    // silently evaluates false and looks like slow progress.
    if (!std::isfinite(k.primal) || !std::isfinite(k.dual) || !std::isfinite(k.gap)) {
        k.primal = k.dual = k.gap = std::numeric_limits<double>::infinity();
    }
    return k;
}

PdlpStats GpuPdlp::solve(const PdlpParams& params, std::vector<double>& x_out,
                         std::vector<double>& y_out) {
    const auto t_start = Clock::now();
    PdlpStats stats;
    sync_count_ = 0;

    const double matrix_norm = estimate_matrix_norm(params.power_iterations);
    stats.matrix_norm = matrix_norm;

    // The GLOBAL bound. Under the fixed-step path this is the step size for
    // the whole solve; under the adaptive path it is only the starting
    // point, and the local bound measured each iteration routinely exceeds
    // it by a large factor (PdlpKernels.cuh).
    const double eta_global = kStepSafety / matrix_norm;
    const double eta_min = params.eta_min_factor / matrix_norm;
    const double eta_max = params.eta_max_factor / matrix_norm;

    // Primal weight balances the two step sizes when the objective and the
    // constraint right-hand sides live on different scales (Applegate et
    // al. 2021, 3.3). Starting from the data's own ratio beats starting
    // from 1 on essentially every real model.
    double omega = 1.0;
    if (cost_norm_ > 0.0 && bound_norm_ > 0.0) {
        omega = std::min(kMaxPrimalWeight, std::max(kMinPrimalWeight, cost_norm_ / bound_norm_));
    }

    gpu::launch_pdlp_fill(d_x_.data(), 0.0, n_, stream_.handle());
    gpu::launch_pdlp_fill(d_x_bar_.data(), 0.0, n_, stream_.handle());
    gpu::launch_pdlp_fill(d_x_sum_.data(), 0.0, n_, stream_.handle());
    gpu::launch_pdlp_fill(d_x_restart_.data(), 0.0, n_, stream_.handle());
    gpu::launch_pdlp_fill(d_y_.data(), 0.0, m_, stream_.handle());
    gpu::launch_pdlp_fill(d_y_sum_.data(), 0.0, m_, stream_.handle());
    gpu::launch_pdlp_fill(d_y_restart_.data(), 0.0, m_, stream_.handle());

    // The adaptive path tracks A x rather than A xbar, so both buffers must
    // start consistent with x = 0: A 0 = 0, and the previous product is the
    // same, which makes the first extrapolation A xbar = 2*0 - 0 = 0.
    gpu::launch_pdlp_fill(d_ax_.data(), 0.0, m_, stream_.handle());
    gpu::launch_pdlp_fill(d_ax_prev_.data(), 0.0, m_, stream_.handle());
    gpu::launch_pdlp_init_step_state(d_step_.data(), eta_global, stream_.handle());

    // Fixed bindings for the adaptive path: A maps the candidate x to its
    // product, A^T maps the candidate y back. Neither is rebound in the
    // loop, so cuSPARSE's preprocessed plan stays valid throughout.
    if (params.adaptive_step) {
        a_->set_vectors(d_x_cand_.data(), d_ax_cand_.data());
        at_->set_vectors(d_y_cand_.data(), d_aty_.data());
    }

    double accepted_total = 0.0;    // accepted steps at the last sync
    double accepted_at_restart = 0.0; // ... at the last restart

    std::int32_t iteration = 0;
    std::int32_t windows_since_restart = 0;
    std::int32_t iterations_since_restart = 0;
    double mu_at_restart = std::numeric_limits<double>::infinity();
    double mu_previous = std::numeric_limits<double>::infinity();
    Kkt best;

    while (iteration < params.max_iterations) {
        const std::int32_t period =
            std::min(params.restart_period, params.max_iterations - iteration);
        if (period <= 0) break;

        // ---- the inner loop: NO host synchronization anywhere in here ----
        // Four queued operations per iteration, `period` iterations deep.
        // This is the whole reason the GPU can win here; see GpuPdlp.hpp.
        if (params.adaptive_step) {
            // Six queued operations per iteration, none of them a
            // synchronization. The step size lives in d_step_ and is read
            // by the two update kernels straight out of device memory, so
            // the accept/reject decision never becomes host-visible. A
            // rejected step writes nothing and the next pass through this
            // loop is its retry -- backtracking with no control flow.
            for (std::int32_t k = 0; k < period; ++k) {
                gpu::launch_pdlp_dual_update_adaptive(
                    d_y_cand_.data(), d_y_.data(), d_ax_.data(), d_ax_prev_.data(),
                    d_row_lower_.data(), d_row_upper_.data(), d_step_.data(), omega, m_,
                    stream_.handle());
                at_->multiply_device_resident(); // aty = A^T y_cand
                gpu::launch_pdlp_primal_update_adaptive(
                    d_x_cand_.data(), d_x_.data(), d_aty_.data(), d_cost_.data(), d_lower_.data(),
                    d_upper_.data(), d_step_.data(), omega, n_, stream_.handle());
                a_->multiply_device_resident(); // ax_cand = A x_cand
                gpu::launch_pdlp_step_probe(d_x_.data(), d_x_cand_.data(), d_y_.data(),
                                             d_y_cand_.data(), d_ax_.data(), d_ax_cand_.data(), n_,
                                             m_, omega, eta_min, eta_max, d_block_scratch_.data(),
                                             grid_max_, d_retire_.data(), d_step_.data(),
                                             stream_.handle());
                gpu::launch_pdlp_commit(d_x_.data(), d_y_.data(), d_ax_.data(), d_ax_prev_.data(),
                                         d_x_sum_.data(), d_y_sum_.data(), d_x_cand_.data(),
                                         d_y_cand_.data(), d_ax_cand_.data(), d_step_.data(), n_,
                                         m_, stream_.handle());
            }
        } else {
            const double tau = eta_global / omega;
            const double sigma = eta_global * omega;
            a_->set_vectors(d_x_bar_.data(), d_ax_bar_.data());
            at_->set_vectors(d_y_.data(), d_aty_.data());
            for (std::int32_t k = 0; k < period; ++k) {
                a_->multiply_device_resident(); // ax_bar = A xbar
                gpu::launch_pdlp_dual_update(d_y_.data(), d_ax_bar_.data(), d_row_lower_.data(),
                                              d_row_upper_.data(), sigma, m_, d_y_sum_.data(), 1.0,
                                              stream_.handle());
                at_->multiply_device_resident(); // aty = A^T y
                gpu::launch_pdlp_primal_update(d_x_.data(), d_x_bar_.data(), d_aty_.data(),
                                                d_cost_.data(), d_lower_.data(), d_upper_.data(),
                                                tau, n_, d_x_sum_.data(), 1.0, stream_.handle());
            }
        }
        iteration += period;
        iterations_since_restart += period;
        ++windows_since_restart;

        // ---- one evaluation, one synchronize ----
        queue_residual_evaluation(d_x_.data(), d_y_.data(), d_ax_.data());
        const Kkt current = read_residuals();

        // The running average is what PDLP's restart scheme is built on:
        // PDHG's ergodic average converges where the raw iterate can cycle,
        // so both are evaluated and the better one is kept.
        //
        // Under the adaptive step the divisor is NOT the number of queued
        // iterations: rejected steps contribute nothing to the sums, so
        // dividing by the queued count would shrink the average toward zero
        // in proportion to the rejection rate. The accepted count is read
        // from the device state at this same synchronize -- it costs no
        // extra round trip because the residual read has already stalled.
        double terms;
        if (params.adaptive_step) {
            d_step_.copy_to_host_async(h_step_.data(), 1u, stream_.handle());
            SIHPS_CUDA_CHECK(cudaStreamSynchronize(stream_.handle()));
            const double total_accepted = h_step_.data()[0].accept_count;
            terms = total_accepted - accepted_at_restart;
            accepted_total = total_accepted;
            if (terms < 1.0) terms = 1.0;
        } else {
            terms = static_cast<double>(windows_since_restart * params.restart_period);
        }
        const double scale = 1.0 / terms;
        gpu::launch_pdlp_scale_into(d_x_avg_.data(), d_x_sum_.data(), scale, n_, stream_.handle());
        gpu::launch_pdlp_scale_into(d_y_avg_.data(), d_y_sum_.data(), scale, m_, stream_.handle());
        // Scratch for the average's A x must NOT be d_ax_: under the
        // adaptive path that buffer holds A x for the live iterate and the
        // next extrapolation depends on it.
        queue_residual_evaluation(d_x_avg_.data(), d_y_avg_.data(),
                                   params.adaptive_step ? d_ax_bar_.data() : d_ax_.data());
        const Kkt average = read_residuals();

        // Both evaluations rebound cuSPARSE's vectors; restore the loop's
        // fixed bindings before the next window queues anything.
        if (params.adaptive_step) {
            a_->set_vectors(d_x_cand_.data(), d_ax_cand_.data());
            at_->set_vectors(d_y_cand_.data(), d_aty_.data());
        }

        const bool average_better = average.worst() < current.worst();
        best = average_better ? average : current;

        if (best.primal <= params.eps_optimal && best.dual <= params.eps_optimal &&
            best.gap <= params.eps_optimal) {
            if (average_better) {
                gpu::launch_pdlp_copy(d_x_.data(), d_x_avg_.data(), n_, stream_.handle());
                gpu::launch_pdlp_copy(d_y_.data(), d_y_avg_.data(), m_, stream_.handle());
            }
            stats.converged = true;
            break;
        }

        if (params.time_limit_seconds > 0.0 &&
            std::chrono::duration<double>(Clock::now() - t_start).count() >
                params.time_limit_seconds) {
            break;
        }

        // The other half of the HYBRID race has already produced a verified
        // answer, so everything this loop could still compute is discarded
        // work. Checked here because the host has just synchronized anyway.
        if (params.cancel != nullptr &&
            params.cancel->load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            break;
        }

        // ---- restart, only when it has been earned ----
        const double mu = best.worst();
        const bool sufficient = mu <= kSufficientDecay * mu_at_restart;
        const bool necessary = mu <= kNecessaryDecay * mu_at_restart && mu > mu_previous;
        const bool artificial =
            static_cast<double>(iterations_since_restart) >=
            kArtificialFraction * static_cast<double>(iteration);
        mu_previous = mu;

        if (!(sufficient || necessary || artificial)) {
            // Keep going from the current iterate WITHOUT touching the
            // average, the restart reference point or the primal weight.
            // The average is a termination/restart candidate here, not the
            // iterate being advanced.
            continue;
        }

        if (average_better) {
            gpu::launch_pdlp_copy(d_x_.data(), d_x_avg_.data(), n_, stream_.handle());
            gpu::launch_pdlp_copy(d_y_.data(), d_y_avg_.data(), m_, stream_.handle());
            gpu::launch_pdlp_copy(d_x_bar_.data(), d_x_avg_.data(), n_, stream_.handle());
            // A x for the restarted iterate was just computed by the
            // average's residual evaluation, so restarting costs no SpMV.
            if (params.adaptive_step) {
                gpu::launch_pdlp_copy(d_ax_.data(), d_ax_bar_.data(), m_, stream_.handle());
            }
        } else {
            gpu::launch_pdlp_copy(d_x_bar_.data(), d_x_.data(), n_, stream_.handle());
        }
        // After a restart the extrapolation must be the identity, which
        // means x^{k-1} = x^k, i.e. A x_prev = A x. Skipping this restarts
        // from a stale extrapolation and undoes the restart's benefit.
        if (params.adaptive_step) {
            gpu::launch_pdlp_copy(d_ax_prev_.data(), d_ax_.data(), m_, stream_.handle());
            accepted_at_restart = accepted_total;
        }
        gpu::launch_pdlp_fill(d_x_sum_.data(), 0.0, n_, stream_.handle());
        gpu::launch_pdlp_fill(d_y_sum_.data(), 0.0, m_, stream_.handle());
        windows_since_restart = 0;
        iterations_since_restart = 0;
        mu_at_restart = mu;
        mu_previous = std::numeric_limits<double>::infinity();
        ++stats.restarts;

        // Re-balance the step sizes from how far each space moved between
        // this restart and the last. Measured over a whole restart interval
        // rather than a single check window -- over 64 iterations the
        // movements are noise, and feeding noise into a multiplicative
        // update is what made the earlier version diverge.
        if (best.primal_move > 1e-14 && best.dual_move > 1e-14) {
            const double target = best.dual_move / best.primal_move;
            const double updated =
                std::exp(kPrimalWeightSmoothing * std::log(target) +
                          (1.0 - kPrimalWeightSmoothing) * std::log(omega));
            if (std::isfinite(updated)) {
                omega = std::min(kMaxPrimalWeight, std::max(kMinPrimalWeight, updated));
            }
        }

        // The movement reference advances only at a restart, so
        // primal_move/dual_move measure a full restart interval.
        gpu::launch_pdlp_copy(d_x_restart_.data(), d_x_.data(), n_, stream_.handle());
        gpu::launch_pdlp_copy(d_y_restart_.data(), d_y_.data(), m_, stream_.handle());
    }

    // Final read-back of whichever iterate the loop settled on.
    x_out.assign(static_cast<std::size_t>(n_), 0.0);
    y_out.assign(static_cast<std::size_t>(m_), 0.0);
    d_x_.copy_to_host_async(h_vec_n_.data(), static_cast<std::size_t>(n_), stream_.handle());
    d_y_.copy_to_host_async(h_vec_m_.data(), static_cast<std::size_t>(m_), stream_.handle());
    done_.record(stream_.handle());
    done_.synchronize();
    ++sync_count_;
    std::copy(h_vec_n_.data(), h_vec_n_.data() + n_, x_out.begin());
    std::copy(h_vec_m_.data(), h_vec_m_.data() + m_, y_out.begin());

    stats.iterations = iteration;
    stats.primal_objective = best.primal_obj;
    stats.dual_objective = best.dual_obj;
    stats.relative_primal_residual = best.primal;
    stats.relative_dual_residual = best.dual;
    stats.relative_gap = best.gap;
    stats.host_syncs = sync_count_;
    if (params.adaptive_step) {
        d_step_.copy_to_host_async(h_step_.data(), 1u, stream_.handle());
        SIHPS_CUDA_CHECK(cudaStreamSynchronize(stream_.handle()));
        const gpu::PdlpStepState& st = h_step_.data()[0];
        stats.final_step_size = st.eta;
        stats.step_size_ratio = st.eta * matrix_norm; // relative to 1/||A||_2
        const double attempted = st.updates;
        const double rejected = attempted - st.accept_count;
        stats.rejected_steps = static_cast<std::int32_t>(rejected > 0.0 ? rejected : 0.0);
    } else {
        stats.final_step_size = eta_global;
        stats.step_size_ratio = kStepSafety;
    }
    stats.seconds = std::chrono::duration<double>(Clock::now() - t_start).count();
    return stats;
}

} // namespace sihps
