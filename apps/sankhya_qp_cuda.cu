#include "sankhya.h"
#include "sankhya_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool csc_to_csr(const sk_csc& csc, std::vector<int>& offsets,
                std::vector<int>& indices, std::vector<double>& values)
{
    const int nnz = csc.p[csc.ncol];
    offsets.assign(static_cast<size_t>(csc.nrow) + 1, 0);
    indices.resize(static_cast<size_t>(nnz));
    values.resize(static_cast<size_t>(nnz));
    for (int p = 0; p < nnz; ++p) ++offsets[static_cast<size_t>(csc.i[p]) + 1];
    for (int row = 0; row < csc.nrow; ++row) offsets[static_cast<size_t>(row) + 1] += offsets[static_cast<size_t>(row)];
    std::vector<int> next(offsets.begin(), offsets.begin() + csc.nrow);
    for (int column = 0; column < csc.ncol; ++column) {
        for (int p = csc.p[column]; p < csc.p[column + 1]; ++p) {
            const int slot = next[static_cast<size_t>(csc.i[p])]++;
            indices[static_cast<size_t>(slot)] = column;
            values[static_cast<size_t>(slot)] = csc.x[p];
        }
    }
    return true;
}

/* Small GPU QPs receive the same admission discipline as the native solver:
 * a stationary first-order point is not an optimality statement for an
 * indefinite Hessian. Larger Hessians remain approximate-only. */
bool small_psd_hessian(const sk_csc& q)
{
    const int n = q.ncol;
    if (n > 512) return true;
    std::vector<double> h(static_cast<size_t>(n) * n, 0.0);
    std::vector<double> l(static_cast<size_t>(n) * n, 0.0);
    for (int col = 0; col < n; ++col) for (int p = q.p[col]; p < q.p[col + 1]; ++p) {
        if (q.i[p] < 0 || q.i[p] >= n || !std::isfinite(q.x[p])) return false;
        h[static_cast<size_t>(q.i[p]) * n + col] += q.x[p];
    }
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        const double a = h[static_cast<size_t>(i) * n + j];
        const double b = h[static_cast<size_t>(j) * n + i];
        if (std::fabs(a - b) > 1e-10 * (1.0 + std::fmax(std::fabs(a), std::fabs(b)))) return false;
        h[static_cast<size_t>(i) * n + j] = h[static_cast<size_t>(j) * n + i] = 0.5 * (a + b);
    }
    for (int i = 0; i < n; ++i) {
        double d = h[static_cast<size_t>(i) * n + i];
        for (int k = 0; k < i; ++k) d -= l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(i) * n + k];
        if (d < -1e-10 * (1.0 + std::fabs(h[static_cast<size_t>(i) * n + i]))) return false;
        l[static_cast<size_t>(i) * n + i] = d > 1e-14 ? std::sqrt(d) : 0.0;
        for (int j = i + 1; j < n; ++j) {
            double v = h[static_cast<size_t>(j) * n + i];
            for (int k = 0; k < i; ++k) v -= l[static_cast<size_t>(j) * n + k] * l[static_cast<size_t>(i) * n + k];
            if (l[static_cast<size_t>(i) * n + i] > 1e-14) l[static_cast<size_t>(j) * n + i] = v / l[static_cast<size_t>(i) * n + i];
            else if (std::fabs(v) > 1e-9) return false;
        }
    }
    return true;
}

void usage(const char* program)
{
    std::fprintf(stderr,
        "usage: %s model.qps [--iterations N] [--time-limit S] "
        "[--theta T] [--tolerance E]\n", program);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { usage(argv[0]); return 64; }
    const char* path = argv[1];
    /* This driver always passes explicit diagonal steps.  tau and sigma are
       intentionally zero: they are fallback scalars for the non-preconditioned
       APIs and must not be reported as active QP parameters here. */
    SankhyaCudaLPSettings settings{100000, 100, 0.0, 0.0, 1.0, 1e-6, 0.0};
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) { usage(argv[0]); return 64; }
        if      (!std::strcmp(argv[i], "--iterations")) settings.max_iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--time-limit")) settings.time_limit = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--theta"))      settings.theta = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tolerance"))  settings.tolerance = std::atof(argv[++i]);
        else { usage(argv[0]); return 64; }
    }

    sk_model model;
    sk_model_init(&model);
    if (sk_read_mps(path, &model) != SK_OK) {
        std::fprintf(stderr, "cannot read model: %s\n", path);
        return 1;
    }
    if (sk_model_num_integer(&model) != 0) {
        std::fprintf(stderr, "CUDA QP path accepts continuous models only\n");
        sk_model_free(&model);
        return 2;
    }

    if (model.Q == nullptr) {
        std::fprintf(stderr, "CUDA QP path requires a QPS model with a Hessian\n");
        sk_model_free(&model);
        return 2;
    }
    if (!small_psd_hessian(*model.Q)) {
        std::fprintf(stderr, "CUDA QP path rejects non-PSD or nonsymmetric small Hessians\n");
        sk_model_free(&model);
        return 2;
    }
    std::vector<double> diagonal(static_cast<size_t>(model.ncol), 0.0);
    bool diagonal_hessian = true;
    {
        for (int column = 0; column < model.ncol; ++column) {
            for (int p = model.Q->p[column]; p < model.Q->p[column + 1]; ++p) {
                if (!std::isfinite(model.Q->x[p])) {
                    std::fprintf(stderr, "CUDA QP path requires a finite Hessian\n");
                    sk_model_free(&model);
                    return 2;
                }
                if (model.Q->i[p] != column) diagonal_hessian = false;
                else if (model.Q->x[p] < 0.0) {
                    std::fprintf(stderr, "CUDA diagonal Hessian entries must be nonnegative\n");
                    sk_model_free(&model);
                    return 2;
                }
                diagonal[static_cast<size_t>(column)] += model.Q->x[p];
            }
        }
    }

    std::vector<int> offsets, indices;
    std::vector<double> values;
    csc_to_csr(model.A, offsets, indices, values);
    std::vector<double> primal_steps(static_cast<size_t>(model.ncol), 0.0);
    std::vector<double> dual_steps(static_cast<size_t>(model.nrow), 0.0);
    for (int column = 0; column < model.ncol; ++column) {
        double mass = 0.0;
        for (int p = model.A.p[column]; p < model.A.p[column + 1]; ++p) {
            const double value = std::fabs(model.A.x[p]);
            mass += value;
            dual_steps[static_cast<size_t>(model.A.i[p])] += value;
        }
        if (model.Q) for (int p = model.Q->p[column]; p < model.Q->p[column + 1]; ++p)
            mass += std::fabs(model.Q->x[p]);
        primal_steps[static_cast<size_t>(column)] = 0.9 / (1.0 + mass);
    }
    for (int row = 0; row < model.nrow; ++row)
        dual_steps[static_cast<size_t>(row)] = 0.9 / (1.0 + dual_steps[static_cast<size_t>(row)]);
    std::fprintf(stderr, "qp_step_policy=diagonal_row_column_mass step_damping=0.9\n");
    std::vector<int> q_offsets, q_indices;
    std::vector<double> q_values;
    if (!diagonal_hessian) csc_to_csr(*model.Q, q_offsets, q_indices, q_values);
    SankhyaCudaCSR matrix{};
    if (sankhya_cuda_csr_create(&matrix, model.nrow, model.ncol,
            static_cast<int>(values.size()), offsets.data(), indices.data(), values.data()) != 0) {
        std::fprintf(stderr, "CUDA CSR setup failed: %s\n", sankhya_cuda_last_error());
        sk_model_free(&model);
        return 1;
    }
    SankhyaCudaCSR hessian{};
    if (!diagonal_hessian && sankhya_cuda_csr_create(&hessian, model.ncol, model.ncol,
            static_cast<int>(q_values.size()), q_offsets.data(), q_indices.data(), q_values.data()) != 0) {
        std::fprintf(stderr, "CUDA Hessian setup failed: %s\n", sankhya_cuda_last_error());
        sankhya_cuda_csr_destroy(&matrix);
        sk_model_free(&model);
        return 1;
    }

    std::vector<double> solution(static_cast<size_t>(model.ncol), 0.0);
    SankhyaCudaLPResult gpu_result{};
    const auto solve_start = std::chrono::steady_clock::now();
    const int rc = sankhya_cuda_qp_pdhg_preconditioned(&matrix,
        diagonal_hessian ? nullptr : &hessian, diagonal_hessian ? diagonal.data() : nullptr,
        model.c, model.rlow, model.rupp, model.clow, model.cupp,
        primal_steps.data(), dual_steps.data(), settings, solution.data(), &gpu_result);
    const double solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    sk_solution checked;
    sk_solution_init(&checked);
    checked.x = solution.data();
    checked.ncol = model.ncol;
    checked.nrow = model.nrow;
    const sk_status verify_status = rc == 0 ? sk_verify(&model, &checked) : SK_ERR_NUMERIC;
    std::printf("{\"file\":\"%s\",\"status\":%d,\"iterations\":%d,\"objective\":%.12g,"
                "\"primal_inf\":%.3e,\"kkt_residual\":%.3e,\"solve_seconds\":%.6f,\"independent_primal_status\":\"%s\",\"independent_objective\":%.12g}\n",
        path, gpu_result.status, gpu_result.iterations, gpu_result.objective + model.objshift,
        gpu_result.maximum_row_violation, gpu_result.maximum_kkt_residual,
        solve_seconds, sk_status_name(verify_status), checked.objective);

    sankhya_cuda_csr_destroy(&matrix);
    if (!diagonal_hessian) sankhya_cuda_csr_destroy(&hessian);
    sk_model_free(&model);
    return rc == 0 && gpu_result.status == 0 && verify_status == SK_OK &&
        checked.primal_infeasibility <= 100.0 * settings.tolerance ? 0 : 3;
}
