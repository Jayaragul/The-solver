#include "sankhya.h"
#include "sankhya_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

double matrix_norm_bound(const sk_csc& a)
{
    std::vector<double> row_sum(static_cast<size_t>(a.nrow), 0.0);
    double col_max = 0.0;
    double row_max = 0.0;
    for (int column = 0; column < a.ncol; ++column) {
        double column_sum = 0.0;
        for (int p = a.p[column]; p < a.p[column + 1]; ++p) {
            const double value = std::fabs(a.x[p]);
            column_sum += value;
            row_sum[static_cast<size_t>(a.i[p])] += value;
        }
        if (column_sum > col_max) col_max = column_sum;
    }
    for (double value : row_sum) if (value > row_max) row_max = value;
    return std::sqrt(col_max * row_max);
}

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

void usage(const char* program)
{
    std::fprintf(stderr,
        "usage: %s model.qps [--iterations N] [--tau T] [--sigma S] "
        "[--theta T] [--tolerance E]\n", program);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { usage(argv[0]); return 64; }
    const char* path = argv[1];
    SankhyaCudaLPSettings settings{100000, 100, 0.0, 0.0, 1.0, 1e-6};
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) { usage(argv[0]); return 64; }
        if      (!std::strcmp(argv[i], "--iterations")) settings.max_iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tau"))        settings.tau = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--sigma"))      settings.sigma = std::atof(argv[++i]);
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

    if (settings.tau == 0.0 && settings.sigma == 0.0) {
        const double norm = matrix_norm_bound(model.A);
        const double step = norm > 0.0 ? 0.5 / norm : 1.0;
        /* Keep tau*sigma fixed for the A coupling, but give constrained
           sparse-Hessian QPs more dual progress.  Explicit Q*x curvature is
           handled in the primal step, so an equal scalar pair is needlessly
           restrictive on the benchmark family. */
        settings.tau = diagonal_hessian ? step : 0.5 * step;
        settings.sigma = diagonal_hessian ? step : 2.0 * step;
        std::fprintf(stderr, "auto_operator_norm=%.17g auto_tau=%.17g auto_sigma=%.17g\n",
            norm, settings.tau, settings.sigma);
    }
    if (settings.tau <= 0.0 || settings.sigma <= 0.0) {
        usage(argv[0]);
        sk_model_free(&model);
        return 64;
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
                "\"primal_inf\":%.3e,\"solve_seconds\":%.6f,\"verify_status\":\"%s\",\"verify_objective\":%.12g}\n",
        path, gpu_result.status, gpu_result.iterations, gpu_result.objective + model.objshift,
        gpu_result.maximum_row_violation, solve_seconds, sk_status_name(verify_status), checked.objective);

    sankhya_cuda_csr_destroy(&matrix);
    if (!diagonal_hessian) sankhya_cuda_csr_destroy(&hessian);
    sk_model_free(&model);
    return rc == 0 && gpu_result.status == 0 && verify_status == SK_OK &&
        checked.primal_infeasibility <= 100.0 * settings.tolerance ? 0 : 3;
}
