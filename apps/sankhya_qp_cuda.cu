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
        std::fprintf(stderr, "CUDA QP path requires a QPS model with a diagonal Hessian\n");
        sk_model_free(&model);
        return 2;
    }
    std::vector<double> diagonal(static_cast<size_t>(model.ncol), 0.0);
    {
        for (int column = 0; column < model.ncol; ++column) {
            for (int p = model.Q->p[column]; p < model.Q->p[column + 1]; ++p) {
                if (model.Q->i[p] != column || !std::isfinite(model.Q->x[p]) || model.Q->x[p] < 0.0) {
                    std::fprintf(stderr, "CUDA QP path requires a finite nonnegative diagonal Hessian\n");
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
        settings.tau = step;
        settings.sigma = step;
        std::fprintf(stderr, "auto_operator_norm=%.17g auto_tau=%.17g auto_sigma=%.17g\n", norm, step, step);
    }
    if (settings.tau <= 0.0 || settings.sigma <= 0.0) {
        usage(argv[0]);
        sk_model_free(&model);
        return 64;
    }

    std::vector<int> offsets, indices;
    std::vector<double> values;
    csc_to_csr(model.A, offsets, indices, values);
    SankhyaCudaCSR matrix{};
    if (sankhya_cuda_csr_create(&matrix, model.nrow, model.ncol,
            static_cast<int>(values.size()), offsets.data(), indices.data(), values.data()) != 0) {
        std::fprintf(stderr, "CUDA CSR setup failed: %s\n", sankhya_cuda_last_error());
        sk_model_free(&model);
        return 1;
    }

    std::vector<double> solution(static_cast<size_t>(model.ncol), 0.0);
    SankhyaCudaLPResult gpu_result{};
    const auto solve_start = std::chrono::steady_clock::now();
    const int rc = sankhya_cuda_diagonal_qp_pdhg(&matrix, diagonal.data(),
        model.c, model.rlow, model.rupp, model.clow, model.cupp, settings,
        solution.data(), &gpu_result);
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
    sk_model_free(&model);
    return rc == 0 && gpu_result.status == 0 && verify_status == SK_OK &&
        checked.primal_infeasibility <= 100.0 * settings.tolerance ? 0 : 3;
}
