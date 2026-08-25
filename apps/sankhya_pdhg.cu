#include "sankhya_lp.h"
#include "sankhya_verify.h"
#include "sankhya_cuda.h"

#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int csc_to_csr(const SankhyaCSC* csc, int** offsets, int** indices, double** values) {
    if (csc == nullptr || csc->rows > INT_MAX || csc->columns > INT_MAX || csc->nonzeros > INT_MAX) return 0;
    *offsets = static_cast<int*>(calloc(csc->rows + 1, sizeof(int)));
    *indices = static_cast<int*>(malloc(csc->nonzeros * sizeof(int)));
    *values = static_cast<double*>(malloc(csc->nonzeros * sizeof(double)));
    if (*offsets == nullptr || (csc->nonzeros > 0 && (*indices == nullptr || *values == nullptr))) return 0;
    for (size_t p = 0; p < csc->nonzeros; ++p) ++(*offsets)[static_cast<size_t>(csc->row_indices[p]) + 1];
    for (size_t row = 0; row < csc->rows; ++row) (*offsets)[row + 1] += (*offsets)[row];
    int* next = static_cast<int*>(malloc(csc->rows * sizeof(int)));
    if (next == nullptr && csc->rows > 0) return 0;
    memcpy(next, *offsets, csc->rows * sizeof(int));
    for (size_t col = 0; col < csc->columns; ++col) {
        for (size_t p = csc->column_offsets[col]; p < csc->column_offsets[col + 1]; ++p) {
            const int slot = next[csc->row_indices[p]]++;
            (*indices)[slot] = static_cast<int>(col);
            (*values)[slot] = csc->values[p];
        }
    }
    free(next);
    return 1;
}

double estimate_operator_norm(const SankhyaCSC* matrix, int iterations) {
    if (matrix == nullptr || matrix->columns == 0) return 0.0;
    double* x = static_cast<double*>(calloc(matrix->columns, sizeof(double)));
    double* y = static_cast<double*>(calloc(matrix->rows, sizeof(double)));
    double* z = static_cast<double*>(calloc(matrix->columns, sizeof(double)));
    if (x == nullptr || y == nullptr || z == nullptr) { free(x); free(y); free(z); return 0.0; }
    const double initial = 1.0 / std::sqrt(static_cast<double>(matrix->columns));
    for (size_t column = 0; column < matrix->columns; ++column) x[column] = initial;
    double lambda = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (sankhya_csc_matvec(matrix, x, y) != SANKHYA_OK ||
            sankhya_csc_transpose_matvec(matrix, y, z) != SANKHYA_OK) break;
        lambda = 0.0;
        for (size_t column = 0; column < matrix->columns; ++column) lambda += z[column] * z[column];
        lambda = std::sqrt(lambda);
        if (!(lambda > 0.0) || !std::isfinite(lambda)) break;
        for (size_t column = 0; column < matrix->columns; ++column) x[column] = z[column] / lambda;
    }
    free(x); free(y); free(z);
    return lambda > 0.0 ? std::sqrt(lambda) : 0.0;
}

void usage(const char* program) {
    std::fprintf(stderr, "usage: %s model.mps [--iterations N] [--tau T] [--sigma S] [--tolerance E]\n", program);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 64; }
    SankhyaCudaLPSettings settings{100000, 100, 0.0, 0.0, 1.0, 1e-6};
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { usage(argv[0]); return 64; }
        if (std::strcmp(argv[i], "--iterations") == 0) settings.max_iterations = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--tau") == 0) settings.tau = std::atof(argv[i + 1]);
        else if (std::strcmp(argv[i], "--sigma") == 0) settings.sigma = std::atof(argv[i + 1]);
        else if (std::strcmp(argv[i], "--tolerance") == 0) settings.tolerance = std::atof(argv[i + 1]);
        else { usage(argv[0]); return 64; }
    }
    SankhyaLPModel model;
    sankhya_lp_model_init(&model);
    if (sankhya_lp_read_mps(argv[1], &model) != SANKHYA_OK) {
        std::fprintf(stderr, "cannot read MPS: %s\n", argv[1]); return 1;
    }
    for (size_t column = 0; column < model.A.columns; ++column) {
        if (model.variable_type[column] != SANKHYA_CONTINUOUS) {
            std::fprintf(stderr, "PDHG CLI currently supports continuous LP only\n");
            sankhya_lp_model_destroy(&model); return 2;
        }
    }
    if (settings.tau == 0.0 && settings.sigma == 0.0) {
        const double norm = estimate_operator_norm(&model.A, 30);
        const double step = norm > 0.0 ? 0.5 / norm : 1.0;
        settings.tau = step;
        settings.sigma = step;
        std::printf("auto_operator_norm=%.17g auto_tau=%.17g auto_sigma=%.17g\n", norm, step, step);
    }
    if (settings.tau <= 0.0 || settings.sigma <= 0.0) {
        std::fprintf(stderr, "tau and sigma must both be positive, or both omitted for automatic selection\n");
        sankhya_lp_model_destroy(&model); return 64;
    }
    int* offsets = nullptr; int* indices = nullptr; double* values = nullptr;
    if (!csc_to_csr(&model.A, &offsets, &indices, &values)) {
        std::fprintf(stderr, "cannot convert CSC model to CSR\n");
        sankhya_lp_model_destroy(&model); return 1;
    }
    SankhyaCudaCSR matrix{};
    if (sankhya_cuda_csr_create(&matrix, static_cast<int>(model.A.rows), static_cast<int>(model.A.columns),
            static_cast<int>(model.A.nonzeros), offsets, indices, values) != 0) {
        std::fprintf(stderr, "CUDA CSR setup failed: %s\n", sankhya_cuda_last_error());
        free(offsets); free(indices); free(values); sankhya_lp_model_destroy(&model); return 1;
    }
    double* solution = static_cast<double*>(calloc(model.A.columns, sizeof(double)));
    SankhyaCudaLPResult result{};
    const auto solve_start = std::chrono::steady_clock::now();
    const int solve_status = sankhya_cuda_lp_pdhg(&matrix, model.objective, model.row_lower, model.row_upper,
        model.column_lower, model.column_upper, settings, solution, &result);
    const double solve_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    SankhyaVerificationResult verification{};
    const SankhyaStatus verify_status = solve_status == 0
        ? sankhya_verify_primal(&model, solution, settings.tolerance, &verification)
        : SANKHYA_INVALID_ARGUMENT;
    std::printf("status=%d iterations=%d objective=%.17g row_violation=%.17g step=%.17g solve_seconds=%.9g\n",
        result.status, result.iterations, result.objective, result.maximum_row_violation, result.maximum_step, solve_seconds);
    if (verify_status == SANKHYA_OK) {
        std::printf("verified_feasible=%d verified_integral=%d primal_violation=%.17g\n",
            verification.feasible, verification.integral, verification.maximum_primal_violation);
    }
    for (size_t column = 0; column < model.A.columns; ++column)
        std::printf("x[%s]=%.17g\n", model.column_names[column], solution[column]);
    free(solution); sankhya_cuda_csr_destroy(&matrix); free(offsets); free(indices); free(values);
    sankhya_lp_model_destroy(&model);
    return solve_status == 0 && result.status == 0 && verify_status == SANKHYA_OK && verification.feasible ? 0 : 3;
}
