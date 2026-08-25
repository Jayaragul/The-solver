#include "sankhya_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

__device__ double project_box(double value, double lower, double upper) {
    if (!isinf(lower) && value < lower) value = lower;
    if (!isinf(upper) && value > upper) value = upper;
    return value;
}

__global__ void initialize_box_kernel(
    int n, const double* lower, const double* upper, double* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = project_box(0.0, lower[i], upper[i]);
}

__global__ void dual_update_kernel(
    int n, double sigma, const double* old_dual, const double* activity,
    const double* lower, const double* upper, double* new_dual) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double trial = old_dual[i] + sigma * activity[i];
    new_dual[i] = trial - sigma * project_box(trial / sigma, lower[i], upper[i]);
}

__global__ void primal_update_extrapolate_kernel(
    int n, double tau, const double* old_x, const double* cost,
    const double* gradient, const double* quadratic_diagonal,
    const double* lower, const double* upper,
    double theta, double* new_x, double* extrapolated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double curvature = quadratic_diagonal == nullptr ? 0.0 : quadratic_diagonal[i];
    const double trial = (old_x[i] - tau * (cost[i] + gradient[i])) / (1.0 + tau * curvature);
    const double next = project_box(trial, lower[i], upper[i]);
    new_x[i] = next;
    extrapolated[i] = next + theta * (next - old_x[i]);
}

int check_cuda() {
    int devices = 0;
    cudaError_t error = cudaGetDeviceCount(&devices);
    if (error != cudaSuccess || devices == 0) return -1;
    return 0;
}

template <typename T>
void free_device(T*& pointer) {
    cudaFree(pointer);
    pointer = nullptr;
}

double row_violation(
    const std::vector<double>& activity,
    const double* lower,
    const double* upper) {
    double maximum = 0.0;
    for (size_t i = 0; i < activity.size(); ++i) {
        if (std::isfinite(lower[i])) maximum = std::max(maximum, lower[i] - activity[i]);
        if (std::isfinite(upper[i])) maximum = std::max(maximum, activity[i] - upper[i]);
    }
    return std::max(0.0, maximum);
}

}  // namespace

int solve_diagonal_qp(
    const SankhyaCudaCSR* matrix,
    const double* quadratic_diagonal,
    const double* c,
    const double* row_lower,
    const double* row_upper,
    const double* col_lower,
    const double* col_upper,
    SankhyaCudaLPSettings settings,
    double* solution,
    SankhyaCudaLPResult* result) {
    if (result == nullptr) return -1;
    result->status = -1;
    result->iterations = 0;
    result->objective = std::numeric_limits<double>::quiet_NaN();
    result->maximum_row_violation = std::numeric_limits<double>::infinity();
    result->maximum_step = std::numeric_limits<double>::infinity();
    if (check_cuda() != 0 || matrix == nullptr || c == nullptr || row_lower == nullptr ||
        row_upper == nullptr || col_lower == nullptr || col_upper == nullptr || solution == nullptr)
        return -1;
    if (settings.max_iterations <= 0 || settings.check_every <= 0 || settings.tau <= 0.0 ||
        settings.sigma <= 0.0 || settings.theta < 0.0 || settings.theta > 1.0 || settings.tolerance <= 0.0)
        return -1;
    const int rows = matrix->rows;
    const int cols = matrix->cols;
    if (rows < 0 || cols < 0) return -1;
    if (quadratic_diagonal != nullptr) {
        for (int column = 0; column < cols; ++column) {
            if (!std::isfinite(quadratic_diagonal[column]) || quadratic_diagonal[column] < 0.0) return -1;
        }
    }

    double* d_c = nullptr;
    double* d_quadratic_diagonal = nullptr;
    double* d_row_lower = nullptr;
    double* d_row_upper = nullptr;
    double* d_col_lower = nullptr;
    double* d_col_upper = nullptr;
    double* d_x = nullptr;
    double* d_x_new = nullptr;
    double* d_x_bar = nullptr;
    double* d_y = nullptr;
    double* d_y_new = nullptr;
    double* d_activity = nullptr;
    double* d_gradient = nullptr;
    auto cleanup = [&]() {
        free_device(d_c); free_device(d_quadratic_diagonal); free_device(d_row_lower); free_device(d_row_upper);
        free_device(d_col_lower); free_device(d_col_upper); free_device(d_x);
        free_device(d_x_new); free_device(d_x_bar); free_device(d_y);
        free_device(d_y_new); free_device(d_activity); free_device(d_gradient);
    };
    auto allocate_copy = [](double** destination, const double* source, size_t count) -> bool {
        if (count == 0) return true;
        if (cudaMalloc(destination, sizeof(double) * count) != cudaSuccess) return false;
        if (cudaMemcpy(*destination, source, sizeof(double) * count, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        return true;
    };
    if (!allocate_copy(&d_c, c, static_cast<size_t>(cols)) ||
        (quadratic_diagonal != nullptr && !allocate_copy(&d_quadratic_diagonal, quadratic_diagonal, static_cast<size_t>(cols))) ||
        !allocate_copy(&d_row_lower, row_lower, static_cast<size_t>(rows)) ||
        !allocate_copy(&d_row_upper, row_upper, static_cast<size_t>(rows)) ||
        !allocate_copy(&d_col_lower, col_lower, static_cast<size_t>(cols)) ||
        !allocate_copy(&d_col_upper, col_upper, static_cast<size_t>(cols)) ||
        (cols > 0 && (cudaMalloc(&d_x, sizeof(double) * cols) != cudaSuccess ||
                      cudaMalloc(&d_x_new, sizeof(double) * cols) != cudaSuccess ||
                      cudaMalloc(&d_x_bar, sizeof(double) * cols) != cudaSuccess ||
                      cudaMalloc(&d_gradient, sizeof(double) * cols) != cudaSuccess)) ||
        (rows > 0 && (cudaMalloc(&d_y, sizeof(double) * rows) != cudaSuccess ||
                      cudaMalloc(&d_y_new, sizeof(double) * rows) != cudaSuccess ||
                      cudaMalloc(&d_activity, sizeof(double) * rows) != cudaSuccess))) {
        cleanup();
        return -1;
    }
    const int block = 256;
    if (cols > 0) {
        initialize_box_kernel<<<(cols + block - 1) / block, block>>>(cols, d_col_lower, d_col_upper, d_x);
        if (cudaGetLastError() != cudaSuccess || cudaMemcpy(d_x_bar, d_x, sizeof(double) * cols, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            cleanup(); return -1;
        }
    }
    if (rows > 0 && cudaMemset(d_y, 0, sizeof(double) * rows) != cudaSuccess) {
        cleanup(); return -1;
    }
    if (cols > 0 && rows == 0 && cudaMemset(d_gradient, 0, sizeof(double) * cols) != cudaSuccess) {
        cleanup(); return -1;
    }
    std::vector<double> host_x(static_cast<size_t>(cols));
    std::vector<double> previous_x(static_cast<size_t>(cols));
    std::vector<double> host_activity(static_cast<size_t>(rows));
    int have_previous_x = 0;
    int final_iteration = settings.max_iterations;
    for (int iteration = 1; iteration <= settings.max_iterations; ++iteration) {
        if (sankhya_cuda_spmv_device_f64(matrix, d_x_bar, d_activity) != 0) { cleanup(); return -1; }
        if (rows > 0) {
            dual_update_kernel<<<(rows + block - 1) / block, block>>>(rows, settings.sigma, d_y, d_activity, d_row_lower, d_row_upper, d_y_new);
            if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }
        }
        if (sankhya_cuda_spmv_transpose_device_f64(matrix, d_y_new, d_gradient) != 0) { cleanup(); return -1; }
        if (cols > 0) {
            primal_update_extrapolate_kernel<<<(cols + block - 1) / block, block>>>(
                cols, settings.tau, d_x, d_c, d_gradient, d_quadratic_diagonal,
                d_col_lower, d_col_upper, settings.theta, d_x_new, d_x_bar);
            if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }
        }
        std::swap(d_x, d_x_new);
        std::swap(d_y, d_y_new);
        if (iteration % settings.check_every == 0 || iteration == settings.max_iterations) {
            if (cols > 0 && cudaMemcpy(host_x.data(), d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost) != cudaSuccess) { cleanup(); return -1; }
            if (sankhya_cuda_spmv_device_f64(matrix, d_x, d_activity) != 0 ||
                (rows > 0 && cudaMemcpy(host_activity.data(), d_activity, sizeof(double) * rows, cudaMemcpyDeviceToHost) != cudaSuccess)) { cleanup(); return -1; }
            double objective = 0.0;
            for (int column = 0; column < cols; ++column) {
                const double x = host_x[static_cast<size_t>(column)];
                objective += c[column] * x;
                if (quadratic_diagonal != nullptr) objective += 0.5 * quadratic_diagonal[column] * x * x;
            }
            double step = have_previous_x ? 0.0 : std::numeric_limits<double>::infinity();
            double scale = 1.0;
            for (int column = 0; column < cols; ++column) {
                const size_t index = static_cast<size_t>(column);
                if (have_previous_x) step = std::max(step, std::fabs(host_x[index] - previous_x[index]));
                scale = std::max(scale, std::fabs(host_x[index]));
                previous_x[index] = host_x[index];
            }
            have_previous_x = 1;
            const double violation = row_violation(host_activity, row_lower, row_upper);
            result->objective = objective;
            result->maximum_row_violation = violation;
            result->maximum_step = step;
            if (violation <= settings.tolerance && step <= settings.tolerance * scale) {
                final_iteration = iteration; result->status = 0; break;
            }
        }
    }
    if (result->status != 0) result->status = 1;
    result->iterations = final_iteration;
    if (cols > 0 && cudaMemcpy(solution, d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost) != cudaSuccess) { cleanup(); return -1; }
    cleanup();
    return 0;
}

extern "C" int sankhya_cuda_lp_pdhg(
    const SankhyaCudaCSR* matrix, const double* c, const double* row_lower,
    const double* row_upper, const double* col_lower, const double* col_upper,
    SankhyaCudaLPSettings settings, double* solution, SankhyaCudaLPResult* result) {
    return solve_diagonal_qp(matrix, nullptr, c, row_lower, row_upper, col_lower,
        col_upper, settings, solution, result);
}

extern "C" int sankhya_cuda_diagonal_qp_pdhg(
    const SankhyaCudaCSR* matrix, const double* quadratic_diagonal, const double* c,
    const double* row_lower, const double* row_upper, const double* col_lower,
    const double* col_upper, SankhyaCudaLPSettings settings, double* solution,
    SankhyaCudaLPResult* result) {
    if (quadratic_diagonal == nullptr) return -1;
    return solve_diagonal_qp(matrix, quadratic_diagonal, c, row_lower, row_upper,
        col_lower, col_upper, settings, solution, result);
}
