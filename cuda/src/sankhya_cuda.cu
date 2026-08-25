#include "sankhya_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
thread_local char g_last_error[512] = "";

int fail(const char* operation, cudaError_t error) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", operation,
                  cudaGetErrorString(error));
    return static_cast<int>(error);
}

int fail_message(const char* message) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
    return -1;
}

template <typename T>
bool valid_ptr(const T* pointer) {
    return pointer != nullptr;
}

__global__ void csr_spmv_f64_kernel(
    int rows, const int* row_offsets, const int* column_indices,
    const double* values, const double* x, double* y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
        sum = fma(values[p], x[column_indices[p]], sum);
    }
    y[row] = sum;
}

__global__ void axpy_f64_kernel(int n, double alpha, const double* x, double* y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) y[index] = fma(alpha, x[index], y[index]);
}

__global__ void csr_transpose_spmv_f64_kernel(
    int rows, const int* row_offsets, const int* column_indices,
    const double* values, const double* y, double* x) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
        atomicAdd(&x[column_indices[p]], values[p] * y[row]);
    }
}

struct DeviceBuffers {
    int* row_offsets = nullptr;
    int* column_indices = nullptr;
    double* values = nullptr;
    double* x = nullptr;
    double* y = nullptr;

    ~DeviceBuffers() {
        cudaFree(row_offsets);
        cudaFree(column_indices);
        cudaFree(values);
        cudaFree(x);
        cudaFree(y);
    }
};

int ensure_cuda_device() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) return fail("cudaGetDeviceCount", error);
    return device_count == 0 ? fail_message("no CUDA device available") : 0;
}

/* Device-pointer primitives enqueue work on the default stream.  A later
 * device-to-host copy or explicit synchronization provides completion. */
int launch_spmv_async(const SankhyaCudaCSR* matrix, const double* device_x, double* device_y) {
    if (matrix == nullptr) return fail_message("null device CSR pointer");
    if (matrix->rows == 0) return 0;
    if (device_y == nullptr || (matrix->cols > 0 && device_x == nullptr))
        return fail_message("null device CSR or vector pointer");
    const int block_size = 256;
    const int grid_size = (matrix->rows + block_size - 1) / block_size;
    csr_spmv_f64_kernel<<<grid_size, block_size>>>(
        matrix->rows,
        static_cast<const int*>(matrix->device_row_offsets),
        static_cast<const int*>(matrix->device_column_indices),
        static_cast<const double*>(matrix->device_values), device_x, device_y);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return fail("CSR SpMV launch", error);
    return 0;
}

int launch_axpy_async(int n, double alpha, const double* device_x, double* device_y) {
    if (n < 0) return fail_message("negative vector dimension");
    if (n == 0) return 0;
    if (device_x == nullptr || device_y == nullptr) return fail_message("null device vector pointer");
    const int block_size = 256;
    axpy_f64_kernel<<<(n + block_size - 1) / block_size, block_size>>>(n, alpha, device_x, device_y);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return fail("AXPY launch", error);
    return 0;
}

int allocate(void** destination, size_t bytes, const char* label) {
    const cudaError_t error = cudaMalloc(destination, bytes);
    return error == cudaSuccess ? 0 : fail(label, error);
}
}  // namespace

extern "C" const char* sankhya_cuda_last_error(void) { return g_last_error; }

extern "C" int sankhya_cuda_csr_create(
    SankhyaCudaCSR* matrix, int rows, int cols, int nnz,
    const int* row_offsets, const int* column_indices, const double* values) {
    g_last_error[0] = '\0';
    if (matrix == nullptr || row_offsets == nullptr || rows < 0 || cols < 0 || nnz < 0 ||
        (nnz > 0 && (column_indices == nullptr || values == nullptr)))
        return fail_message("invalid persistent CSR input");
    if (row_offsets[0] != 0 || row_offsets[rows] != nnz)
        return fail_message("invalid CSR row-offset endpoints");
    for (int row = 0; row < rows; ++row)
        if (row_offsets[row] < 0 || row_offsets[row] > row_offsets[row + 1] ||
            row_offsets[row + 1] > nnz)
            return fail_message("invalid CSR row-offset ordering");
    for (int p = 0; p < nnz; ++p)
        if (column_indices[p] < 0 || column_indices[p] >= cols || !std::isfinite(values[p]))
            return fail_message("invalid CSR column index or coefficient");
    if (ensure_cuda_device()) return -1;
    sankhya_cuda_csr_destroy(matrix);
    matrix->rows = rows; matrix->cols = cols; matrix->nnz = nnz;
    if (allocate(&matrix->device_row_offsets, sizeof(int) * static_cast<size_t>(rows + 1), "cudaMalloc row_offsets")) goto failure;
    if (nnz > 0 && allocate(&matrix->device_column_indices, sizeof(int) * static_cast<size_t>(nnz), "cudaMalloc column_indices")) goto failure;
    if (nnz > 0 && allocate(&matrix->device_values, sizeof(double) * static_cast<size_t>(nnz), "cudaMalloc values")) goto failure;
    {
        cudaError_t error = cudaMemcpy(matrix->device_row_offsets, row_offsets, sizeof(int) * static_cast<size_t>(rows + 1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) { fail("cudaMemcpy row_offsets", error); goto failure; }
        if (nnz > 0) {
            error = cudaMemcpy(matrix->device_column_indices, column_indices, sizeof(int) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) { fail("cudaMemcpy column_indices", error); goto failure; }
            error = cudaMemcpy(matrix->device_values, values, sizeof(double) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) { fail("cudaMemcpy values", error); goto failure; }
        }
    }
    return 0;
failure:
    sankhya_cuda_csr_destroy(matrix);
    return -1;
}

extern "C" void sankhya_cuda_csr_destroy(SankhyaCudaCSR* matrix) {
    if (matrix == nullptr) return;
    cudaFree(matrix->device_row_offsets);
    cudaFree(matrix->device_column_indices);
    cudaFree(matrix->device_values);
    matrix->rows = matrix->cols = matrix->nnz = 0;
    matrix->device_row_offsets = nullptr;
    matrix->device_column_indices = nullptr;
    matrix->device_values = nullptr;
}

extern "C" int sankhya_cuda_spmv_device_f64(
    const SankhyaCudaCSR* matrix, const double* device_x, double* device_y) {
    g_last_error[0] = '\0';
    if (ensure_cuda_device()) return -1;
    return launch_spmv_async(matrix, device_x, device_y);
}

extern "C" int sankhya_cuda_spmv_transpose_device_f64(
    const SankhyaCudaCSR* matrix, const double* device_y, double* device_x) {
    g_last_error[0] = '\0';
    if (ensure_cuda_device()) return -1;
    if (matrix == nullptr) return fail_message("null device CSR pointer");
    if (matrix->rows == 0 || matrix->cols == 0) return 0;
    if (device_y == nullptr || device_x == nullptr)
        return fail_message("null device CSR or transpose vector pointer");
    cudaError_t error = cudaMemset(device_x, 0, sizeof(double) * static_cast<size_t>(matrix->cols));
    if (error != cudaSuccess) return fail("transpose SpMV memset", error);
    const int block_size = 256;
    const int grid_size = (matrix->rows + block_size - 1) / block_size;
    csr_transpose_spmv_f64_kernel<<<grid_size, block_size>>>(
        matrix->rows, static_cast<const int*>(matrix->device_row_offsets),
        static_cast<const int*>(matrix->device_column_indices),
        static_cast<const double*>(matrix->device_values), device_y, device_x);
    if ((error = cudaGetLastError()) != cudaSuccess) return fail("transpose SpMV launch", error);
    return 0;
}

extern "C" int sankhya_cuda_axpy_device_f64(
    int n, double alpha, const double* device_x, double* device_y) {
    g_last_error[0] = '\0';
    if (ensure_cuda_device()) return -1;
    return launch_axpy_async(n, alpha, device_x, device_y);
}

extern "C" int sankhya_cuda_spmv_f64(
    int rows, int cols, int nnz, const int* row_offsets,
    const int* column_indices, const double* values, const double* x,
    double* y) {
    g_last_error[0] = '\0';
    if (rows < 0 || cols < 0 || nnz < 0) return fail_message("negative CSR dimension");
    if (!valid_ptr(row_offsets) || !valid_ptr(y) ||
        (nnz > 0 && (!valid_ptr(column_indices) || !valid_ptr(values))) ||
        (cols > 0 && !valid_ptr(x))) {
        return fail_message("null CSR or vector pointer");
    }
    if (rows == 0) return 0;
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) return fail("cudaGetDeviceCount", error);
    if (device_count == 0) return fail_message("no CUDA device available");

    DeviceBuffers device;
    if (allocate(reinterpret_cast<void**>(&device.row_offsets),
                 sizeof(int) * static_cast<size_t>(rows + 1), "cudaMalloc row_offsets")) return -1;
    if (nnz > 0 && allocate(reinterpret_cast<void**>(&device.column_indices),
                 sizeof(int) * static_cast<size_t>(nnz), "cudaMalloc column_indices")) return -1;
    if (nnz > 0 && allocate(reinterpret_cast<void**>(&device.values),
                 sizeof(double) * static_cast<size_t>(nnz), "cudaMalloc values")) return -1;
    if (cols > 0 && allocate(reinterpret_cast<void**>(&device.x),
                 sizeof(double) * static_cast<size_t>(cols), "cudaMalloc x")) return -1;
    if (allocate(reinterpret_cast<void**>(&device.y),
                 sizeof(double) * static_cast<size_t>(rows), "cudaMalloc y")) return -1;

    const size_t row_bytes = sizeof(int) * static_cast<size_t>(rows + 1);
    const size_t index_bytes = sizeof(int) * static_cast<size_t>(nnz);
    const size_t value_bytes = sizeof(double) * static_cast<size_t>(nnz);
    const size_t x_bytes = sizeof(double) * static_cast<size_t>(cols);
    const size_t y_bytes = sizeof(double) * static_cast<size_t>(rows);
    if ((error = cudaMemcpy(device.row_offsets, row_offsets, row_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        return fail("cudaMemcpy row_offsets", error);
    if (nnz > 0 && (error = cudaMemcpy(device.column_indices, column_indices, index_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        return fail("cudaMemcpy column_indices", error);
    if (nnz > 0 && (error = cudaMemcpy(device.values, values, value_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        return fail("cudaMemcpy values", error);
    if (cols > 0 && (error = cudaMemcpy(device.x, x, x_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        return fail("cudaMemcpy x", error);

    const int block_size = 256;
    const int grid_size = (rows + block_size - 1) / block_size;
    csr_spmv_f64_kernel<<<grid_size, block_size>>>(rows, device.row_offsets,
        device.column_indices, device.values, device.x, device.y);
    if ((error = cudaGetLastError()) != cudaSuccess) return fail("CSR SpMV launch", error);
    /* The blocking device-to-host copy below provides the required stream
       completion and also reports any asynchronous kernel failure. */
    if ((error = cudaMemcpy(y, device.y, y_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
        return fail("cudaMemcpy y", error);
    return 0;
}

extern "C" int sankhya_cuda_axpy_f64(int n, double alpha, const double* x, double* y) {
    g_last_error[0] = '\0';
    if (n < 0) return fail_message("negative vector dimension");
    if (n == 0) return 0;
    if (!valid_ptr(x) || !valid_ptr(y)) return fail_message("null vector pointer");
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) return fail("cudaGetDeviceCount", error);
    if (device_count == 0) return fail_message("no CUDA device available");
    double* device_x = nullptr;
    double* device_y = nullptr;
    if (allocate(reinterpret_cast<void**>(&device_x), sizeof(double) * static_cast<size_t>(n), "cudaMalloc x")) return -1;
    if (allocate(reinterpret_cast<void**>(&device_y), sizeof(double) * static_cast<size_t>(n), "cudaMalloc y")) {
        cudaFree(device_x);
        return -1;
    }
    error = cudaMemcpy(device_x, x, sizeof(double) * static_cast<size_t>(n), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaMemcpy(device_y, y, sizeof(double) * static_cast<size_t>(n), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(device_x); cudaFree(device_y);
        return fail("cudaMemcpy AXPY input", error);
    }
    const int block_size = 256;
    axpy_f64_kernel<<<(n + block_size - 1) / block_size, block_size>>>(n, alpha, device_x, device_y);
    if ((error = cudaGetLastError()) != cudaSuccess) {
        cudaFree(device_x); cudaFree(device_y); return fail("AXPY launch", error);
    }
    error = cudaMemcpy(y, device_y, sizeof(double) * static_cast<size_t>(n), cudaMemcpyDeviceToHost);
    cudaFree(device_x); cudaFree(device_y);
    return error == cudaSuccess ? 0 : fail("cudaMemcpy AXPY output", error);
}
