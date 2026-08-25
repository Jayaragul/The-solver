#include "sankhya_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    // [[1, 0, 2], [0, 3, 4]] * [2, 5, 7] = [16, 43]
    const int offsets[] = {0, 2, 4};
    const int columns[] = {0, 2, 1, 2};
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const double x[] = {2.0, 5.0, 7.0};
    double y[] = {0.0, 0.0};
    SankhyaCudaCSR rejected{};
    const int bad_offsets[] = {0, 3, 2};
    if (sankhya_cuda_csr_create(&rejected, 2, 3, 4, bad_offsets, columns, values) == 0) return 1;
    const int bad_columns[] = {0, 3, 1, 2};
    if (sankhya_cuda_csr_create(&rejected, 2, 3, 4, offsets, bad_columns, values) == 0) return 2;
    const double bad_values[] = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0, 4.0};
    if (sankhya_cuda_csr_create(&rejected, 2, 3, 4, offsets, columns, bad_values) == 0) return 3;
    const int status = sankhya_cuda_spmv_f64(2, 3, 4, offsets, columns, values, x, y);
    if (status != 0) {
        std::fprintf(stderr, "SpMV failed: %s\n", sankhya_cuda_last_error());
        return 4;
    }
    if (std::fabs(y[0] - 16.0) > 1e-12 || std::fabs(y[1] - 43.0) > 1e-12) {
        std::fprintf(stderr, "wrong result: %.17g %.17g\n", y[0], y[1]);
        return 5;
    }

    SankhyaCudaCSR persistent{};
    if (sankhya_cuda_csr_create(&persistent, 2, 3, 4, offsets, columns, values) != 0) {
        std::fprintf(stderr, "persistent CSR failed: %s\n", sankhya_cuda_last_error());
        return 6;
    }
    double* device_x = nullptr;
    double* device_y = nullptr;
    if (cudaMalloc(&device_x, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&device_y, sizeof(y)) != cudaSuccess) return 7;
    if (cudaMemcpy(device_x, x, sizeof(x), cudaMemcpyHostToDevice) != cudaSuccess ||
        sankhya_cuda_spmv_device_f64(&persistent, device_x, device_y) != 0 ||
        cudaMemcpy(y, device_y, sizeof(y), cudaMemcpyDeviceToHost) != cudaSuccess) return 8;
    if (std::fabs(y[0] - 16.0) > 1e-12 || std::fabs(y[1] - 43.0) > 1e-12) return 9;
    /* Device primitives are queued: chain A*x -> A^T*y -> A*x without a
       host synchronization between launches.  The final copy establishes
       visibility and verifies default-stream ordering. */
    if (sankhya_cuda_spmv_transpose_device_f64(&persistent, device_y, device_x) != 0 ||
        sankhya_cuda_spmv_device_f64(&persistent, device_x, device_y) != 0 ||
        cudaMemcpy(y, device_y, sizeof(y), cudaMemcpyDeviceToHost) != cudaSuccess) return 10;
    cudaFree(device_x);
    cudaFree(device_y);
    sankhya_cuda_csr_destroy(&persistent);
    if (std::fabs(y[0] - 424.0) > 1e-12 || std::fabs(y[1] - 1203.0) > 1e-12) return 11;
    return 0;
}
