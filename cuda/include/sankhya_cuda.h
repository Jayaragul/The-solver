#pragma once

#if defined(_WIN32) && defined(SANKHYA_CUDA_BUILD)
#define SANKHYA_CUDA_API __declspec(dllexport)
#else
#define SANKHYA_CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Matrix storage that remains resident on the selected CUDA device. */
typedef struct SankhyaCudaCSR {
    int rows;
    int cols;
    int nnz;
    void* device_row_offsets;
    void* device_column_indices;
    void* device_values;
} SankhyaCudaCSR;

SANKHYA_CUDA_API int sankhya_cuda_csr_create(
    SankhyaCudaCSR* matrix,
    int rows,
    int cols,
    int nnz,
    const int* row_offsets,
    const int* column_indices,
    const double* values);
SANKHYA_CUDA_API void sankhya_cuda_csr_destroy(SankhyaCudaCSR* matrix);

/* Device-pointer variants for iterative algorithms.  These enqueue work on
   CUDA's default stream and return after launch; use a later device-to-host
   copy or explicit synchronization when host visibility is required. */
SANKHYA_CUDA_API int sankhya_cuda_spmv_device_f64(
    const SankhyaCudaCSR* matrix, const double* device_x, double* device_y);
SANKHYA_CUDA_API int sankhya_cuda_spmv_transpose_device_f64(
    const SankhyaCudaCSR* matrix, const double* device_y, double* device_x);
SANKHYA_CUDA_API int sankhya_cuda_axpy_device_f64(int n, double alpha, const double* device_x, double* device_y);

typedef struct SankhyaCudaLPSettings {
    int max_iterations;
    int check_every;
    double tau;
    double sigma;
    double theta;
    double tolerance;
} SankhyaCudaLPSettings;

typedef struct SankhyaCudaLPResult {
    int status; /* 0 approximate convergence, 1 iteration limit, negative error */
    int iterations;
    double objective;
    double maximum_row_violation;
    double maximum_step;
} SankhyaCudaLPResult;

/* Solve min c'x subject to row_lower <= A*x <= row_upper and
   col_lower <= x <= col_upper. All model arrays are host memory. */
SANKHYA_CUDA_API int sankhya_cuda_lp_pdhg(
    const SankhyaCudaCSR* matrix,
    const double* c,
    const double* row_lower,
    const double* row_upper,
    const double* col_lower,
    const double* col_upper,
    SankhyaCudaLPSettings settings,
    double* solution,
    SankhyaCudaLPResult* result);

/* Solve the convex diagonal QP
   min 0.5 * sum_i quadratic_diagonal[i] * x[i]^2 + c'x
   under the same row and column bounds as sankhya_cuda_lp_pdhg. Every
   diagonal entry must be finite and nonnegative. This is deliberately a
   diagonal-Q specialization, not a claim of general QPS support. */
SANKHYA_CUDA_API int sankhya_cuda_diagonal_qp_pdhg(
    const SankhyaCudaCSR* matrix,
    const double* quadratic_diagonal,
    const double* c,
    const double* row_lower,
    const double* row_upper,
    const double* col_lower,
    const double* col_upper,
    SankhyaCudaLPSettings settings,
    double* solution,
    SankhyaCudaLPResult* result);

/* Solve a convex sparse QP with a full resident CSR Hessian. The caller must
   provide Q with dimensions cols x cols; the first-order path uses an
   explicit Q*x product and reports approximate convergence. */
SANKHYA_CUDA_API int sankhya_cuda_sparse_qp_pdhg(
    const SankhyaCudaCSR* matrix,
    const SankhyaCudaCSR* hessian,
    const double* c,
    const double* row_lower,
    const double* row_upper,
    const double* col_lower,
    const double* col_upper,
    SankhyaCudaLPSettings settings,
    double* solution,
    SankhyaCudaLPResult* result);

/* Return zero on success. All host pointers are ordinary CPU memory. */
SANKHYA_CUDA_API int sankhya_cuda_spmv_f64(
    int rows,
    int cols,
    int nnz,
    const int* row_offsets,
    const int* column_indices,
    const double* values,
    const double* x,
    double* y);

/* y <- alpha*x + y, with x and y in host memory. */
SANKHYA_CUDA_API int sankhya_cuda_axpy_f64(int n, double alpha, const double* x, double* y);

/* Thread-local diagnostic for the last nonzero return code. */
SANKHYA_CUDA_API const char* sankhya_cuda_last_error(void);

#ifdef __cplusplus
}
#endif
