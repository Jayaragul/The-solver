#include "GpuSpMV.hpp"

#include "CudaCheck.hpp"

#include <algorithm>

namespace sihps {

GpuCsrSpMV::GpuCsrSpMV(const CSRMatrix& host_matrix, CusparseHandle& handle, CudaStream& stream,
                        cusparseSpMVAlg_t algorithm)
    : rows_(host_matrix.rows()),
      cols_(host_matrix.cols()),
      nnz_(host_matrix.nnz()),
      handle_(handle),
      stream_(stream),
      d_row_ptr_(static_cast<std::size_t>(rows_) + 1),
      d_col_idx_(static_cast<std::size_t>(nnz_)),
      d_values_(static_cast<std::size_t>(nnz_)),
      d_x_(static_cast<std::size_t>(cols_)),
      d_y_(static_cast<std::size_t>(rows_)),
      d_workspace_(0),
      h_x_staging_(static_cast<std::size_t>(cols_)),
      h_y_staging_(static_cast<std::size_t>(rows_)),
      mat_descr_(nullptr),
      vec_x_descr_(nullptr),
      vec_y_descr_(nullptr),
      algorithm_(algorithm) {
    // One-time upload of the matrix's structure and values, persistent for
    // this object's lifetime (docs/architecture/CPU_GPU.md \S3.1). Staged
    // through pinned host memory for true async DMA, even though this
    // particular transfer only happens once -- the row_ptr/col_idx/values
    // arrays are exactly the data CPU_GPU.md \S3.2 identifies as crossing
    // PCIe once per solve.
    PinnedBuffer<std::int32_t> h_row_ptr_staging(static_cast<std::size_t>(rows_) + 1);
    PinnedBuffer<std::int32_t> h_col_idx_staging(static_cast<std::size_t>(nnz_));
    PinnedBuffer<double> h_values_staging(static_cast<std::size_t>(nnz_));

    std::copy(host_matrix.row_ptr(), host_matrix.row_ptr() + rows_ + 1, h_row_ptr_staging.data());
    std::copy(host_matrix.col_idx(), host_matrix.col_idx() + nnz_, h_col_idx_staging.data());
    std::copy(host_matrix.values(), host_matrix.values() + nnz_, h_values_staging.data());

    d_row_ptr_.copy_from_host_async(h_row_ptr_staging.data(), static_cast<std::size_t>(rows_) + 1,
                                     stream_.handle());
    d_col_idx_.copy_from_host_async(h_col_idx_staging.data(), static_cast<std::size_t>(nnz_),
                                     stream_.handle());
    d_values_.copy_from_host_async(h_values_staging.data(), static_cast<std::size_t>(nnz_),
                                    stream_.handle());
    stream_.synchronize(); // one-time setup; not part of the repeated hot path

    SIHPS_CUSPARSE_CHECK(cusparseCreateCsr(&mat_descr_, rows_, cols_, nnz_, d_row_ptr_.data(),
                                            d_col_idx_.data(), d_values_.data(),
                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    SIHPS_CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_descr_, cols_, d_x_.data(), CUDA_R_64F));
    SIHPS_CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_descr_, rows_, d_y_.data(), CUDA_R_64F));

    handle_.set_stream(stream_.handle());

    const double alpha = 1.0;
    const double beta = 0.0;
    std::size_t buffer_size = 0;
    SIHPS_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle_.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, mat_descr_, vec_x_descr_, &beta,
                                                  vec_y_descr_, CUDA_R_64F, algorithm_,
                                                  &buffer_size));

    d_workspace_ = CudaBuffer<unsigned char>(buffer_size); // sized once, at init

    // Hoist whatever per-matrix analysis the algorithm needs out of the
    // repeated call. The matrix structure and the workspace are fixed for
    // this object's lifetime and only x's CONTENTS change between calls,
    // which is exactly the precondition cusparseSpMV_preprocess documents.
    // Skipping this would leave that analysis inside every SpMV -- and on
    // an LP hot path that call happens once or twice per simplex
    // iteration, thousands of times per solve.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12400
    SIHPS_CUSPARSE_CHECK(cusparseSpMV_preprocess(
        handle_.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_descr_, vec_x_descr_, &beta,
        vec_y_descr_, CUDA_R_64F, algorithm_, d_workspace_.data()));
    stream_.synchronize();
#endif
}

GpuCsrSpMV::~GpuCsrSpMV() {
    if (vec_y_descr_) cusparseDestroyDnVec(vec_y_descr_);
    if (vec_x_descr_) cusparseDestroyDnVec(vec_x_descr_);
    if (mat_descr_) cusparseDestroySpMat(mat_descr_);
}

void GpuCsrSpMV::multiply_device_resident() {
    const double alpha = 1.0;
    const double beta = 0.0;
    SIHPS_CUSPARSE_CHECK(cusparseSpMV(handle_.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                       mat_descr_, vec_x_descr_, &beta, vec_y_descr_, CUDA_R_64F,
                                       algorithm_, d_workspace_.data()));
}

void GpuCsrSpMV::upload_x_async(const double* pinned_host_x) {
    d_x_.copy_from_host_async(pinned_host_x, static_cast<std::size_t>(cols_), stream_.handle());
}

void GpuCsrSpMV::set_vectors(double* d_x, double* d_y) {
    SIHPS_CUSPARSE_CHECK(cusparseDnVecSetValues(vec_x_descr_, d_x ? d_x : d_x_.data()));
    SIHPS_CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y_descr_, d_y ? d_y : d_y_.data()));
}

void GpuCsrSpMV::multiply(const double* host_x, double* host_y) {
    // Restore the internal binding: a caller may have pointed the
    // descriptors elsewhere via set_vectors, and this entry point is
    // defined in terms of this object own buffers.
    set_vectors(nullptr, nullptr);
    std::copy(host_x, host_x + cols_, h_x_staging_.data());
    d_x_.copy_from_host_async(h_x_staging_.data(), static_cast<std::size_t>(cols_),
                               stream_.handle());
    multiply_device_resident();
    d_y_.copy_to_host_async(h_y_staging_.data(), static_cast<std::size_t>(rows_), stream_.handle());
    stream_.synchronize();
    std::copy(h_y_staging_.data(), h_y_staging_.data() + rows_, host_y);
}

} // namespace sihps
