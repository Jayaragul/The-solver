#include "CusparseHandle.hpp"

#include "CudaCheck.hpp"

namespace sihps {

CusparseHandle::CusparseHandle() : handle_(nullptr) {
    SIHPS_CUSPARSE_CHECK(cusparseCreate(&handle_));
}

CusparseHandle::~CusparseHandle() {
    if (handle_) cusparseDestroy(handle_);
}

CusparseHandle::CusparseHandle(CusparseHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

CusparseHandle& CusparseHandle::operator=(CusparseHandle&& other) noexcept {
    if (this != &other) {
        if (handle_) cusparseDestroy(handle_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void CusparseHandle::set_stream(cudaStream_t stream) {
    SIHPS_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
}

} // namespace sihps
