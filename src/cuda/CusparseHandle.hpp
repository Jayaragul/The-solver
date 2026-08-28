#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

namespace sihps {

// RAII wrapper around a cusparseHandle_t. Deterministic cleanup, per
// prompt.md \S3.3.
class CusparseHandle {
public:
    CusparseHandle();
    ~CusparseHandle();

    CusparseHandle(const CusparseHandle&) = delete;
    CusparseHandle& operator=(const CusparseHandle&) = delete;
    CusparseHandle(CusparseHandle&& other) noexcept;
    CusparseHandle& operator=(CusparseHandle&& other) noexcept;

    cusparseHandle_t handle() const noexcept { return handle_; }
    void set_stream(cudaStream_t stream);

private:
    cusparseHandle_t handle_;
};

} // namespace sihps
