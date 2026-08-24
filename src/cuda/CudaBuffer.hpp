#pragma once

#include "CudaCheck.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace sihps {

// RAII owner of a device-resident array of T. Non-copyable, movable
// (ownership transfer only). Every solve-lifetime device allocation in
// this project is an instance of this type (docs/architecture/MEMORY.md
// \S3.2) -- there is no path in this codebase that calls cudaMalloc/
// cudaFree directly outside of this class.
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() noexcept : ptr_(nullptr), count_(0) {}

    explicit CudaBuffer(std::size_t count) : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            SIHPS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    // Async H2D/D2H, tied to an explicit stream -- never an implicit
    // synchronizing copy (docs/architecture/CPU_GPU.md \S3.3). The caller
    // is responsible for synchronizing (event or stream) before reading
    // the result on the host or reusing the source buffer.
    void copy_from_host_async(const T* host_ptr, std::size_t count, cudaStream_t stream) {
        SIHPS_CUDA_CHECK(
            cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host_async(T* host_ptr, std::size_t count, cudaStream_t stream) const {
        SIHPS_CUDA_CHECK(
            cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

private:
    T* ptr_;
    std::size_t count_;
};

} // namespace sihps
