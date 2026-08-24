#pragma once

#include "CudaCheck.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace sihps {

// RAII owner of a pinned (page-locked) host array of T. Used only for
// buffers that cross PCIe repeatedly (docs/architecture/CPU_GPU.md \S3.3):
// pinning a one-time-use buffer is pure overhead and is deliberately not
// done elsewhere in this codebase (e.g. RawModel input, MEMORY.md \S2).
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() noexcept : ptr_(nullptr), count_(0) {}

    explicit PinnedBuffer(std::size_t count) : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            SIHPS_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T),
                                            cudaHostAllocDefault));
        }
    }

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
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

private:
    T* ptr_;
    std::size_t count_;
};

} // namespace sihps
