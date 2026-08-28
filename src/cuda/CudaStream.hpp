#pragma once

#include <cuda_runtime.h>

namespace sihps {

// RAII wrapper around a CUDA stream. Deterministic cleanup regardless of
// control flow, including exceptions -- prompt.md \S3.3: "all CUDA
// resources must have deterministic cleanup."
class CudaStream {
public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    cudaStream_t handle() const noexcept { return stream_; }

    // Blocking wait for all work queued on this stream to complete. Used
    // only at explicit synchronization points (construction-time setup,
    // test/benchmark boundaries) -- never inside a performance-critical
    // hot path (prompt.md \S3.6), where CudaEvent-based waits are used
    // instead.
    void synchronize() const;

private:
    cudaStream_t stream_;
};

} // namespace sihps
