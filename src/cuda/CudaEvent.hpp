#pragma once

#include <cuda_runtime.h>

namespace sihps {

// RAII wrapper around a CUDA event. Used for point-to-point
// synchronization (waiting for one specific kernel/copy) and for
// wall-clock-independent GPU timing (elapsed_ms) -- prompt.md \S3.6
// prohibits blanket cudaDeviceSynchronize() in hot paths; this is the
// alternative.
class CudaEvent {
public:
    CudaEvent();
    ~CudaEvent();

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void synchronize() const;

    // Elapsed time between two previously-recorded, already-completed
    // events, in milliseconds. Caller must have synchronized `end` (or the
    // stream it was recorded on) before calling this.
    static float elapsed_ms(const CudaEvent& start, const CudaEvent& end);

    cudaEvent_t handle() const noexcept { return event_; }

private:
    cudaEvent_t event_;
};

} // namespace sihps
