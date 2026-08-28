#include "CudaEvent.hpp"

#include "CudaCheck.hpp"

namespace sihps {

CudaEvent::CudaEvent() : event_(nullptr) {
    SIHPS_CUDA_CHECK(cudaEventCreate(&event_));
}

CudaEvent::~CudaEvent() {
    if (event_) cudaEventDestroy(event_);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_) cudaEventDestroy(event_);
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(cudaStream_t stream) {
    SIHPS_CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaEvent::synchronize() const {
    SIHPS_CUDA_CHECK(cudaEventSynchronize(event_));
}

float CudaEvent::elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float ms = 0.0f;
    SIHPS_CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
    return ms;
}

} // namespace sihps
