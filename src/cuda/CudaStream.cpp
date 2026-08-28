#include "CudaStream.hpp"

#include "CudaCheck.hpp"

namespace sihps {

CudaStream::CudaStream() : stream_(nullptr) {
    SIHPS_CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaStream::~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() const {
    SIHPS_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

} // namespace sihps
