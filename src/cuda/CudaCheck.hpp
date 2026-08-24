#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace sihps {

struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(std::string("CUDA error: ") + cudaGetErrorString(err) + " (" + expr +
                         ") at " + file + ":" + std::to_string(line));
    }
}

inline void cusparse_check(cusparseStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw CudaError(std::string("cuSPARSE error: ") + cusparseGetErrorString(status) +
                         " (" + expr + ") at " + file + ":" + std::to_string(line));
    }
}

} // namespace sihps

// prompt.md \S3.3/\S3.10: every CUDA/cuSPARSE call is error-checked, never
// assumed to succeed.
#define SIHPS_CUDA_CHECK(expr) ::sihps::cuda_check((expr), #expr, __FILE__, __LINE__)
#define SIHPS_CUSPARSE_CHECK(expr) ::sihps::cusparse_check((expr), #expr, __FILE__, __LINE__)
