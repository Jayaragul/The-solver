#include "CudaDevice.hpp"

#include "CudaCheck.hpp"

#include <cuda_runtime.h>

namespace sihps {

int CudaDevice::device_count() {
    int count = 0;
    SIHPS_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo CudaDevice::query(int index) {
    cudaDeviceProp prop{};
    SIHPS_CUDA_CHECK(cudaGetDeviceProperties(&prop, index));
    DeviceInfo info;
    info.index = index;
    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_global_mem_bytes = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    return info;
}

DeviceInfo CudaDevice::select(int index) {
    SIHPS_CUDA_CHECK(cudaSetDevice(index));
    return query(index);
}

void CudaDevice::memory_info(std::size_t& free_bytes, std::size_t& total_bytes) {
    SIHPS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

} // namespace sihps
