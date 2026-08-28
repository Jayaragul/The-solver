#pragma once

#include <cstddef>
#include <string>

namespace sihps {

struct DeviceInfo {
    int index;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    std::size_t total_global_mem_bytes;
    int multiprocessor_count;
};

// Device discovery/selection (prompt.md \S3.3). The CUDA Runtime API has
// no separate "context" handle distinct from "the currently selected
// device for this thread" -- so unlike the Driver API, there is no
// separate context object to RAII-wrap here.
class CudaDevice {
public:
    static int device_count();
    static DeviceInfo query(int index);

    // Selects `index` as the active device for this thread and returns its
    // queried info.
    static DeviceInfo select(int index);

    // Free/total VRAM on the currently selected device. Per
    // docs/architecture/MEMORY.md \S1: query this at runtime, never assume
    // the nameplate VRAM figure is actually available.
    static void memory_info(std::size_t& free_bytes, std::size_t& total_bytes);
};

} // namespace sihps
