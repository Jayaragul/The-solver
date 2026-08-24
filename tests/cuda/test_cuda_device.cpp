#include "../test_framework.hpp"
#include "cuda/CudaDevice.hpp"

using sihps::CudaDevice;

SIHPS_TEST(at_least_one_cuda_device_is_visible) {
    SIHPS_ASSERT_TRUE(CudaDevice::device_count() >= 1);
}

SIHPS_TEST(device_query_returns_sane_properties) {
    auto info = CudaDevice::query(0);
    SIHPS_ASSERT_TRUE(!info.name.empty());
    SIHPS_ASSERT_TRUE(info.compute_capability_major >= 1);
    SIHPS_ASSERT_TRUE(info.total_global_mem_bytes > 0);
    SIHPS_ASSERT_TRUE(info.multiprocessor_count > 0);
}

SIHPS_TEST(device_select_matches_query) {
    auto selected = CudaDevice::select(0);
    auto queried = CudaDevice::query(0);
    SIHPS_ASSERT_EQ(selected.name, queried.name);
    SIHPS_ASSERT_EQ(selected.compute_capability_major, queried.compute_capability_major);
}

SIHPS_TEST(memory_info_reports_free_less_than_or_equal_to_total) {
    CudaDevice::select(0);
    std::size_t free_bytes = 0, total_bytes = 0;
    CudaDevice::memory_info(free_bytes, total_bytes);
    SIHPS_ASSERT_TRUE(total_bytes > 0);
    SIHPS_ASSERT_TRUE(free_bytes <= total_bytes);
}
