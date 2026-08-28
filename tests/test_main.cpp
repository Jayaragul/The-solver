#include "test_framework.hpp"

int main() {
    int passed = 0;
    int failed = 0;
    for (auto& tc : sihps::test::registry()) {
        try {
            tc.fn();
            std::printf("[PASS] %s\n", tc.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[FAIL] %s: %s\n", tc.name.c_str(), e.what());
            ++failed;
        } catch (...) {
            std::printf("[FAIL] %s: unknown exception\n", tc.name.c_str());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed, %zu total\n", passed, failed,
                 sihps::test::registry().size());
    return failed == 0 ? 0 : 1;
}
