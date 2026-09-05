#include "test_framework.hpp"

int main(int argc, char** argv) {
    if (argc > 2) return 2;
    const std::string filter = argc == 2 ? argv[1] : "";
    int passed = 0;
    int failed = 0;
    for (auto& tc : sihps::test::registry()) {
        if (tc.name.find(filter) == std::string::npos) continue;
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
    if (passed + failed == 0) {
        std::fprintf(stderr, "No tests matched: %s\n", filter.c_str());
        return 2;
    }
    std::printf("\n%d passed, %d failed, %d total\n", passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
