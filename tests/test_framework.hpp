#pragma once

// Minimal self-registering test harness. Deliberately not a third-party
// framework (GoogleTest/Catch2): this project's dependency discipline
// (prompt.md \S1) extends to tooling, not just the solver core, and a
// ~60-line harness is enough for what these tests need.

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sihps::test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        registry().push_back({std::move(name), std::move(fn)});
    }
};

struct AssertionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace sihps::test

#define SIHPS_TEST(name)                                                     \
    static void sihps_test_##name();                                         \
    static ::sihps::test::Registrar sihps_registrar_##name(                  \
        #name, sihps_test_##name);                                           \
    static void sihps_test_##name()

#define SIHPS_ASSERT_TRUE(cond)                                              \
    do {                                                                     \
        if (!(cond)) {                                                      \
            throw ::sihps::test::AssertionFailure(                          \
                std::string("ASSERT_TRUE failed: ") + #cond + " at " +       \
                __FILE__ ":" + std::to_string(__LINE__));                    \
        }                                                                    \
    } while (0)

#define SIHPS_ASSERT_EQ(a, b)                                                \
    do {                                                                     \
        if (!((a) == (b))) {                                                \
            throw ::sihps::test::AssertionFailure(                          \
                std::string("ASSERT_EQ failed: ") + #a + " != " + #b +       \
                " at " __FILE__ ":" + std::to_string(__LINE__));             \
        }                                                                    \
    } while (0)

#define SIHPS_ASSERT_NEAR(a, b, tol)                                         \
    do {                                                                     \
        double sihps_da = static_cast<double>(a);                           \
        double sihps_db = static_cast<double>(b);                           \
        if (std::fabs(sihps_da - sihps_db) > (tol)) {                       \
            throw ::sihps::test::AssertionFailure(                          \
                std::string("ASSERT_NEAR failed: ") + #a + " vs " + #b +     \
                " at " __FILE__ ":" + std::to_string(__LINE__));             \
        }                                                                    \
    } while (0)

#define SIHPS_ASSERT_THROWS(expr)                                            \
    do {                                                                     \
        bool sihps_threw = false;                                            \
        try {                                                                \
            expr;                                                            \
        } catch (...) {                                                     \
            sihps_threw = true;                                              \
        }                                                                    \
        if (!sihps_threw) {                                                 \
            throw ::sihps::test::AssertionFailure(                          \
                std::string("ASSERT_THROWS failed: ") + #expr + " at " +     \
                __FILE__ ":" + std::to_string(__LINE__));                    \
        }                                                                    \
    } while (0)
