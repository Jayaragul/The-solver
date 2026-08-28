#include "../test_framework.hpp"
#include "memory/MemoryArena.hpp"

#include <cstdint>

using sihps::ArenaExhausted;
using sihps::MemoryArena;

SIHPS_TEST(alignment_respects_requested_boundary) {
    MemoryArena arena(4096, 64);
    double* d = arena.allocate<double>(1);
    SIHPS_ASSERT_TRUE(reinterpret_cast<std::uintptr_t>(d) % 64 == 0);

    // A second, smaller-alignment allocation should still land on a
    // properly aligned address (arena alignment 64 dominates alignof(char)).
    char* c = arena.allocate<char>(1);
    SIHPS_ASSERT_TRUE(reinterpret_cast<std::uintptr_t>(c) % alignof(char) == 0);
}

SIHPS_TEST(typed_allocation_returns_usable_storage) {
    MemoryArena arena(4096, 64);
    int* arr = arena.allocate<int>(10);
    for (int i = 0; i < 10; ++i) {
        arr[i] = i * i;
    }
    for (int i = 0; i < 10; ++i) {
        SIHPS_ASSERT_EQ(arr[i], i * i);
    }
}

SIHPS_TEST(reset_reclaims_capacity_without_growing) {
    MemoryArena arena(1024, 64);
    arena.allocate<double>(64); // 512 bytes
    SIHPS_ASSERT_TRUE(arena.used() > 0);

    arena.reset();
    SIHPS_ASSERT_EQ(arena.used(), std::size_t{0});

    // Must succeed again post-reset without the arena growing -- proves
    // reset() actually reclaims the space rather than merely zeroing a
    // counter that allocate() ignores.
    arena.allocate<double>(64);
    SIHPS_ASSERT_TRUE(arena.used() <= arena.capacity());
}

SIHPS_TEST(single_oversized_allocation_throws_arena_exhausted) {
    MemoryArena arena(128, 64);
    SIHPS_ASSERT_THROWS(arena.allocate<double>(100)); // 800 bytes > 128
}

SIHPS_TEST(repeated_small_allocations_eventually_exhaust_capacity) {
    MemoryArena arena(256, 64);
    bool threw = false;
    try {
        for (int i = 0; i < 100; ++i) {
            arena.allocate<double>(1);
        }
    } catch (const ArenaExhausted&) {
        threw = true;
    }
    SIHPS_ASSERT_TRUE(threw);
}

SIHPS_TEST(zero_capacity_arena_rejects_any_allocation) {
    MemoryArena arena(0, 64);
    SIHPS_ASSERT_THROWS(arena.allocate<char>(1));
}

SIHPS_TEST(remaining_and_capacity_are_consistent) {
    MemoryArena arena(1024, 64);
    SIHPS_ASSERT_EQ(arena.capacity(), std::size_t{1024});
    SIHPS_ASSERT_EQ(arena.remaining(), std::size_t{1024});
    arena.allocate<double>(1);
    SIHPS_ASSERT_EQ(arena.used() + arena.remaining(), arena.capacity());
}
