#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace sihps {

// Thrown when an allocation would exceed the arena's fixed capacity.
// This is a hard failure by design (docs/architecture/MEMORY.md): a solver
// that silently degrades on allocation failure violates the same
// numerical-reliability mandate as one that silently accepts an unverified
// result (docs/architecture/NUMERICS.md \S6).
struct ArenaExhausted : std::bad_alloc {
    ArenaExhausted(std::size_t requested_bytes, std::size_t remaining_bytes) noexcept
        : requested_bytes(requested_bytes), remaining_bytes(remaining_bytes) {}
    const char* what() const noexcept override { return "MemoryArena: capacity exhausted"; }
    std::size_t requested_bytes;
    std::size_t remaining_bytes;
};

// Monotonic bump allocator over a single fixed-size, fixed-alignment block.
//
// Design invariants (docs/architecture/MEMORY.md \S3.1, prompt.md \S3.1):
//   - The entire backing block is allocated once, at construction.
//   - allocate<T>() never calls malloc/new; it only advances an offset.
//   - reset() is O(1) and never fragments, since it just rewinds the offset.
//   - Not thread-safe by design: v1's solve loop is single-threaded
//     (docs/architecture/MILP.md \S4), so no synchronization is paid for a
//     property nothing yet needs. A future parallel-B&B milestone would
//     give each worker thread its own arena rather than share one.
//
// Not copyable or movable: an arena's identity is tied to one fixed backing
// allocation for its whole lifetime; callers that need to resize construct
// a new arena rather than mutate an existing one in place.
class MemoryArena {
public:
    explicit MemoryArena(std::size_t capacity_bytes, std::size_t alignment = 64);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

    // Allocates `count` objects of T, aligned to at least max(alignof(T),
    // the arena's configured alignment). Throws ArenaExhausted if doing so
    // would exceed the arena's fixed capacity.
    template <typename T>
    T* allocate(std::size_t count) {
        static_assert(std::is_trivially_destructible_v<T>,
            "MemoryArena never runs destructors on reset(); T must be trivially destructible.");
        const std::size_t bytes = count * sizeof(T);
        const std::size_t align = alignment_ > alignof(T) ? alignment_ : alignof(T);
        return static_cast<T*>(allocate_raw(bytes, align));
    }

    // O(1) bump-pointer reset. Does not zero memory and does not run
    // destructors (see static_assert above): callers that need destructor
    // semantics must not store non-trivial types in this arena.
    void reset() noexcept { offset_ = 0; }

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t used() const noexcept { return offset_; }
    std::size_t remaining() const noexcept { return capacity_ - offset_; }

private:
    void* allocate_raw(std::size_t bytes, std::size_t align);

    std::byte* base_;
    std::size_t capacity_;
    std::size_t offset_;
    std::size_t alignment_;
};

} // namespace sihps
