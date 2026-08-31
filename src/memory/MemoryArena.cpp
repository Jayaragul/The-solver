#include "MemoryArena.hpp"

#include <cstdlib>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace sihps {

MemoryArena::MemoryArena(std::size_t capacity_bytes, std::size_t alignment)
    : base_(nullptr), capacity_(capacity_bytes), offset_(0), alignment_(alignment) {
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
        throw std::invalid_argument("MemoryArena: alignment must be a nonzero power of two");
    }
    // std::aligned_alloc requires the allocation size to be a multiple of
    // the alignment; round up (a zero-capacity arena still gets one
    // alignment-sized block so it always has a valid, if useless, base_).
    const std::size_t rounded =
        ((capacity_ + alignment_ - 1) / alignment_) * alignment_;
    const std::size_t allocation_size = rounded == 0 ? alignment_ : rounded;
#if defined(_MSC_VER)
    void* p = _aligned_malloc(allocation_size, alignment_);
#else
    void* p = std::aligned_alloc(alignment_, allocation_size);
#endif
    if (!p) {
        throw std::bad_alloc();
    }
    base_ = static_cast<std::byte*>(p);
}

MemoryArena::~MemoryArena() {
#if defined(_MSC_VER)
    _aligned_free(base_);
#else
    std::free(base_);
#endif
}

void* MemoryArena::allocate_raw(std::size_t bytes, std::size_t align) {
    // Round the current offset up to `align`. `align` is always a power of
    // two here (either alignment_, validated in the constructor, or
    // alignof(T), which is always a power of two by the language).
    const std::size_t aligned_offset = (offset_ + (align - 1)) & ~(align - 1);
    const std::size_t new_offset = aligned_offset + bytes;
    if (new_offset > capacity_ || new_offset < aligned_offset /* overflow */) {
        throw ArenaExhausted(bytes, capacity_ - offset_);
    }
    offset_ = new_offset;
    return base_ + aligned_offset;
}

} // namespace sihps
