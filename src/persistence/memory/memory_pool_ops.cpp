// Memory Pool Implementation for Persistent Homology
// Arena-style allocation for fast, cache-aligned memory management

#include "nerve/persistence/memory/memory_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace nerve::persistence::mempool
{

// Arena Size Constants
constexpr size_t MB_16 = 16 * BYTES_PER_MB;
constexpr size_t MB_100 = 100 * BYTES_PER_MB;
constexpr size_t MB_128 = 128 * BYTES_PER_MB;
constexpr size_t MB_512 = 512 * BYTES_PER_MB;
constexpr size_t GB_1 = BYTES_PER_GB;

// Arena implementation
Arena::Arena(size_t capacity)
    : capacity_(capacity)
    , offset_(0)
    , owns_buffer_(true)
{
    // Allocate aligned to cache line
    buffer_ = static_cast<char *>(aligned_alloc(CACHE_LINE_SIZE, capacity));
    if (!buffer_)
    {
        throw std::bad_alloc();
    }
}

Arena::~Arena()
{
    if (owns_buffer_ && buffer_)
    {
        free(buffer_);
    }
}

Arena::Arena(Arena &&other) noexcept
    : buffer_(other.buffer_)
    , capacity_(other.capacity_)
    , offset_(other.offset_)
    , owns_buffer_(other.owns_buffer_)
{
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

Arena &Arena::operator=(Arena &&other) noexcept
{
    if (this != &other)
    {
        if (owns_buffer_ && buffer_)
        {
            free(buffer_);
        }
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        owns_buffer_ = other.owns_buffer_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void *Arena::allocate(size_t size)
{
    size_t aligned_size = alignToCacheLine(size);

    if (offset_ + aligned_size > capacity_)
    {
        return nullptr;
    }

    void *ptr = buffer_ + offset_;
    offset_ += aligned_size;

    return ptr;
}

bool Arena::contains(void *ptr) const
{
    char *char_ptr = static_cast<char *>(ptr);
    return char_ptr >= buffer_ && char_ptr < buffer_ + capacity_;
}

void Arena::reset()
{
    offset_ = 0;
}

// ColumnMemoryPool implementation
ColumnMemoryPool::ColumnMemoryPool(size_t initial_arena_size)
    : arena_size_(initial_arena_size)
{
    // Create initial arena
    arenas_.push_back(std::make_unique<Arena>(arena_size_));
}

ColumnMemoryPool::~ColumnMemoryPool() = default;

void *ColumnMemoryPool::allocate(size_t size)
{
    total_allocated_ += size;

    // Try free lists first (for small allocations)
    if (size <= SMALL_THRESHOLD)
    {
        num_small_allocs_++;

        if (!free_list_small_.empty())
        {
            void *ptr = free_list_small_.back();
            free_list_small_.pop_back();
            return ptr;
        }
    }
    else if (size <= MEDIUM_THRESHOLD)
    {
        if (!free_list_medium_.empty())
        {
            void *ptr = free_list_medium_.back();
            free_list_medium_.pop_back();
            return ptr;
        }
    }

    // Try to allocate from existing arenas
    void *ptr = allocateFromArena(size);
    if (ptr)
    {
        return ptr;
    }

    // Need new arena
    num_large_allocs_++;
    arenas_.push_back(std::make_unique<Arena>(std::max(arena_size_, alignToCacheLine(size))));

    return arenas_.back()->allocate(size);
}

void *ColumnMemoryPool::allocateFromArena(size_t size)
{
    // Try to allocate from existing arenas
    for (auto &arena : arenas_)
    {
        void *ptr = arena->allocate(size);
        if (ptr)
        {
            return ptr;
        }
    }
    return nullptr;
}

void *ColumnMemoryPool::allocateFromFreeList(size_t size)
{
    if (size <= SMALL_THRESHOLD && !free_list_small_.empty())
    {
        void *ptr = free_list_small_.back();
        free_list_small_.pop_back();
        return ptr;
    }
    if (size <= MEDIUM_THRESHOLD && !free_list_medium_.empty())
    {
        void *ptr = free_list_medium_.back();
        free_list_medium_.pop_back();
        return ptr;
    }
    return nullptr;
}

void ColumnMemoryPool::deallocate(void *ptr, size_t size)
{
    if (!ptr)
        return;

    // Check if pointer belongs to any arena
    for (auto &arena : arenas_)
    {
        if (arena->contains(ptr))
        {
            // Return to free list for reuse
            if (size <= SMALL_THRESHOLD)
            {
                free_list_small_.push_back(ptr);
            }
            else if (size <= MEDIUM_THRESHOLD)
            {
                free_list_medium_.push_back(ptr);
            }
            // Large allocations not tracked in free list
            return;
        }
    }

    // Pointer is not owned by this pool.
}

void *ColumnMemoryPool::allocateAligned(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        return allocate(size);
    }

    // For simplicity, allocate with max alignment
    void *ptr = allocate(size + alignment);
    if (!ptr)
        return nullptr;

    void *aligned_ptr = ptr;
    size_t available = size + alignment;
    return std::align(alignment, size, aligned_ptr, available);
}

std::vector<void *> ColumnMemoryPool::allocateBatch(const std::vector<size_t> &sizes)
{
    std::vector<void *> result;
    result.reserve(sizes.size());

    for (size_t size : sizes)
    {
        result.push_back(allocate(size));
    }

    return result;
}

void ColumnMemoryPool::deallocateBatch(const std::vector<void *> &ptrs,
                                       const std::vector<size_t> &sizes)
{
    for (size_t i = 0; i < ptrs.size(); ++i)
    {
        deallocate(ptrs[i], sizes[i]);
    }
}

void ColumnMemoryPool::reset()
{
    // Reset all arenas
    for (auto &arena : arenas_)
    {
        arena->reset();
    }

    // Clear free lists
    free_list_small_.clear();
    free_list_medium_.clear();

    // Reset counters
    total_allocated_ = 0;
    num_small_allocs_ = 0;
    num_large_allocs_ = 0;
}

void ColumnMemoryPool::shrinkToFit()
{
    // Remove empty arenas (except first)
    size_t i = 1;
    while (i < arenas_.size())
    {
        if (arenas_[i]->used() == 0)
        {
            using ArenaIteratorDiff = std::vector<std::unique_ptr<Arena>>::difference_type;
            arenas_.erase(std::next(arenas_.begin(), static_cast<ArenaIteratorDiff>(i)));
        }
        else
        {
            ++i;
        }
    }
}

ColumnMemoryPool::Stats ColumnMemoryPool::getStats() const
{
    Stats stats;
    stats.total_allocated = total_allocated_.load();
    stats.num_arenas = arenas_.size();
    stats.num_small_allocs = num_small_allocs_.load();
    stats.num_large_allocs = num_large_allocs_.load();

    stats.total_used = 0;
    double total_util = 0.0;
    for (const auto &arena : arenas_)
    {
        stats.total_used += arena->used();
        total_util += arena->utilization();
    }

    stats.average_utilization =
        arenas_.empty() ? 0.0 : total_util / static_cast<double>(arenas_.size());

    return stats;
}

ColumnMemoryPool &ColumnMemoryPool::instance()
{
    static ColumnMemoryPool pool;
    return pool;
}

Arena *ColumnMemoryPool::findArenaForPointer(void *ptr)
{
    for (auto &arena : arenas_)
    {
        if (arena->contains(ptr))
        {
            return arena.get();
        }
    }
    return nullptr;
}

// ScopedArena implementation
ScopedArena::ScopedArena(size_t capacity)
    : arena_(capacity)
{}

ScopedArena::~ScopedArena() = default;

void *ScopedArena::allocate(size_t size)
{
    return arena_.allocate(size);
}

void ScopedArena::reset()
{
    arena_.reset();
}

// Benchmark
MemoryPoolBenchmark benchmarkMemoryPool(const std::vector<size_t> &allocation_sizes, int iterations)
{
    if (allocation_sizes.empty() || iterations <= 0)
    {
        throw std::invalid_argument("memory pool benchmark inputs are invalid");
    }
    if (std::any_of(allocation_sizes.begin(), allocation_sizes.end(),
                    [](size_t size) { return size == 0; }))
    {
        throw std::invalid_argument("memory pool benchmark allocation sizes must be positive");
    }

    MemoryPoolBenchmark bench;

    // Standard malloc benchmark
    auto start_malloc = std::chrono::high_resolution_clock::now();
    std::vector<void *> ptrs;
    ptrs.reserve(allocation_sizes.size());

    for (int iter = 0; iter < iterations; ++iter)
    {
        ptrs.clear();
        for (size_t size : allocation_sizes)
        {
            ptrs.push_back(malloc(size));
        }
        for (void *ptr : ptrs)
        {
            free(ptr);
        }
    }

    auto end_malloc = std::chrono::high_resolution_clock::now();
    bench.malloc_time_ms =
        std::chrono::duration<double, std::milli>(end_malloc - start_malloc).count();

    // Memory pool benchmark
    ColumnMemoryPool pool;

    auto start_pool = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter)
    {
        ptrs.clear();
        for (size_t size : allocation_sizes)
        {
            ptrs.push_back(pool.allocate(size));
        }
        for (size_t i = 0; i < ptrs.size(); ++i)
        {
            pool.deallocate(ptrs[i], allocation_sizes[i]);
        }
    }

    auto end_pool = std::chrono::high_resolution_clock::now();
    bench.pool_time_ms = std::chrono::duration<double, std::milli>(end_pool - start_pool).count();

    bench.speedup = (std::isfinite(bench.malloc_time_ms) && std::isfinite(bench.pool_time_ms) &&
                     bench.malloc_time_ms >= 0.0 && bench.pool_time_ms > 0.0)
                        ? bench.malloc_time_ms / bench.pool_time_ms
                        : 1.0;
    bench.fragmentation_malloc = 0; // Would need external tool
    bench.fragmentation_pool = 0;

    return bench;
}

// Configuration
MemoryPoolConfig getOptimalMemoryPoolConfig(size_t expected_num_columns, size_t avg_column_size)
{
    MemoryPoolConfig config;

    size_t total_expected = expected_num_columns * avg_column_size;

    // Choose arena size based on expected usage
    if (total_expected < MB_100)
    {                                      // < 100MB
        config.initial_arena_size = MB_16; // 16MB
    }
    else if (total_expected < GB_1)
    {                                       // < 1GB
        config.initial_arena_size = MB_128; // 128MB
    }
    else
    {
        config.initial_arena_size = MB_512; // 512MB
    }

    config.use_free_lists = true;
    config.track_stats = true;
    config.auto_shrink = false;

    return config;
}

// Estimate
MemoryPoolEstimate estimateMemoryPoolBenefit(size_t num_allocations, size_t avg_allocation_size)
{
    MemoryPoolEstimate estimate;

    if (num_allocations < 1000)
    {
        // Not enough allocations to justify overhead
        estimate.allocation_speedup = 1.0;
        estimate.memory_overhead_bytes = 0;
        estimate.recommended = false;
        return estimate;
    }

    // Estimate speedup based on allocation count
    if (num_allocations < 10000)
    {
        estimate.allocation_speedup = 1.5;
    }
    else if (num_allocations < 100000)
    {
        estimate.allocation_speedup = 2.0;
    }
    else
    {
        estimate.allocation_speedup = 3.0;
    }
    estimate.memory_overhead_bytes = std::max<size_t>(avg_allocation_size, sizeof(void *)) *
                                     std::min<size_t>(num_allocations, 1024);

    // Arena overhead (one arena minimum)
    estimate.memory_overhead_bytes = MB_16; // 16MB

    // Recommended if many allocations
    estimate.recommended = (num_allocations > 5000);

    return estimate;
}

} // namespace nerve::persistence::mempool
