
#pragma once

#include <cstddef>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>

namespace nerve::gpu
{

/**
 * @brief Configuration for GPU batch-memory allocation.
 *
 * The manager pre-allocates a managed pool and serves aligned allocations
 * from that pool.
 */
struct GPUBatchConfig
{
    std::size_t max_batch_size = 64;
    std::size_t min_batch_size = 1;
    bool enable_mixed_precision = false;
    bool enable_tensor_cores = true;
    std::size_t max_streams = 4;
    bool enable_pinned_memory = true;
    std::size_t memory_pool_size = std::size_t{256} * 1024U * 1024U;
    bool enable_profiling = false;
};

/**
 * @brief Managed-memory pool for GPU batch pipelines.
 *
 * This class is intentionally minimal and production-safe:
 * - deterministic alignment from pool allocations
 * - explicit tracking for standalone pinned allocations
 * - non-throwing deallocation and predictable shutdown cleanup
 */
class GPUBatchMemoryManager
{
public:
    struct MemoryStats
    {
        std::size_t total_allocated = 0;
        std::size_t total_freed = 0;
        std::size_t current_allocated = 0;
        std::size_t peak_allocated = 0;
        std::size_t fragmentation_count = 0;
    };

    explicit GPUBatchMemoryManager(const GPUBatchConfig &config);
    ~GPUBatchMemoryManager();

    GPUBatchMemoryManager(const GPUBatchMemoryManager &) = delete;
    GPUBatchMemoryManager &operator=(const GPUBatchMemoryManager &) = delete;
    GPUBatchMemoryManager(GPUBatchMemoryManager &&) = delete;
    GPUBatchMemoryManager &operator=(GPUBatchMemoryManager &&) = delete;

    [[nodiscard]] void *allocate(std::size_t size, std::size_t alignment = 256);
    [[nodiscard]] void *allocatePinned(std::size_t size, std::size_t alignment = 256);
    void deallocate(void *ptr);
    void deallocatePinned(void *ptr);

    template <typename T>
    [[nodiscard]] T *allocateTyped(std::size_t count = 1)
    {
        static_assert(!std::is_void_v<T>, "allocateTyped requires non-void type");
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            return nullptr;
        }
        return static_cast<T *>(allocate(sizeof(T) * count, alignof(T)));
    }

    template <typename T>
    void deallocateTyped(T *ptr)
    {
        deallocate(static_cast<void *>(ptr));
    }

    void resetPool();
    void reset();
    void resetMemoryStats();

    [[nodiscard]] std::size_t getTotalAllocated() const;
    [[nodiscard]] std::size_t getPeakAllocated() const;
    [[nodiscard]] std::size_t getAllocatedSize() const;
    [[nodiscard]] std::size_t get_free_size() const;
    [[nodiscard]] MemoryStats getMemoryStats() const;
    [[nodiscard]] std::size_t getPoolSize() const noexcept { return pool_size_; }
    [[nodiscard]] std::size_t getPoolUsed() const;

private:
    struct AllocationRecord
    {
        void *ptr = nullptr;
        std::size_t size = 0;
        bool pinned = false;
    };

    [[nodiscard]] static bool isPowerOfTwo(std::size_t value) noexcept;
    [[nodiscard]] static std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept;

    GPUBatchConfig config_{};
    void *memory_pool_ = nullptr;
    std::size_t pool_size_ = 0;
    std::size_t pool_offset_ = 0;
    std::size_t total_allocated_ = 0;
    std::size_t peak_allocated_ = 0;
    std::size_t total_freed_ = 0;
    std::size_t fragmentation_events_ = 0;
    std::vector<AllocationRecord> tracked_allocations_;
    mutable std::mutex mutex_;
};

} // namespace nerve::gpu
