// GPUBatchMemoryManager implementation.
//
// The allocator serves aligned slices from a managed pool and reports
// exhaustion by returning nullptr.

#include "nerve/gpu/batch_manager.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace nerve::gpu
{
namespace
{

std::size_t saturatingAdd(const std::size_t lhs, const std::size_t rhs) noexcept
{
    constexpr std::size_t max_value = std::numeric_limits<std::size_t>::max();
    return rhs > max_value - lhs ? max_value : lhs + rhs;
}

} // namespace

bool GPUBatchMemoryManager::isPowerOfTwo(const std::size_t value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t GPUBatchMemoryManager::alignUp(const std::size_t value,
                                           const std::size_t alignment) noexcept
{
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

GPUBatchMemoryManager::GPUBatchMemoryManager(const GPUBatchConfig &config)
    : config_(config)
{
    if (config_.memory_pool_size == 0)
    {
        return;
    }
    if (cudaMallocManaged(&memory_pool_, config_.memory_pool_size) == cudaSuccess)
    {
        pool_size_ = config_.memory_pool_size;
        pool_offset_ = 0;
    }
    else
    {
        memory_pool_ = nullptr;
        pool_size_ = 0;
        pool_offset_ = 0;
    }
}

GPUBatchMemoryManager::~GPUBatchMemoryManager()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const AllocationRecord &record : tracked_allocations_)
    {
        if (record.ptr != nullptr)
        {
            if (record.pinned)
            {
                cudaFreeHost(record.ptr);
            }
            else
            {
                cudaFree(record.ptr);
            }
        }
    }
    tracked_allocations_.clear();
    if (memory_pool_ != nullptr)
    {
        cudaFree(memory_pool_);
        memory_pool_ = nullptr;
    }
}

void *GPUBatchMemoryManager::allocate(std::size_t size, std::size_t alignment)
{
    if (size == 0)
    {
        return nullptr;
    }
    if (!isPowerOfTwo(alignment))
    {
        alignment = 256;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t aligned_size = alignUp(size, alignment);
    if (aligned_size == 0 ||
        aligned_size > std::numeric_limits<std::size_t>::max() - total_allocated_)
    {
        return nullptr;
    }

    if (memory_pool_ != nullptr && pool_offset_ <= pool_size_ &&
        aligned_size <= pool_size_ - pool_offset_)
    {
        char *base = static_cast<char *>(memory_pool_);
        void *ptr = base + pool_offset_;
        pool_offset_ += aligned_size;
        total_allocated_ += aligned_size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);
        return ptr;
    }

    return nullptr;
}

void *GPUBatchMemoryManager::allocatePinned(std::size_t size, std::size_t alignment)
{
    if (size == 0 || !config_.enable_pinned_memory)
    {
        return nullptr;
    }
    if (!isPowerOfTwo(alignment))
    {
        alignment = 256;
    }

    const std::size_t aligned_size = alignUp(size, alignment);
    if (aligned_size == 0)
    {
        return nullptr;
    }
    void *ptr = nullptr;
    if (cudaHostAlloc(&ptr, aligned_size, cudaHostAllocDefault) != cudaSuccess)
    {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (aligned_size > std::numeric_limits<std::size_t>::max() - total_allocated_)
    {
        cudaFreeHost(ptr);
        return nullptr;
    }
    try
    {
        tracked_allocations_.push_back(AllocationRecord{
            .ptr = ptr,
            .size = aligned_size,
            .pinned = true,
        });
    }
    catch (...)
    {
        cudaFreeHost(ptr);
        return nullptr;
    }
    total_allocated_ += aligned_size;
    peak_allocated_ = std::max(peak_allocated_, total_allocated_);
    return ptr;
}

void GPUBatchMemoryManager::deallocate(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (memory_pool_ != nullptr)
    {
        const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(memory_pool_);
        const std::uintptr_t target = reinterpret_cast<std::uintptr_t>(ptr);
        if (target >= start && target - start < pool_size_)
        {
            return;
        }
    }

    for (auto it = tracked_allocations_.begin(); it != tracked_allocations_.end(); ++it)
    {
        if (it->ptr != ptr)
        {
            continue;
        }
        if (it->pinned)
        {
            cudaFreeHost(ptr);
        }
        else
        {
            cudaFree(ptr);
        }
        total_freed_ = saturatingAdd(total_freed_, it->size);
        total_allocated_ -= std::min(total_allocated_, it->size);
        tracked_allocations_.erase(it);
        return;
    }

    fragmentation_events_ = saturatingAdd(fragmentation_events_, 1);
}

void GPUBatchMemoryManager::deallocatePinned(void *ptr)
{
    deallocate(ptr);
}

void GPUBatchMemoryManager::resetPool()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pool_offset_ = 0;
    total_allocated_ = 0;
    for (const AllocationRecord &record : tracked_allocations_)
    {
        if (record.ptr != nullptr)
        {
            if (record.pinned)
            {
                cudaFreeHost(record.ptr);
            }
            else
            {
                cudaFree(record.ptr);
            }
            total_freed_ = saturatingAdd(total_freed_, record.size);
        }
    }
    tracked_allocations_.clear();
}

void GPUBatchMemoryManager::reset()
{
    resetPool();
}

void GPUBatchMemoryManager::resetMemoryStats()
{
    std::lock_guard<std::mutex> lock(mutex_);
    peak_allocated_ = total_allocated_;
    total_freed_ = 0;
    fragmentation_events_ = 0;
}

std::size_t GPUBatchMemoryManager::getTotalAllocated() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

std::size_t GPUBatchMemoryManager::getPeakAllocated() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_allocated_;
}

std::size_t GPUBatchMemoryManager::getPoolUsed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_offset_;
}

std::size_t GPUBatchMemoryManager::getAllocatedSize() const
{
    return getTotalAllocated();
}

std::size_t GPUBatchMemoryManager::get_free_size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_size_ < pool_offset_)
    {
        return 0;
    }
    return pool_size_ - pool_offset_;
}

GPUBatchMemoryManager::MemoryStats GPUBatchMemoryManager::getMemoryStats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return MemoryStats{
        .total_allocated = saturatingAdd(total_allocated_, total_freed_),
        .total_freed = total_freed_,
        .current_allocated = total_allocated_,
        .peak_allocated = peak_allocated_,
        .fragmentation_count = fragmentation_events_,
    };
}

} // namespace nerve::gpu
