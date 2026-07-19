
#include "nerve/config.hpp"
#include "nerve/core/memory/memory_pool.hpp"
#include "nerve/core/rng/determinism_contract.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace nerve::core
{

constinit const std::size_t DEFAULT_BLOCK_RESERVE = 256;

namespace
{

[[nodiscard]] bool additionWouldExceed(std::size_t lhs, std::size_t rhs, std::size_t limit) noexcept
{
    return lhs > limit || rhs > limit - lhs;
}

[[nodiscard]] std::size_t contractLimitBytes(const DeterminismContract &contract) noexcept
{
    constexpr std::size_t bytes_per_mb = 1024ULL * 1024ULL;
    if (contract.max_memory_usage_mb > std::numeric_limits<std::size_t>::max() / bytes_per_mb)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return contract.max_memory_usage_mb * bytes_per_mb;
}

[[nodiscard]] std::size_t alignedAllocationSize(std::size_t size, std::size_t alignment)
{
    if (alignment == 0 || !std::has_single_bit(alignment))
    {
        throw std::invalid_argument("MemoryPool alignment must be a non-zero power of two");
    }
    if (alignment <= alignof(std::max_align_t))
    {
        return size;
    }
    const std::size_t padding = alignment - 1;
    if (size > std::numeric_limits<std::size_t>::max() - padding)
    {
        throw std::bad_alloc();
    }
    return (size + padding) & ~padding;
}

[[nodiscard]] bool isPointerAligned(void *ptr, std::size_t alignment) noexcept
{
    if (alignment == 0 || !std::has_single_bit(alignment))
    {
        return false;
    }
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

[[nodiscard]] void *allocateBlock(std::size_t allocation_size, std::size_t alignment)
{
    void *ptr = alignment <= alignof(std::max_align_t)
                    ? std::malloc(allocation_size)
                    :
#ifdef _WIN32
                    _aligned_malloc(allocation_size, alignment)
#else
                    std::aligned_alloc(alignment, allocation_size)
#endif
                    ;
    if (!ptr)
    {
        throw std::bad_alloc();
    }
    return ptr;
}

void freeBlock(void *ptr) noexcept
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

MemoryPool::MemoryPool(std::size_t pool_size)
    : pool_size_(pool_size)
    , allocated_(0)
    , is_deterministic_(false)
{
    blocks_.reserve(DEFAULT_BLOCK_RESERVE);
}

MemoryPool::MemoryPool(const DeterminismContract &contract, std::size_t pool_size)
    : pool_size_(pool_size)
    , allocated_(0)
    , determinism_contract_(contract)
    , is_deterministic_(true)
{
    blocks_.reserve(DEFAULT_BLOCK_RESERVE);
    determinism_metadata_ = DeterminismMetadata{};
    determinism_metadata_.was_deterministic = true;
    determinism_metadata_.achieved_level = contract.level;
    determinism_metadata_.rng_seed_used = contract.rng_seed;
    if (!DeterminismEnforcer::canSatisfyContract(contract))
    {
        auto violations = DeterminismEnforcer::getContractViolations(contract);
        const std::string reason = violations.empty() ? "unspecified violation" : violations[0];
        throw std::runtime_error("Cannot satisfy determinism contract: " + reason);
    }
}

MemoryPool::~MemoryPool()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &block : blocks_)
    {
        if (block.ptr)
        {
            freeBlock(block.ptr);
        }
    }
}

MemoryPool::MemoryPool(MemoryPool &&other) noexcept
{
    std::lock_guard<std::mutex> lock(other.mutex_);
    pool_size_ = other.pool_size_;
    blocks_ = std::move(other.blocks_);
    allocated_ = other.allocated_;
    determinism_contract_ = other.determinism_contract_;
    determinism_metadata_ = other.determinism_metadata_;
    is_deterministic_ = other.is_deterministic_;
    deterministic_allocations_ = std::move(other.deterministic_allocations_);
#ifndef NDEBUG
    live_allocations_ = std::move(other.live_allocations_);
#endif
    other.pool_size_ = 0;
    other.allocated_ = 0;
    other.is_deterministic_ = false;
    other.blocks_.clear();
}

MemoryPool &MemoryPool::operator=(MemoryPool &&other) noexcept
{
    if (this != &other)
    {
        std::lock(mutex_, other.mutex_);
        std::lock_guard<std::mutex> lock1(mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.mutex_, std::adopt_lock);
        for (const auto &block : blocks_)
        {
            if (block.ptr)
            {
                freeBlock(block.ptr);
            }
        }
        pool_size_ = other.pool_size_;
        blocks_ = std::move(other.blocks_);
        allocated_ = other.allocated_;
        determinism_contract_ = other.determinism_contract_;
        determinism_metadata_ = other.determinism_metadata_;
        is_deterministic_ = other.is_deterministic_;
        deterministic_allocations_ = std::move(other.deterministic_allocations_);
#ifndef NDEBUG
        live_allocations_ = std::move(other.live_allocations_);
#endif
        other.pool_size_ = 0;
        other.allocated_ = 0;
        other.is_deterministic_ = false;
        other.blocks_.clear();
    }
    return *this;
}

void *MemoryPool::allocate(std::size_t size, std::size_t alignment)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (size == 0)
        return nullptr;
    const std::size_t allocation_size = alignedAllocationSize(size, alignment);
    std::size_t block_idx = find_free_block(size, alignment);
    if (block_idx != blocks_.size())
    {
        MemoryBlock &block = blocks_[block_idx];
        block.in_use = true;
        allocated_ += block.size;

#ifndef NDEBUG
        live_allocations_.insert(block.ptr);
#endif

        return block.ptr;
    }
    return allocateFromPool(allocation_size, alignment);
}

void *MemoryPool::allocate(std::size_t size, std::size_t alignment,
                           const DeterminismContract &contract)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (size == 0)
        return nullptr;
    const std::size_t allocation_size = alignedAllocationSize(size, alignment);
    if (!DeterminismEnforcer::canSatisfyContract(contract))
    {
        auto violations = DeterminismEnforcer::getContractViolations(contract);
        const std::string reason = violations.empty() ? "unspecified violation" : violations[0];
        throw std::runtime_error("Cannot satisfy determinism contract: " + reason);
    }
    if (additionWouldExceed(allocated_, allocation_size, contractLimitBytes(contract)) &&
        contract.fail_on_non_deterministic)
    {
        throw std::runtime_error("Memory allocation would exceed deterministic contract limits");
    }
    validateDeterminismConstraints(size);
    std::size_t block_idx = find_free_block_deterministic(size, alignment);
    if (block_idx != blocks_.size())
    {
        MemoryBlock &block = blocks_[block_idx];
        block.in_use = true;
        allocated_ += block.size;
        if (is_deterministic_)
        {
            deterministic_allocations_.push_back(block.ptr);
        }

#ifndef NDEBUG
        live_allocations_.insert(block.ptr);
#endif

        updateDeterminismMetadata();
        return block.ptr;
    }
    return allocateFromPoolDeterministic(allocation_size, alignment, contract);
}

void MemoryPool::deallocate(void *ptr, std::size_t size) noexcept
{
    if (!ptr || size == 0)
        return;
    std::lock_guard<std::mutex> lock(mutex_);

#ifndef NDEBUG
    const auto live = live_allocations_.find(ptr);
    const bool owned = live != live_allocations_.end();
    assert(owned && "double-free or invalid pointer in MemoryPool");
    if (!owned)
    {
        return;
    }
    live_allocations_.erase(live);
#endif

    for (auto &block : blocks_)
    {
        if (block.ptr == ptr)
        {
            block.in_use = false;
            allocated_ = block.size <= allocated_ ? allocated_ - block.size : 0;
            coalesce_free_blocks();
            return;
        }
    }
}

void MemoryPool::deallocate(void *ptr, std::size_t size,
                            const DeterminismContract &contract) noexcept
{
    (void)contract;
#ifndef NDEBUG
    assert(DeterminismEnforcer::canSatisfyContract(contract) &&
           "deallocation determinism contract is not satisfiable");
#endif
    if (!ptr || size == 0)
        return;
    std::lock_guard<std::mutex> lock(mutex_);

#ifndef NDEBUG
    const auto live = live_allocations_.find(ptr);
    const bool owned = live != live_allocations_.end();
    assert(owned && "double-free or invalid pointer in MemoryPool");
    if (!owned)
    {
        return;
    }
    live_allocations_.erase(live);
#endif

    for (auto &block : blocks_)
    {
        if (block.ptr == ptr)
        {
            block.in_use = false;
            allocated_ = block.size <= allocated_ ? allocated_ - block.size : 0;
            if (is_deterministic_)
            {
                auto it = std::find(deterministic_allocations_.begin(),
                                    deterministic_allocations_.end(), ptr);
                if (it != deterministic_allocations_.end())
                {
                    deterministic_allocations_.erase(it);
                }
            }
            coalesce_free_blocks();
            updateDeterminismMetadata();
            return;
        }
    }
}

std::size_t MemoryPool::allocated() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
}

std::size_t MemoryPool::capacity() const noexcept
{
    return pool_size_;
}

std::size_t MemoryPool::numBlocks() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size();
}

double MemoryPool::fragmentationRatio() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_size_ == 0)
        return 0.0;
    std::size_t total_free = 0;
    std::size_t max_free = 0;
    for (const auto &block : blocks_)
    {
        if (!block.in_use)
        {
            total_free += block.size;
            max_free = std::max(max_free, block.size);
        }
    }
    if (total_free == 0)
        return 0.0;
    return 1.0 - (static_cast<double>(max_free) / static_cast<double>(total_free));
}

void MemoryPool::defragment()
{
    std::lock_guard<std::mutex> lock(mutex_);
    coalesce_free_blocks();
}

void MemoryPool::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &block : blocks_)
    {
        if (block.ptr)
        {
            freeBlock(block.ptr);
        }
    }
    blocks_.clear();
    allocated_ = 0;

#ifndef NDEBUG
    live_allocations_.clear();
#endif
}

void *MemoryPool::allocateAligned(std::size_t size, std::size_t alignment)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (size == 0)
        return nullptr;
    const std::size_t allocation_size = alignedAllocationSize(size, alignment);
    if (additionWouldExceed(allocated_, allocation_size, pool_size_))
    {
        throw std::bad_alloc();
    }
    void *ptr = allocateBlock(allocation_size, alignment);
    blocks_.emplace_back(ptr, allocation_size, alignment);
    allocated_ += allocation_size;

#ifndef NDEBUG
    live_allocations_.insert(ptr);
#endif

    return ptr;
}

bool MemoryPool::contains(void *ptr) const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &block : blocks_)
    {
        if (block.ptr == ptr)
        {
            return true;
        }
    }
    return false;
}

std::vector<MemoryBlock> MemoryPool::getBlocks() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_;
}

void *MemoryPool::allocateFromPool(std::size_t size, std::size_t alignment)
{
    const std::size_t allocation_size = alignedAllocationSize(size, alignment);
    if (additionWouldExceed(allocated_, allocation_size, pool_size_))
    {
        throw std::bad_alloc();
    }
    void *ptr = allocateBlock(allocation_size, alignment);
    blocks_.emplace_back(ptr, allocation_size, alignment);
    allocated_ += allocation_size;

#ifndef NDEBUG
    live_allocations_.insert(ptr);
#endif

    return ptr;
}

void MemoryPool::coalesce_free_blocks()
{
    // Blocks are individually allocated, so adjacent addresses do not imply a
    // shared allocation that can be merged safely.
}

std::size_t MemoryPool::find_free_block(std::size_t size, std::size_t alignment) const
{
    for (std::size_t i = 0; i < blocks_.size(); ++i)
    {
        const auto &block = blocks_[i];
        if (!block.in_use && block.size >= size && isPointerAligned(block.ptr, alignment))
        {
            return i;
        }
    }
    return blocks_.size();
}

} // namespace nerve::core
