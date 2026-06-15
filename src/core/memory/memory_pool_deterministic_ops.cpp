
#include "nerve/core/memory/memory_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace nerve::core
{

constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;

namespace
{

[[nodiscard]] size_t contractLimitBytes(const DeterminismContract &contract) noexcept
{
    if (contract.max_memory_usage_mb > std::numeric_limits<size_t>::max() / BYTES_PER_MB)
    {
        return std::numeric_limits<size_t>::max();
    }
    return contract.max_memory_usage_mb * BYTES_PER_MB;
}

[[nodiscard]] bool additionWouldExceed(size_t lhs, size_t rhs, size_t limit) noexcept
{
    return lhs > limit || rhs > limit - lhs;
}

} // namespace

[[nodiscard]] DeterminismMetadata MemoryPool::getDeterminismMetadata() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return determinism_metadata_;
}

[[nodiscard]] bool MemoryPool::isDeterministic() const noexcept
{
    return is_deterministic_;
}

void MemoryPool::setDeterminismContract(const DeterminismContract &contract)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!DeterminismEnforcer::canSatisfyContract(contract))
    {
        auto violations = DeterminismEnforcer::getContractViolations(contract);
        const std::string reason = violations.empty() ? "unspecified violation" : violations[0];
        throw std::runtime_error("Cannot satisfy determinism contract: " + reason);
    }
    determinism_contract_ = contract;
    is_deterministic_ = true;
    determinism_metadata_.was_deterministic = true;
    determinism_metadata_.achieved_level = contract.level;
    determinism_metadata_.rng_seed_used = contract.rng_seed;
    updateDeterminismMetadata();
}

void MemoryPool::reset(const DeterminismContract &contract)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &block : blocks_)
    {
        if (block.ptr)
        {
            std::free(block.ptr);
        }
    }
    blocks_.clear();
    allocated_ = 0;
    deterministic_allocations_.clear();

#ifndef NDEBUG
    live_allocations_.clear();
#endif

    determinism_contract_ = contract;
    is_deterministic_ = true;
    determinism_metadata_ = DeterminismMetadata{};
    determinism_metadata_.was_deterministic = true;
    determinism_metadata_.achieved_level = contract.level;
    determinism_metadata_.rng_seed_used = contract.rng_seed;
}

void *MemoryPool::allocateAligned(std::size_t size, std::size_t alignment,
                                  const DeterminismContract &contract)
{
    return allocate(size, alignment, contract);
}

void *MemoryPool::allocateFromPoolDeterministic(std::size_t size, std::size_t alignment,
                                                const DeterminismContract &contract)
{
    if (additionWouldExceed(allocated_, size, contractLimitBytes(contract)) &&
        contract.fail_on_non_deterministic)
    {
        throw std::runtime_error("Memory allocation would exceed deterministic contract limits");
    }
    validateDeterminismConstraints(size);
    if (additionWouldExceed(allocated_, size, pool_size_))
    {
        throw std::bad_alloc();
    }

    void *ptr = alignment <= alignof(std::max_align_t) ? std::malloc(size)
                                                       : std::aligned_alloc(alignment, size);
    if (!ptr)
    {
        throw std::bad_alloc();
    }

    blocks_.emplace_back(ptr, size, alignment);
    allocated_ += size;
    if (is_deterministic_)
    {
        deterministic_allocations_.push_back(ptr);
    }

#ifndef NDEBUG
    live_allocations_.insert(ptr);
#endif

    updateDeterminismMetadata();
    return ptr;
}

std::size_t MemoryPool::find_free_block_deterministic(std::size_t size, std::size_t alignment)
{
    for (std::size_t i = 0; i < blocks_.size(); ++i)
    {
        const auto &block = blocks_[i];
        if (!block.in_use && block.size >= size &&
            (reinterpret_cast<std::uintptr_t>(block.ptr) % alignment) == 0)
        {
            return i;
        }
    }
    return blocks_.size();
}

void MemoryPool::updateDeterminismMetadata()
{
    if (!is_deterministic_)
    {
        return;
    }

    determinism_metadata_.actual_memory_usage_mb = allocated_ / BYTES_PER_MB;
    if (determinism_contract_.start_time != std::chrono::steady_clock::time_point{})
    {
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - determinism_contract_.start_time);
        determinism_metadata_.actual_execution_time = duration;
    }

    if (allocated_ > contractLimitBytes(determinism_contract_))
    {
        determinism_metadata_.warnings.push_back("Memory usage exceeds contract limit");
    }
}

void MemoryPool::validateDeterminismConstraints(std::size_t size) const
{
    if (!is_deterministic_)
    {
        return;
    }

    if (additionWouldExceed(allocated_, size, contractLimitBytes(determinism_contract_)))
    {
        if (determinism_contract_.fail_on_non_deterministic)
        {
            throw std::runtime_error(
                "Memory allocation would exceed deterministic contract limits");
        }
    }

    if (determinism_contract_.max_execution_time.count() > 0)
    {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - determinism_contract_.start_time);
        if (elapsed > determinism_contract_.max_execution_time)
        {
            if (determinism_contract_.fail_on_non_deterministic)
            {
                throw std::runtime_error(
                    "Execution time would exceed deterministic contract limits");
            }
        }
    }
}

} // namespace nerve::core
