
#pragma once
#include "nerve/config.hpp"
#include "nerve/core/rng/determinism_contract.hpp"

#include <cassert>
#include <cstddef>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace nerve::core
{

struct MemoryBlock
{
    void *ptr;
    std::size_t size;
    std::size_t alignment;
    bool in_use;
    constexpr MemoryBlock(void *p, std::size_t s, std::size_t a)
        : ptr(p)
        , size(s)
        , alignment(a)
        , in_use(true)
    {}
};

class MemoryPool
{
public:
    explicit MemoryPool(std::size_t pool_size = nerve::kDefaultMemoryPoolSize);
    explicit MemoryPool(const DeterminismContract &contract,
                        std::size_t pool_size = nerve::kDefaultMemoryPoolSize);
    ~MemoryPool();
    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;
    MemoryPool(MemoryPool &&other) noexcept;
    MemoryPool &operator=(MemoryPool &&other) noexcept;

    [[nodiscard]] void *allocate(std::size_t size,
                                 std::size_t alignment = alignof(std::max_align_t));
    [[nodiscard]] void *allocate(std::size_t size, std::size_t alignment,
                                 const DeterminismContract &contract);
    void deallocate(void *ptr, std::size_t size) noexcept;
    void deallocate(void *ptr, std::size_t size, const DeterminismContract &contract) noexcept;

    [[nodiscard]] std::size_t allocated() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t numBlocks() const noexcept;
    [[nodiscard]] double fragmentationRatio() const noexcept;
    [[nodiscard]] DeterminismMetadata getDeterminismMetadata() const;
    void defragment();
    void reset();
    void reset(const DeterminismContract &contract);

    [[nodiscard]] void *allocateAligned(std::size_t size, std::size_t alignment);
    [[nodiscard]] void *allocateAligned(std::size_t size, std::size_t alignment,
                                        const DeterminismContract &contract);
    [[nodiscard]] bool contains(void *ptr) const noexcept;
    [[nodiscard]] std::vector<MemoryBlock> getBlocks() const;
    [[nodiscard]] bool isDeterministic() const noexcept;
    void setDeterminismContract(const DeterminismContract &contract);

private:
    std::size_t pool_size_;
    std::vector<MemoryBlock> blocks_;
    std::size_t allocated_;
    mutable std::mutex mutex_;
    DeterminismContract determinism_contract_;
    DeterminismMetadata determinism_metadata_;
    bool is_deterministic_ = false;
    std::vector<void *> deterministic_allocations_;

#ifndef NDEBUG
    std::unordered_set<void *>
        live_allocations_; // Track live allocations for double-free detection
#endif

    void *allocateFromPool(std::size_t size, std::size_t alignment);
    void *allocateFromPoolDeterministic(std::size_t size, std::size_t alignment,
                                        const DeterminismContract &contract);
    void coalesce_free_blocks();
    std::size_t find_free_block(std::size_t size, std::size_t alignment) const;
    std::size_t find_free_block_deterministic(std::size_t size, std::size_t alignment);
    void updateDeterminismMetadata();
    void validateDeterminismConstraints(std::size_t size) const;
};

} // namespace nerve::core
