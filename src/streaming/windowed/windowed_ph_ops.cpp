
#include "nerve/streaming/windowed_ph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace nerve::streaming
{

namespace
{

// NUMA Memory Pool Constants
constexpr std::size_t POOL_DEFAULT_SIZE_MB = 1;
constexpr std::size_t POOL_DEFAULT_BLOCK_SIZE = 4096; // 4KB default block

// Memory Unit Constants
constexpr std::size_t BYTES_PER_MB = 1024ULL * 1024ULL;

std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0) [[unlikely]]
    {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) [[likely]]
    {
        return value;
    }
    return value + (alignment - remainder);
}

void validatePair(const Pair &pair)
{
    if (!std::isfinite(pair.birth) || std::isnan(pair.death) ||
        pair.death == -std::numeric_limits<double>::infinity() || pair.dimension < 0 ||
        (!pair.isInfinite() && (!std::isfinite(pair.death) || pair.death < pair.birth)))
    {
        throw std::invalid_argument("windowed persistence diagram contains an invalid pair");
    }
}

double pairLifetime(const Pair &pair)
{
    validatePair(pair);
    if (pair.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    return pair.death - pair.birth;
}

} // namespace

NUMAMemoryPool::NUMAMemoryPool(const PoolConfig &config)
    : config_(config)
    , allocated_bytes_(0)
    , current_numa_node_(0)
{
    if (config_.pool_size_mb == 0) [[unlikely]]
    {
        config_.pool_size_mb = POOL_DEFAULT_SIZE_MB;
    }
    if (config_.block_size_bytes == 0) [[unlikely]]
    {
        config_.block_size_bytes = POOL_DEFAULT_BLOCK_SIZE;
    }
    if (config_.preallocate)
    {
        initializePool();
    }
}

NUMAMemoryPool::NUMAMemoryPool()
    : NUMAMemoryPool(PoolConfig{})
{}
NUMAMemoryPool::~NUMAMemoryPool()
{
    reset();
}

void NUMAMemoryPool::initializePool()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!memory_blocks_.empty()) [[likely]]
    {
        return;
    }
    const std::size_t total_bytes = config_.pool_size_mb * BYTES_PER_MB;
    const std::size_t block_size = alignUp(config_.block_size_bytes, alignof(std::max_align_t));
    const std::size_t num_blocks = std::max<std::size_t>(1, total_bytes / block_size);

    memory_blocks_.reserve(num_blocks);
    block_used_.assign(num_blocks, false);
    for (std::size_t i = 0; i < num_blocks; ++i)
    {
        void *block = std::aligned_alloc(alignof(std::max_align_t), block_size);
        if (block == nullptr) [[unlikely]]
        {
            throw std::runtime_error("NUMAMemoryPool failed to preallocate block");
        }
        memory_blocks_.push_back(block);
    }
}

void *NUMAMemoryPool::allocate(std::size_t size)
{
    if (size == 0) [[unlikely]]
    {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (memory_blocks_.empty() && config_.preallocate)
    {
        initializePool();
    }
    if (size <= config_.block_size_bytes && !memory_blocks_.empty())
    {
        return allocateFromPool(size);
    }

    const std::size_t aligned = alignUp(size, alignof(std::max_align_t));
    void *ptr = std::aligned_alloc(alignof(std::max_align_t), aligned);
    if (ptr != nullptr) [[likely]]
    {
        allocated_bytes_ += aligned;
    }
    return ptr;
}

void NUMAMemoryPool::deallocate(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    bool from_pool = false;
    for (std::size_t i = 0; i < memory_blocks_.size(); ++i)
    {
        if (memory_blocks_[i] == ptr)
        {
            from_pool = true;
            if (block_used_[i])
            {
                block_used_[i] = false;
                if (allocated_bytes_ >= config_.block_size_bytes)
                {
                    allocated_bytes_ -= config_.block_size_bytes;
                }
                else
                {
                    allocated_bytes_ = 0;
                }
            }
            break;
        }
    }
    if (!from_pool)
    {
        std::free(ptr);
    }
}

void *NUMAMemoryPool::allocateFromPool(std::size_t /*size*/)
{
    for (std::size_t i = 0; i < memory_blocks_.size(); ++i)
    {
        if (!block_used_[i])
        {
            block_used_[i] = true;
            allocated_bytes_ += config_.block_size_bytes;
            return memory_blocks_[i];
        }
    }
    return nullptr;
}

void NUMAMemoryPool::deallocateToPool(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (std::size_t i = 0; i < memory_blocks_.size(); ++i)
    {
        if (memory_blocks_[i] == ptr && block_used_[i])
        {
            block_used_[i] = false;
            if (allocated_bytes_ >= config_.block_size_bytes)
            {
                allocated_bytes_ -= config_.block_size_bytes;
            }
            else
            {
                allocated_bytes_ = 0;
            }
            break;
        }
    }
}

void NUMAMemoryPool::reset()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (void *block : memory_blocks_)
    {
        std::free(block);
    }
    memory_blocks_.clear();
    block_used_.clear();
    allocated_bytes_ = 0;
}

std::size_t NUMAMemoryPool::getAllocatedBytes() const
{
    return allocated_bytes_;
}

std::size_t NUMAMemoryPool::get_free_bytes() const
{
    const std::size_t total = config_.pool_size_mb * 1024ULL * 1024ULL;
    return total > allocated_bytes_ ? total - allocated_bytes_ : 0;
}

std::size_t NUMAMemoryPool::getPoolSize() const
{
    return config_.pool_size_mb * 1024ULL * 1024ULL;
}
void NUMAMemoryPool::bindToNumaNode(int node)
{
    current_numa_node_ = node;
}
int NUMAMemoryPool::getCurrentNumaNode() const
{
    return current_numa_node_;
}

PartialRecomputeHeuristic::PartialRecomputeHeuristic(const HeuristicConfig &config)
    : config_(config)
{}
PartialRecomputeHeuristic::PartialRecomputeHeuristic()
    : PartialRecomputeHeuristic(HeuristicConfig{})
{}

PartialRecomputeHeuristic::RecomputeStrategy
PartialRecomputeHeuristic::determineStrategy(const std::vector<algebra::Simplex> &added_simplices,
                                             const std::vector<algebra::Simplex> &removed_simplices,
                                             const Diagram &current_diagram) const
{
    if (config_.strategy == RecomputeStrategy::FULL_RECOMPUTE ||
        config_.strategy == RecomputeStrategy::INCREMENTAL_UPDATE ||
        config_.strategy == RecomputeStrategy::HYBRID)
    {
        return config_.strategy;
    }

    const std::size_t total_changes = added_simplices.size() + removed_simplices.size();
    if (total_changes == 0)
    {
        return RecomputeStrategy::INCREMENTAL_UPDATE;
    }

    const double change_magnitude = computeChangeMagnitude(added_simplices, removed_simplices);
    const double complexity = computeComplexityScore(current_diagram);

    if (shouldUseIncremental(added_simplices) && removed_simplices.empty())
    {
        return RecomputeStrategy::INCREMENTAL_UPDATE;
    }
    if (shouldUseHybrid(current_diagram.count(), change_magnitude) && complexity < 5000.0)
    {
        return RecomputeStrategy::HYBRID;
    }
    return RecomputeStrategy::FULL_RECOMPUTE;
}

double PartialRecomputeHeuristic::computeChangeMagnitude(
    const std::vector<algebra::Simplex> &added_simplices,
    const std::vector<algebra::Simplex> &removed_simplices) const
{
    double weighted_change = 0.0;
    for (const auto &simplex : added_simplices)
    {
        weighted_change += static_cast<double>(simplex.dimension() + 1);
    }
    for (const auto &simplex : removed_simplices)
    {
        weighted_change += static_cast<double>(simplex.dimension() + 1);
    }
    const double normalization =
        static_cast<double>(std::max<std::size_t>(1, config_.min_recompute_size));
    return weighted_change / normalization;
}

double PartialRecomputeHeuristic::estimateRecomputeCost(RecomputeStrategy strategy,
                                                        std::size_t window_size) const
{
    const double n = static_cast<double>(std::max<std::size_t>(1, window_size));
    switch (strategy)
    {
        case RecomputeStrategy::FULL_RECOMPUTE:
            return n * n;
        case RecomputeStrategy::INCREMENTAL_UPDATE:
            return n * std::log2(n + 1.0);
        case RecomputeStrategy::HYBRID:
            return (n * std::log2(n + 1.0)) + (0.1 * n * n);
        case RecomputeStrategy::ADAPTIVE:
        default:
            return n * n;
    }
}

double PartialRecomputeHeuristic::estimateUpdateAccuracy(RecomputeStrategy strategy) const
{
    switch (strategy)
    {
        case RecomputeStrategy::FULL_RECOMPUTE:
            return 1.0;
        case RecomputeStrategy::HYBRID:
            return 0.995;
        case RecomputeStrategy::INCREMENTAL_UPDATE:
            return 0.99;
        case RecomputeStrategy::ADAPTIVE:
        default:
            return 1.0;
    }
}

void PartialRecomputeHeuristic::setConfig(const HeuristicConfig &config)
{
    config_ = config;
}
PartialRecomputeHeuristic::HeuristicConfig PartialRecomputeHeuristic::getConfig() const
{
    return config_;
}

bool PartialRecomputeHeuristic::shouldUseIncremental(
    const std::vector<algebra::Simplex> &changes) const
{
    return changes.size() <= config_.min_recompute_size;
}

bool PartialRecomputeHeuristic::shouldUseHybrid(std::size_t window_size,
                                                double change_magnitude) const
{
    return window_size > config_.min_recompute_size &&
           change_magnitude < std::max(0.0, config_.change_threshold);
}

double PartialRecomputeHeuristic::computeComplexityScore(const Diagram &diagram) const
{
    if (diagram.count() == 0)
    {
        return 0.0;
    }
    double score = 0.0;
    for (const auto &pair : diagram.getPairs())
    {
        const double lifetime = pairLifetime(pair);
        score += std::isfinite(lifetime) ? lifetime : 10.0;
    }
    return score / static_cast<double>(diagram.count());
}

} // namespace nerve::streaming
