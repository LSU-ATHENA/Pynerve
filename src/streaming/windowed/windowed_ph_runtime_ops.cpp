
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/windowed_ph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::streaming
{

namespace
{

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

Diagram diagramFromPairs(const std::vector<Pair> &pairs)
{
    Diagram diagram;
    for (const auto &pair : pairs)
    {
        diagram.addPair(pair);
    }
    return diagram;
}

} // namespace

Diagram AcceleratedWindowedPH::computeTopLifetimeTruncation()
{
    std::vector<Pair> pairs(current_diagram_.getPairs().begin(), current_diagram_.getPairs().end());
    if (pairs.size() <= 1)
    {
        return current_diagram_;
    }
    std::ranges::sort(pairs, std::greater{}, [](const Pair &p) { return pairLifetime(p); });
    const std::size_t keep =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(pairs.size())));
    pairs.resize(keep);
    return diagramFromPairs(pairs);
}

Diagram AcceleratedWindowedPH::computeSparseApproximation()
{
    const double threshold = std::max(0.0, config_.approximation_error_budget);
    std::vector<Pair> filtered;
    filtered.reserve(current_diagram_.count());
    for (const auto &pair : current_diagram_.getPairs())
    {
        if (pair.isInfinite() || pairLifetime(pair) >= threshold)
        {
            filtered.push_back(pair);
        }
    }
    if (filtered.empty() && current_diagram_.count() > 0)
    {
        filtered.push_back(current_diagram_.getPairs().front());
    }
    return diagramFromPairs(filtered);
}

void AcceleratedWindowedPH::startWorkerThreads()
{
    shutdown_flag_.store(false);
    const std::size_t thread_count = std::max<std::size_t>(1, config_.num_threads);
    for (std::size_t i = 0; i < thread_count; ++i)
    {
        worker_threads_.emplace_back(&AcceleratedWindowedPH::workerThreadFunction, this);
    }
}

void AcceleratedWindowedPH::stopWorkerThreads()
{
    shutdown_flag_.store(true);
    work_queue_cv_.notify_all();
    for (auto &thread : worker_threads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void AcceleratedWindowedPH::workerThreadFunction()
{
    while (!shutdown_flag_.load())
    {
        std::vector<algebra::Simplex> batch;
        {
            std::unique_lock<std::mutex> lock(work_queue_mutex_);
            work_queue_cv_.wait(lock,
                                [this] { return shutdown_flag_.load() || !work_queue_.empty(); });
            if (shutdown_flag_.load())
            {
                return;
            }
            batch = std::move(work_queue_.front());
            work_queue_.pop();
        }
        processBatch(batch);
    }
}

void AcceleratedWindowedPH::processBatch(const std::vector<algebra::Simplex> &batch)
{
    if (batch.empty())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(window_mutex_);
    for (const auto &simplex : batch)
    {
        simplex_window_.push(simplex);
        while (simplex_window_.size() > current_window_size_)
        {
            simplex_window_.pop();
            window_engine_synced_ = false;
        }
    }
    processWindowUpdate();
}

void AcceleratedWindowedPH::updatePerformanceMetrics(
    double update_time_ms, PartialRecomputeHeuristic::RecomputeStrategy strategy)
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.last_window_update_time_ms = update_time_ms;
    metrics_.total_updates_processed += 1;
    if (metrics_.total_updates_processed == 1)
    {
        metrics_.average_update_time_ms = update_time_ms;
    }
    else
    {
        const double count = static_cast<double>(metrics_.total_updates_processed);
        metrics_.average_update_time_ms =
            ((metrics_.average_update_time_ms * (count - 1.0)) + update_time_ms) / count;
    }

    switch (strategy)
    {
        case PartialRecomputeHeuristic::RecomputeStrategy::FULL_RECOMPUTE:
            metrics_.full_recomputes_used += 1;
            break;
        case PartialRecomputeHeuristic::RecomputeStrategy::INCREMENTAL_UPDATE:
        case PartialRecomputeHeuristic::RecomputeStrategy::HYBRID:
        case PartialRecomputeHeuristic::RecomputeStrategy::ADAPTIVE:
            metrics_.partial_recomputes_used += 1;
            break;
    }

    metrics_.memory_usage_mb = static_cast<double>(getMemoryUsage()) / (1024.0 * 1024.0);
    if (memory_pool_ != nullptr && memory_pool_->getPoolSize() > 0)
    {
        metrics_.numa_efficiency_score = static_cast<double>(memory_pool_->getAllocatedBytes()) /
                                         static_cast<double>(memory_pool_->getPoolSize());
    }
    else
    {
        metrics_.numa_efficiency_score = 0.0;
    }
}

void AcceleratedWindowedPH::checkHotPathViolations(double update_time_ms)
{
    if (!config_.obey_hot_path_invariant)
    {
        return;
    }
    const bool enforce = hot_path_mode_enabled_.load() || config_.obey_hot_path_invariant;
    if (!enforce)
    {
        return;
    }
    const double budget_ms = static_cast<double>(config_.max_hot_path_latency_ns) / 1'000'000.0;
    if (update_time_ms <= budget_ms)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.hot_path_invariant_violated = true;
}

void AcceleratedWindowedPH::optimizeMemoryLayout()
{
    if (memory_pool_ == nullptr)
    {
        return;
    }
    const std::size_t pool_size = memory_pool_->getPoolSize();
    if (pool_size == 0)
    {
        return;
    }
    const double pressure =
        static_cast<double>(memory_pool_->getAllocatedBytes()) / static_cast<double>(pool_size);
    if (pressure > 0.90)
    {
        memory_pool_->reset();
        preallocate_memory_pools();
    }
}

void AcceleratedWindowedPH::preallocate_memory_pools()
{
    if (memory_pool_ == nullptr)
    {
        return;
    }
    const std::size_t pool_size = memory_pool_->getPoolSize();
    const std::size_t block_size =
        std::max<std::size_t>(1, config_.numa_pool_config.block_size_bytes);
    const std::size_t warm_blocks =
        std::min<std::size_t>(64, std::max<std::size_t>(1, pool_size / block_size / 8));
    std::vector<void *> allocations;
    allocations.reserve(warm_blocks);
    for (std::size_t i = 0; i < warm_blocks; ++i)
    {
        void *ptr = memory_pool_->allocate(block_size);
        if (ptr == nullptr)
        {
            break;
        }
        allocations.push_back(ptr);
    }
    for (void *ptr : allocations)
    {
        memory_pool_->deallocate(ptr);
    }
}

} // namespace nerve::streaming
