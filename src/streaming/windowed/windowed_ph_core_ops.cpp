
#include "nerve/streaming/windowed_ph.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <set>

namespace nerve::streaming
{

namespace
{
constexpr std::size_t kDefaultMaxDimension = 3;

std::vector<algebra::Simplex> queueToVector(std::queue<algebra::Simplex> window)
{
    std::vector<algebra::Simplex> simplices;
    simplices.reserve(window.size());
    while (!window.empty())
    {
        simplices.push_back(window.front());
        window.pop();
    }
    return simplices;
}
} // namespace

AcceleratedWindowedPH::AcceleratedWindowedPH(const OptimizationConfig &config)
    : config_(config)
    , memory_pool_(nullptr)
    , heuristic_(nullptr)
    , windowed_ph_(nullptr)
    , simplex_window_()
    , current_window_size_(1000)
    , current_diagram_()
    , metrics_()
    , window_mutex_()
    , metrics_mutex_()
    , worker_threads_()
    , work_queue_()
    , work_queue_mutex_()
    , work_queue_cv_()
    , shutdown_flag_(false)
    , hot_path_mode_enabled_(false)
    , previous_window_snapshot_()
    , window_engine_synced_(false)
{
    initializeOptimizations();
}

AcceleratedWindowedPH::AcceleratedWindowedPH()
    : AcceleratedWindowedPH(OptimizationConfig{})
{}
AcceleratedWindowedPH::~AcceleratedWindowedPH()
{
    stopWorkerThreads();
}

void AcceleratedWindowedPH::initializeOptimizations()
{
    stopWorkerThreads();
    if (config_.enable_numa_optimization)
    {
        memory_pool_ = std::make_unique<NUMAMemoryPool>(config_.numa_pool_config);
    }
    else
    {
        memory_pool_.reset();
    }
    heuristic_ = std::make_unique<PartialRecomputeHeuristic>(config_.heuristic_config);
    current_window_size_ = std::max<std::size_t>(1, current_window_size_);
    windowed_ph_ = std::make_unique<WindowedPH>(current_window_size_, kDefaultMaxDimension);
    previous_window_snapshot_.clear();
    window_engine_synced_ = false;
    if (config_.enable_parallel_processing && config_.num_threads > 1)
    {
        startWorkerThreads();
    }
    preallocate_memory_pools();
}

void AcceleratedWindowedPH::addSimplex(const algebra::Simplex &simplex)
{
    if (config_.enable_batch_processing)
    {
        addSimplicesBatch({simplex});
        return;
    }
    processBatch({simplex});
}

void AcceleratedWindowedPH::addSimplicesBatch(const std::vector<algebra::Simplex> &simplices)
{
    if (simplices.empty())
    {
        return;
    }
    if (config_.enable_parallel_processing && !worker_threads_.empty() &&
        simplices.size() >= config_.batch_size)
    {
        {
            std::lock_guard<std::mutex> lock(work_queue_mutex_);
            work_queue_.push(simplices);
        }
        work_queue_cv_.notify_one();
        return;
    }
    processBatch(simplices);
}

void AcceleratedWindowedPH::slideWindow(std::size_t new_window_size)
{
    const std::size_t sanitized_window_size = std::max<std::size_t>(1, new_window_size);
    if (sanitized_window_size == current_window_size_)
    {
        return;
    }
    const std::size_t old_size = current_window_size_;
    current_window_size_ = sanitized_window_size;
    handleWindowSlide(old_size, sanitized_window_size);
}

void AcceleratedWindowedPH::advanceWindow(std::size_t step_size)
{
    if (step_size == 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(window_mutex_);
    for (std::size_t i = 0; i < step_size && !simplex_window_.empty(); ++i)
    {
        simplex_window_.pop();
    }
    processWindowUpdate();
}

Diagram AcceleratedWindowedPH::compute()
{
    std::lock_guard<std::mutex> lock(window_mutex_);
    return current_diagram_;
}

Diagram AcceleratedWindowedPH::computePersistenceApproximate()
{
    if (!config_.enable_approximate_mode)
    {
        return compute();
    }
    std::lock_guard<std::mutex> lock(window_mutex_);
    return computeSparseApproximation();
}

Diagram AcceleratedWindowedPH::computePersistenceTopLifetimeTruncation()
{
    if (!config_.enable_witness_mode)
    {
        return compute();
    }
    std::lock_guard<std::mutex> lock(window_mutex_);
    return computeTopLifetimeTruncation();
}

void AcceleratedWindowedPH::setWindowSize(std::size_t window_size)
{
    slideWindow(window_size);
}

void AcceleratedWindowedPH::setOptimizationConfig(const OptimizationConfig &config)
{
    config_ = config;
    initializeOptimizations();
}

AcceleratedWindowedPH::OptimizationConfig AcceleratedWindowedPH::getOptimizationConfig() const
{
    return config_;
}

AcceleratedWindowedPH::PerformanceMetrics AcceleratedWindowedPH::getPerformanceMetrics() const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void AcceleratedWindowedPH::resetPerformanceMetrics()
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = PerformanceMetrics{};
}

void AcceleratedWindowedPH::bindToNumaNode(int node)
{
    if (memory_pool_ != nullptr)
    {
        memory_pool_->bindToNumaNode(node);
    }
}

std::size_t AcceleratedWindowedPH::getMemoryUsage() const
{
    std::size_t usage = simplex_window_.size() * sizeof(algebra::Simplex);
    if (memory_pool_ != nullptr)
    {
        usage += memory_pool_->getAllocatedBytes();
    }
    return usage;
}

void AcceleratedWindowedPH::compactMemory()
{
    optimizeMemoryLayout();
}

bool AcceleratedWindowedPH::checkHotPathInvariants() const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return !metrics_.hot_path_invariant_violated;
}

void AcceleratedWindowedPH::enableHotPathMode(bool enable)
{
    hot_path_mode_enabled_.store(enable);
}

void AcceleratedWindowedPH::processWindowUpdate()
{
    const auto start = std::chrono::high_resolution_clock::now();
    const std::vector<algebra::Simplex> current_snapshot = queueToVector(simplex_window_);
    std::set<algebra::Simplex> previousSet(previous_window_snapshot_.begin(),
                                           previous_window_snapshot_.end());
    std::set<algebra::Simplex> currentSet(current_snapshot.begin(), current_snapshot.end());

    std::vector<algebra::Simplex> added_simplices;
    std::vector<algebra::Simplex> removed_simplices;
    added_simplices.reserve(currentSet.size());
    removed_simplices.reserve(previousSet.size());

    for (const auto &simplex : currentSet)
    {
        if (previousSet.find(simplex) == previousSet.end())
        {
            added_simplices.push_back(simplex);
        }
    }
    for (const auto &simplex : previousSet)
    {
        if (currentSet.find(simplex) == currentSet.end())
        {
            removed_simplices.push_back(simplex);
        }
    }

    PartialRecomputeHeuristic::RecomputeStrategy strategy =
        PartialRecomputeHeuristic::RecomputeStrategy::FULL_RECOMPUTE;
    if (heuristic_ != nullptr)
    {
        strategy =
            heuristic_->determineStrategy(added_simplices, removed_simplices, current_diagram_);
    }

    switch (strategy)
    {
        case PartialRecomputeHeuristic::RecomputeStrategy::INCREMENTAL_UPDATE:
            current_diagram_ = recomputeIncremental(added_simplices, removed_simplices);
            break;
        case PartialRecomputeHeuristic::RecomputeStrategy::HYBRID:
            current_diagram_ = recomputePartial(added_simplices, removed_simplices);
            break;
        case PartialRecomputeHeuristic::RecomputeStrategy::ADAPTIVE:
        case PartialRecomputeHeuristic::RecomputeStrategy::FULL_RECOMPUTE:
        default:
            current_diagram_ = recomputeFull();
            break;
    }

    previous_window_snapshot_ = current_snapshot;
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    updatePerformanceMetrics(elapsed_ms, strategy);
    checkHotPathViolations(elapsed_ms);
}

void AcceleratedWindowedPH::handleWindowSlide(std::size_t /*old_size*/, std::size_t new_size)
{
    std::lock_guard<std::mutex> lock(window_mutex_);
    while (simplex_window_.size() > new_size)
    {
        simplex_window_.pop();
    }
    if (windowed_ph_ != nullptr)
    {
        windowed_ph_->setWindowSize(new_size);
    }
    processWindowUpdate();
}

Diagram AcceleratedWindowedPH::recomputeIncremental(const std::vector<algebra::Simplex> &added,
                                                    const std::vector<algebra::Simplex> &removed)
{
    if (!removed.empty() || windowed_ph_ == nullptr || !window_engine_synced_)
    {
        return recomputeFull();
    }
    for (const auto &simplex : added)
    {
        windowed_ph_->addSimplexToWindow(simplex);
    }
    window_engine_synced_ = true;
    return windowed_ph_->getWindowPersistence();
}

Diagram AcceleratedWindowedPH::recomputePartial(const std::vector<algebra::Simplex> &added,
                                                const std::vector<algebra::Simplex> &removed)
{
    if (removed.empty() && added.size() <= config_.heuristic_config.max_partial_updates)
    {
        return recomputeIncremental(added, removed);
    }
    return recomputeFull();
}

Diagram AcceleratedWindowedPH::recomputeFull()
{
    std::vector<algebra::Simplex> snapshot = queueToVector(simplex_window_);
    auto rebuilt = std::make_unique<WindowedPH>(current_window_size_, kDefaultMaxDimension);
    for (const auto &simplex : snapshot)
    {
        rebuilt->addSimplexToWindow(simplex);
    }
    Diagram diagram = rebuilt->getWindowPersistence();
    windowed_ph_ = std::move(rebuilt);
    window_engine_synced_ = true;
    return diagram;
}

} // namespace nerve::streaming
