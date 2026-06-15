
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/incremental.hpp"

#include <algorithm>
#include <chrono>
#include <ranges>

namespace nerve::streaming
{

namespace
{
Size sequenceMaxDimension(const std::vector<std::vector<Simplex>> &sequence)
{
    Size max_dim = 0;
    for (const auto &batch : sequence)
    {
        for (const auto &simplex : batch)
        {
            max_dim = std::max(max_dim, simplex.dimension());
        }
    }
    return max_dim;
}

Size betaZeroFromDiagram(const Diagram &diagram)
{
    const auto &pairs = diagram.getPairs();
    Size beta_zero = 0;
    for (const auto &pair : pairs)
    {
        if (pair.dimension == 0 && pair.isInfinite())
        {
            ++beta_zero;
        }
    }
    return beta_zero;
}

} // namespace

DistributedPH::DistributedPH()
    : num_workers_(1)
    , max_dimension_(3)
    , communication_pattern_("round_robin")
    , merge_strategy_("exact_recompute")
    , local_ph_()
    , global_filtration_()
    , global_diagram_()
    , results_collected_(false)
{
    initializeWorkers();
}

DistributedPH::DistributedPH(Size num_workers, Size maxDimension)
    : num_workers_(std::max<Size>(1, num_workers))
    , max_dimension_(maxDimension)
    , communication_pattern_("round_robin")
    , merge_strategy_("exact_recompute")
    , local_ph_()
    , global_filtration_()
    , global_diagram_()
    , results_collected_(false)
{
    initializeWorkers();
}

void DistributedPH::distributeComplex(const SimplicialComplex &complex)
{
    global_filtration_.clear();
    for (const auto &entry : complex.getFilteredSimplices())
    {
        const auto inserted = global_filtration_.emplace(entry.first, entry.second);
        if (!inserted.second)
        {
            inserted.first->second = std::min(inserted.first->second, entry.second);
        }
    }
    rebuildWorkersFromGlobalFiltration();
    results_collected_ = false;
}

void DistributedPH::distributeFiltration(const std::vector<std::pair<Simplex, double>> &filtration)
{
    for (const auto &entry : filtration)
    {
        const auto inserted = global_filtration_.emplace(entry.first, entry.second);
        if (!inserted.second)
        {
            inserted.first->second = std::min(inserted.first->second, entry.second);
        }
    }
    rebuildWorkersFromGlobalFiltration();
    results_collected_ = false;
}

void DistributedPH::collectResults()
{
    mergeLocalResults();
    results_collected_ = true;
}

Diagram DistributedPH::getGlobalPersistence() const
{
    return global_diagram_;
}

std::vector<Diagram> DistributedPH::getLocalPersistence() const
{
    std::vector<Diagram> out;
    out.reserve(local_ph_.size());
    for (const auto &worker : local_ph_)
    {
        out.push_back(worker.getPersistenceDiagram());
    }
    return out;
}

void DistributedPH::setNumWorkers(Size num_workers)
{
    num_workers_ = std::max<Size>(1, num_workers);
    rebuildWorkersFromGlobalFiltration();
    results_collected_ = false;
}

void DistributedPH::setCommunicationPattern(const std::string &pattern)
{
    communication_pattern_ = pattern;
}

void DistributedPH::setMergeStrategy(const std::string &strategy)
{
    merge_strategy_ = strategy;
}

void DistributedPH::initializeWorkers()
{
    local_ph_.clear();
    local_ph_.reserve(num_workers_);
    for (Size i = 0; i < num_workers_; ++i)
    {
        local_ph_.emplace_back(max_dimension_);
    }
}

void DistributedPH::rebuildWorkersFromGlobalFiltration()
{
    initializeWorkers();
    if (global_filtration_.empty())
    {
        return;
    }

    std::vector<std::pair<Simplex, double>> sorted(global_filtration_.begin(),
                                                   global_filtration_.end());
    std::ranges::sort(sorted, {}, [](const auto &p) {
        return std::tuple(p.first.dimension(), p.second, p.first);
    });

    for (Size i = 0; i < sorted.size(); ++i)
    {
        const Size worker = i % num_workers_;
        local_ph_[worker].addFiltrationStep({sorted[i]});
    }
}

void DistributedPH::mergeLocalResults()
{
    updateCurrentPairs(computeZigzagPairs());
}

std::vector<Pair> DistributedPH::computeZigzagPairs() const
{
    if (global_filtration_.empty())
    {
        return {};
    }

    SimplicialComplex global_complex;
    std::vector<std::pair<Simplex, double>> sorted(global_filtration_.begin(),
                                                   global_filtration_.end());
    std::ranges::sort(sorted, {}, [](const auto &p) {
        return std::tuple(p.first.dimension(), p.second, p.first);
    });

    for (const auto &entry : sorted)
    {
        global_complex.addSimplexWithFiltration(entry.first, entry.second);
    }

    const auto exact = persistence::computeExactPersistenceZ2(global_complex, max_dimension_);
    std::vector<Pair> merged = exact.pairs;
    std::ranges::sort(merged, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });
    return merged;
}

void DistributedPH::updateCurrentPairs(const std::vector<Pair> &pairs)
{
    global_diagram_.clear();
    for (const auto &pair : pairs)
    {
        global_diagram_.addPair(pair);
    }
}

RealtimePH::RealtimePH()
    : update_frequency_(30.0)
    , max_dimension_(3)
    , change_threshold_(0.05)
    , buffer_size_(256)
    , monitoring_active_(false)
    , incremental_ph_(max_dimension_)
    , stream_buffer_()
    , persistence_history_()
    , timestamp_history_()
    , betti_timeline_()
{}

RealtimePH::RealtimePH(double update_frequency, Size maxDimension)
    : update_frequency_(update_frequency)
    , max_dimension_(maxDimension)
    , change_threshold_(0.05)
    , buffer_size_(256)
    , monitoring_active_(false)
    , incremental_ph_(max_dimension_)
    , stream_buffer_()
    , persistence_history_()
    , timestamp_history_()
    , betti_timeline_()
{}

void RealtimePH::startMonitoring()
{
    monitoring_active_ = true;
}

void RealtimePH::stopMonitoring()
{
    monitoring_active_ = false;
    processBuffer();
}

void RealtimePH::addStreamData(const std::vector<Simplex> &simplices)
{
    stream_buffer_.insert(stream_buffer_.end(), simplices.begin(), simplices.end());
    if (monitoring_active_ && stream_buffer_.size() >= buffer_size_)
    {
        processBuffer();
    }
}

void RealtimePH::processStreamBatch()
{
    processBuffer();
}

Diagram RealtimePH::getLatestPersistence() const
{
    if (!persistence_history_.empty())
    {
        return persistence_history_.back();
    }
    return incremental_ph_.getPersistenceDiagram();
}

std::vector<double> RealtimePH::getPersistenceTimeline() const
{
    return timestamp_history_;
}

std::vector<Size> RealtimePH::getBettiTimeline() const
{
    return betti_timeline_;
}

void RealtimePH::setUpdateFrequency(double frequency)
{
    update_frequency_ = std::max(0.0, frequency);
}

void RealtimePH::setChangeThreshold(double threshold)
{
    change_threshold_ = std::max(0.0, threshold);
}

void RealtimePH::setBufferSize(Size size)
{
    buffer_size_ = std::max<Size>(1, size);
}

void RealtimePH::processBuffer()
{
    if (stream_buffer_.empty())
    {
        return;
    }

    const Diagram before = incremental_ph_.getPersistenceDiagram();
    incremental_ph_.addSimplices(stream_buffer_);
    stream_buffer_.clear();
    const Diagram after = incremental_ph_.getPersistenceDiagram();

    if (detectSignificantChange(before, after) || persistence_history_.empty())
    {
        updateHistory(after);
    }
}

bool RealtimePH::detectSignificantChange(const Diagram &old_diagram,
                                         const Diagram &new_diagram) const
{
    return detail::diagramSupDistance(old_diagram, new_diagram) >= change_threshold_;
}

void RealtimePH::updateHistory(const Diagram &diagram)
{
    persistence_history_.push_back(diagram);
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const double seconds = std::chrono::duration<double>(now).count();
    timestamp_history_.push_back(seconds);
    betti_timeline_.push_back(betaZeroFromDiagram(diagram));
}

std::vector<Pair>
computeIncrementalPersistence(const std::vector<std::vector<Simplex>> &simplex_sequence)
{
    IncrementalPH incremental(sequenceMaxDimension(simplex_sequence));
    for (const auto &simplices : simplex_sequence)
    {
        incremental.addSimplices(simplices);
    }
    return incremental.getPersistencePairs();
}

std::vector<std::vector<Pair>> computeZigzagPersistence(
    const std::vector<std::pair<std::vector<Simplex>, std::vector<Simplex>>> &zigzag_sequence)
{
    Size max_dim = 0;
    for (const auto &[added, removed] : zigzag_sequence)
    {
        for (const auto &simplex : added)
        {
            max_dim = std::max(max_dim, simplex.dimension());
        }
        for (const auto &simplex : removed)
        {
            max_dim = std::max(max_dim, simplex.dimension());
        }
    }

    ZigzagPH zigzag(max_dim);
    std::vector<std::vector<Pair>> history;
    history.reserve(zigzag_sequence.size());
    for (const auto &[added, removed] : zigzag_sequence)
    {
        for (const auto &simplex : added)
        {
            zigzag.addSimplex(simplex);
        }
        for (const auto &simplex : removed)
        {
            zigzag.removeSimplex(simplex);
        }
        const Diagram diagram = zigzag.getCurrentPersistence();
        const auto &pairs = diagram.getPairs();
        history.emplace_back(pairs.begin(), pairs.end());
    }
    return history;
}

Diagram computeWindowedPersistence(const std::vector<std::vector<Simplex>> &data_windows,
                                   Size window_size)
{
    WindowedPH windowed(window_size, sequenceMaxDimension(data_windows));
    for (const auto &window : data_windows)
    {
        for (const auto &simplex : window)
        {
            windowed.addSimplexToWindow(simplex);
        }
    }
    return windowed.getWindowPersistence();
}

std::vector<Diagram> computeDistributedPersistence(const std::vector<SimplicialComplex> &complexes,
                                                   Size num_workers)
{
    Size max_dim = 0;
    for (const auto &complex : complexes)
    {
        max_dim = std::max(max_dim, static_cast<Size>(complex.maxDimension()));
    }

    DistributedPH distributed(num_workers, max_dim);
    std::vector<Diagram> results;
    results.reserve(complexes.size());
    for (const auto &complex : complexes)
    {
        distributed.distributeComplex(complex);
        distributed.collectResults();
        results.push_back(distributed.getGlobalPersistence());
    }
    return results;
}

} // namespace nerve::streaming
