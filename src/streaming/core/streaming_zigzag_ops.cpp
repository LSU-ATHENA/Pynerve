
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/zigzag_filters.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <stdexcept>

namespace nerve::streaming
{

// Memory Unit Constants
constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

// Default Constants
constexpr size_t DEFAULT_MEMORY_LIMIT_GB = 1;
constexpr int DEFAULT_CHECKPOINT_INTERVAL = 32;
constexpr int DEFAULT_MAX_CHECKPOINTS = 8;

namespace
{

bool filtrationStepLess(const std::pair<Simplex, double> &lhs,
                        const std::pair<Simplex, double> &rhs)
{
    if (lhs.first.dimension() != rhs.first.dimension())
    {
        return lhs.first.dimension() < rhs.first.dimension();
    }
    if (lhs.second != rhs.second)
    {
        return lhs.second < rhs.second;
    }
    return lhs.first < rhs.first;
}

} // namespace

ZigzagPH::ZigzagPH()
    : max_dimension_(3)
    , track_intermediate_(true)
    , current_step_(0)
    , deterministic_ordering_(false)
    , deterministic_seed_(0)
    , memory_limit_bytes_(DEFAULT_MEMORY_LIMIT_GB * BYTES_PER_GB)
    , witness_mode_(false)
    , sparse_mode_(false)
    , current_complex_()
    , zigzag_history_()
    , current_pairs_()
    , addition_stack_()
    , removal_stack_()
    , checkpoints_()
    , checkpoint_interval_(DEFAULT_CHECKPOINT_INTERVAL)
    , max_checkpoints_(DEFAULT_MAX_CHECKPOINTS)
    , incremental_engine_(max_dimension_)
    , incremental_engine_ready_(false)
    , current_memory_usage_(0)
{}

ZigzagPH::ZigzagPH(Size max_dimension)
    : max_dimension_(max_dimension)
    , track_intermediate_(true)
    , current_step_(0)
    , deterministic_ordering_(false)
    , deterministic_seed_(0)
    , memory_limit_bytes_(DEFAULT_MEMORY_LIMIT_GB * BYTES_PER_GB)
    , witness_mode_(false)
    , sparse_mode_(false)
    , current_complex_()
    , zigzag_history_()
    , current_pairs_()
    , addition_stack_()
    , removal_stack_()
    , checkpoints_()
    , checkpoint_interval_(DEFAULT_CHECKPOINT_INTERVAL)
    , max_checkpoints_(DEFAULT_MAX_CHECKPOINTS)
    , incremental_engine_(max_dimension_)
    , incremental_engine_ready_(false)
    , current_memory_usage_(0)
{}

void ZigzagPH::addSimplex(const Simplex &simplex)
{
    processAddition(simplex);
    ++current_step_;
    if (track_intermediate_)
    {
        updateZigzagHistory();
    }
    saveCheckpointIfNeeded();
    updateMemoryUsage();
}

void ZigzagPH::removeSimplex(const Simplex &simplex)
{
    processRemoval(simplex);
    ++current_step_;
    if (track_intermediate_)
    {
        updateZigzagHistory();
    }
    saveCheckpointIfNeeded();
    updateMemoryUsage();
}

void ZigzagPH::addFiltrationStep(const std::vector<std::pair<Simplex, double>> &step)
{
    std::vector<std::pair<Simplex, double>> ordered_step(step.begin(), step.end());
    if (deterministic_ordering_)
    {
        std::ranges::sort(ordered_step, filtrationStepLess);
    }
    for (const auto &[simplex, value] : ordered_step)
    {
        processAdditionWithFiltration(simplex, value);
    }

    ++current_step_;
    if (track_intermediate_)
    {
        updateZigzagHistory();
    }
    saveCheckpointIfNeeded();
    updateMemoryUsage();
}

void ZigzagPH::removeFiltrationStep(const std::vector<std::pair<Simplex, double>> &step)
{
    std::vector<std::pair<Simplex, double>> ordered_step(step.begin(), step.end());
    if (deterministic_ordering_)
    {
        std::ranges::sort(ordered_step, filtrationStepLess);
    }
    for (const auto &[simplex, value] : ordered_step)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("zigzag filtration removal values must be finite");
        }
        processRemoval(simplex);
    }

    ++current_step_;
    if (track_intermediate_)
    {
        updateZigzagHistory();
    }
    saveCheckpointIfNeeded();
    updateMemoryUsage();
}

std::vector<std::vector<Pair>> ZigzagPH::getZigzagPersistence() const
{
    return zigzag_history_;
}

Diagram ZigzagPH::getCurrentPersistence() const
{
    Diagram diagram;
    for (const auto &pair : computeZigzagPairs())
    {
        diagram.addPair(pair);
    }
    return diagram;
}

Size ZigzagPH::currentStep() const
{
    return current_step_;
}

void ZigzagPH::reset()
{
    current_complex_.clear();
    zigzag_history_.clear();
    current_pairs_.clear();
    addition_stack_.clear();
    removal_stack_.clear();
    checkpoints_.clear();
    current_step_ = 0;
    incremental_engine_.clear();
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_ready_ = false;
}

void ZigzagPH::setMaxDimension(Size max_dim)
{
    max_dimension_ = max_dim;
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_ready_ = false;
    recomputeCurrentPairs();
}

void ZigzagPH::setTrackIntermediate(bool track)
{
    track_intermediate_ = track;
}

void ZigzagPH::setDeterministicOrdering(bool deterministic)
{
    deterministic_ordering_ = deterministic;
}

void ZigzagPH::setMemoryLimit(size_t memory_limit_bytes)
{
    memory_limit_bytes_ = memory_limit_bytes;
}

void ZigzagPH::setWitnessMode(bool witness_mode)
{
    witness_mode_ = witness_mode;
}

void ZigzagPH::setSparseMode(bool sparse_mode)
{
    sparse_mode_ = sparse_mode;
}

size_t ZigzagPH::getMemoryUsage() const
{
    return current_memory_usage_;
}

bool ZigzagPH::isMemoryLimitExceeded() const
{
    return current_memory_usage_ > memory_limit_bytes_;
}

void ZigzagPH::setDeterministicSeed(uint32_t seed)
{
    deterministic_seed_ = seed;
}

uint32_t ZigzagPH::getDeterministicSeed() const
{
    return deterministic_seed_;
}

void ZigzagPH::processAddition(const Simplex &simplex)
{
    processAdditionWithFiltration(simplex, static_cast<double>(current_step_));
}

void ZigzagPH::processAdditionWithFiltration(const Simplex &simplex, double filtration)
{
    if (!std::isfinite(filtration))
    {
        throw std::invalid_argument("zigzag filtration addition value must be finite");
    }
    addition_stack_.push_back(simplex);
    current_complex_.addSimplexWithFiltration(simplex, filtration);

    recomputeCurrentPairs();
}

void ZigzagPH::processRemoval(const Simplex &simplex)
{
    removal_stack_.push_back(simplex);
    current_complex_.removeSimplex(simplex);
    recomputeCurrentPairs();
}

void ZigzagPH::updateZigzagHistory()
{
    zigzag_history_.push_back(current_pairs_);
}

std::vector<Pair> ZigzagPH::computeZigzagPairs() const
{
    if (witness_mode_)
    {
        return computeWitnessPairs();
    }
    if (sparse_mode_)
    {
        return computeSparsePairs();
    }
    return current_pairs_;
}

void ZigzagPH::updateMemoryUsage()
{
    size_t checkpoint_bytes = 0;
    for (const auto &checkpoint : checkpoints_)
    {
        checkpoint_bytes += checkpoint.pairs.size() * sizeof(Pair);
        checkpoint_bytes += checkpoint.complex.size() * sizeof(Simplex);
    }
    current_memory_usage_ = current_pairs_.size() * sizeof(Pair) +
                            addition_stack_.size() * sizeof(Simplex) +
                            removal_stack_.size() * sizeof(Simplex) + checkpoint_bytes;
}

void ZigzagPH::saveCheckpointIfNeeded()
{
    if (checkpoint_interval_ == 0)
    {
        return;
    }
    if (current_step_ == 0 || (current_step_ % checkpoint_interval_) != 0)
    {
        return;
    }

    Checkpoint checkpoint;
    checkpoint.step = current_step_;
    checkpoint.complex = current_complex_;
    checkpoint.pairs = current_pairs_;
    checkpoints_.push_back(std::move(checkpoint));
    while (checkpoints_.size() > max_checkpoints_)
    {
        checkpoints_.pop_front();
    }
}

void ZigzagPH::rebuildIncrementalEngine()
{
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_.rebuildFromComplex(current_complex_);
    incremental_engine_ready_ = true;
}

void ZigzagPH::recomputeCurrentPairs()
{
    auto exact = persistence::computeExactPersistenceZ2(current_complex_, max_dimension_);
    current_pairs_ = std::move(exact.pairs);
    rebuildIncrementalEngine();
}

void ZigzagPH::applyDeterministicOrdering(std::vector<Simplex> &simplices)
{
    std::ranges::sort(simplices, {}, &Simplex::dimension);
}

uint32_t ZigzagPH::hashSimplex(const Simplex &simplex) const
{
    uint32_t hash = deterministic_seed_ ^ 0x9E3779B9u;
    for (const auto vertex : simplex.vertices())
    {
        hash ^= static_cast<uint32_t>(vertex) + 0x9E3779B9u + (hash << 6u) + (hash >> 2u);
    }
    return hash;
}

std::vector<Pair> ZigzagPH::computeWitnessPairs() const
{
    if (current_pairs_.empty())
    {
        return {};
    }
    return detail::computeWitnessModePairs(current_complex_, max_dimension_, deterministic_seed_);
}

std::vector<Pair> ZigzagPH::computeSparsePairs() const
{
    if (current_pairs_.empty())
    {
        return {};
    }
    return detail::computeSparseModePairs(current_complex_, max_dimension_, deterministic_seed_);
}

void ZigzagPH::updateCurrentPairs(const std::vector<Pair> &new_pairs)
{
    current_pairs_ = new_pairs;
}

} // namespace nerve::streaming
