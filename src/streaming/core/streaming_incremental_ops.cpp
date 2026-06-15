
#include "nerve/streaming/incremental.hpp"

#include <algorithm>
#include <chrono>

namespace nerve::streaming
{

IncrementalPH::IncrementalPH()
    : max_dimension_(3)
    , coefficient_field_(2)
    , algorithm_("standard")
    , current_complex_()
    , current_pairs_()
    , current_betti_()
    , simplex_to_index_()
    , index_to_simplex_()
    , is_active_()
    , filtration_values_()
    , total_computation_time_(0.0)
    , num_updates_(0)
    , next_filtration_value_(0.0)
    , incremental_engine_(max_dimension_)
    , incremental_engine_ready_(false)
{
    initializeIncrementalStructures();
}

IncrementalPH::IncrementalPH(Size maxDimension)
    : max_dimension_(maxDimension)
    , coefficient_field_(2)
    , algorithm_("standard")
    , current_complex_()
    , current_pairs_()
    , current_betti_()
    , simplex_to_index_()
    , index_to_simplex_()
    , is_active_()
    , filtration_values_()
    , total_computation_time_(0.0)
    , num_updates_(0)
    , next_filtration_value_(0.0)
    , incremental_engine_(max_dimension_)
    , incremental_engine_ready_(false)
{
    initializeIncrementalStructures();
}

void IncrementalPH::addSimplex(const Simplex &simplex)
{
    const auto start = std::chrono::high_resolution_clock::now();
    if (isSimplexPresent(simplex))
    {
        return;
    }

    const double filtration = next_filtration_value_++;
    current_complex_.addSimplexWithFiltration(simplex, filtration);

    const Size index = index_to_simplex_.size();
    simplex_to_index_[simplex] = index;
    index_to_simplex_.push_back(simplex);
    is_active_.push_back(true);
    filtration_values_.push_back(filtration);

    updatePersistenceAfterAddition(simplex);
    const auto end = std::chrono::high_resolution_clock::now();
    total_computation_time_ += std::chrono::duration<double>(end - start).count();
    ++num_updates_;
}

void IncrementalPH::removeSimplex(const Simplex &simplex)
{
    const auto start = std::chrono::high_resolution_clock::now();
    if (!isSimplexPresent(simplex))
    {
        return;
    }

    current_complex_.removeSimplex(simplex);
    const Size index = getSimplexIndex(simplex);
    if (index < is_active_.size())
    {
        is_active_[index] = false;
    }

    updatePersistenceAfterRemoval(simplex);
    const auto end = std::chrono::high_resolution_clock::now();
    total_computation_time_ += std::chrono::duration<double>(end - start).count();
    ++num_updates_;
}

void IncrementalPH::addSimplices(const std::vector<Simplex> &simplices)
{
    for (const auto &simplex : simplices)
    {
        addSimplex(simplex);
    }
}

void IncrementalPH::removeSimplices(const std::vector<Simplex> &simplices)
{
    for (const auto &simplex : simplices)
    {
        removeSimplex(simplex);
    }
}

void IncrementalPH::addComplex(const SimplicialComplex &complex)
{
    for (Dimension dim = 0; dim <= complex.maxDimension(); ++dim)
    {
        addSimplices(complex.simplicesOfDimension(dim));
    }
}

void IncrementalPH::addFiltrationStep(
    const std::vector<std::pair<Simplex, double>> &filtration_step)
{
    for (const auto &[simplex, value] : filtration_step)
    {
        if (isSimplexPresent(simplex))
        {
            current_complex_.setFiltration(simplex, value);
            const Size index = getSimplexIndex(simplex);
            if (index < filtration_values_.size())
            {
                filtration_values_[index] = value;
            }
            incremental_engine_ready_ = false;
            continue;
        }

        current_complex_.addSimplexWithFiltration(simplex, value);
        const Size index = index_to_simplex_.size();
        simplex_to_index_[simplex] = index;
        index_to_simplex_.push_back(simplex);
        is_active_.push_back(true);
        filtration_values_.push_back(value);
        next_filtration_value_ = std::max(next_filtration_value_, value + 1.0);
        updatePersistenceAfterAddition(simplex);
        ++num_updates_;
    }

    if (!incremental_engine_ready_)
    {
        recomputePersistence();
    }
}

Diagram IncrementalPH::getPersistenceDiagram() const
{
    Diagram diagram;
    for (const auto &pair : current_pairs_)
    {
        diagram.addPair(pair);
    }
    return diagram;
}

std::vector<Pair> IncrementalPH::getPersistencePairs() const
{
    return current_pairs_;
}

std::vector<Size> IncrementalPH::getBettiNumbers() const
{
    return current_betti_;
}

Size IncrementalPH::numSimplices() const
{
    return current_complex_.size();
}

Size IncrementalPH::getMaxDimension() const
{
    return max_dimension_;
}

void IncrementalPH::reset()
{
    current_complex_.clear();
    current_pairs_.clear();
    current_betti_.clear();
    simplex_to_index_.clear();
    index_to_simplex_.clear();
    is_active_.clear();
    filtration_values_.clear();
    total_computation_time_ = 0.0;
    num_updates_ = 0;
    next_filtration_value_ = 0.0;
    incremental_engine_.clear();
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_ready_ = false;
    initializeIncrementalStructures();
}

void IncrementalPH::clear()
{
    reset();
}

void IncrementalPH::setMaxDimension(Size max_dim)
{
    max_dimension_ = max_dim;
    current_betti_.assign(max_dimension_ + 1, 0);
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_ready_ = false;
    recomputePersistence();
}

void IncrementalPH::setCoefficientField(int p)
{
    coefficient_field_ = p;
}

void IncrementalPH::setAlgorithm(const std::string &algorithm)
{
    algorithm_ = algorithm;
}

double IncrementalPH::getComputationTime() const
{
    return total_computation_time_;
}

Size IncrementalPH::getNumUpdates() const
{
    return num_updates_;
}

void IncrementalPH::initializeIncrementalStructures()
{
    current_betti_.assign(max_dimension_ + 1, 0);
    incremental_engine_.setMaxDim(max_dimension_);
    incremental_engine_ready_ = false;
    recomputePersistence();
}

void IncrementalPH::recomputePersistence()
{
    current_pairs_.clear();
    if (current_complex_.size() > 0)
    {
        incremental_engine_.setMaxDim(max_dimension_);
        incremental_engine_.rebuildFromComplex(current_complex_);
        auto exact = incremental_engine_.snapshot();
        current_pairs_ = std::move(exact.pairs);
        incremental_engine_ready_ = true;
    }
    else
    {
        incremental_engine_.clear();
        incremental_engine_ready_ = true;
    }
    updateBettiNumbers();
}

Size IncrementalPH::getSimplexIndex(const Simplex &simplex) const
{
    const auto it = simplex_to_index_.find(simplex);
    return it != simplex_to_index_.end() ? it->second : static_cast<Size>(-1);
}

bool IncrementalPH::isSimplexPresent(const Simplex &simplex) const
{
    const auto it = simplex_to_index_.find(simplex);
    if (it == simplex_to_index_.end() || it->second >= is_active_.size())
    {
        return false;
    }
    return is_active_[it->second];
}

void IncrementalPH::updateBettiNumbers()
{
    std::fill(current_betti_.begin(), current_betti_.end(), 0);
    for (const auto &pair : current_pairs_)
    {
        if (pair.isInfinite() && pair.dimension >= 0 &&
            pair.dimension < static_cast<Dimension>(current_betti_.size()))
        {
            current_betti_[pair.dimension]++;
        }
    }
}

void IncrementalPH::updatePersistenceAfterAddition(const Simplex & /*simplex*/)
{
    if (!incremental_engine_ready_)
    {
        recomputePersistence();
        return;
    }

    const Simplex &simplex = index_to_simplex_.back();
    const double filtration = current_complex_.getFiltration(simplex);
    if (!incremental_engine_.addSimplex(simplex, filtration))
    {
        recomputePersistence();
        return;
    }

    auto exact = incremental_engine_.snapshot();
    current_pairs_ = std::move(exact.pairs);
    updateBettiNumbers();
}

void IncrementalPH::updatePersistenceAfterRemoval(const Simplex & /*simplex*/)
{
    recomputePersistence();
}

} // namespace nerve::streaming
