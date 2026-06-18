
#pragma once

#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

namespace nerve::core
{
struct DeterminismContract;
}

namespace nerve::persistence
{

// Pair is defined in nerve/types.hpp (namespace nerve)
// Diagram class is the canonical persistence result container.
using nerve::Pair;

class Diagram
{
public:
    Diagram() = default;
    explicit Diagram(std::vector<Pair> pairs)
        : pairs_(std::move(pairs))
    {}

    void addPair(const Pair &pair) { pairs_.push_back(pair); }
    void add_pair(const Pair &pair) { addPair(pair); }

    errors::ErrorResult<void> addPair(const Pair &pair, const core::DeterminismContract &contract);
    errors::ErrorResult<void> add_pair(const Pair &pair, const core::DeterminismContract &contract)
    {
        return addPair(pair, contract);
    }
    errors::ErrorResult<std::vector<Pair>>
    pairsOfDimension(Dimension dim, const core::DeterminismContract &contract) const;
    errors::ErrorResult<std::vector<int>>
    computeBettiNumbers(const core::DeterminismContract &contract) const;

    const std::vector<Pair> &getPairs() const { return pairs_; }
    const std::vector<Pair> &pairs() const { return getPairs(); }

    std::vector<Pair> &getPairs() { return pairs_; }
    std::vector<Pair> &pairs() { return getPairs(); }

    Size count() const { return pairs_.size(); }

    bool isEmpty() const { return pairs_.empty(); }

    void clear() { pairs_.clear(); }

    std::vector<Pair> getPairsByDimension(Dimension dim) const;
    std::vector<Size> computeBetti() const;
    double getMaxPersistence() const;
    double getAveragePersistence() const;

private:
    std::vector<Pair> pairs_;
};

using Point = std::pair<Field, Field>;

} // namespace nerve::persistence
