
#include "nerve/common.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <set>
#include <utility>
#include <vector>
namespace nerve::persistence
{
namespace
{

bool validBettiPair(const Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    if (!std::isfinite(pair.birth) || (!finite_death && !pair.isInfinite()) || pair.dimension < 0)
    {
        return false;
    }
    return !finite_death || pair.death < 0.0 || pair.death >= pair.birth;
}

} // namespace

errors::ErrorResult<void> Diagram::addPair(const Pair &pair,
                                           const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }
    pairs_.push_back(pair);
    return errors::ErrorResult<void>::success();
}
errors::ErrorResult<Vector<Pair>>
Diagram::pairsOfDimension(Dimension dim, const core::DeterminismContract &contract) const
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<Vector<Pair>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }
    Vector<Pair> result;
    for (const auto &pair : pairs_)
    {
        if (pair.dimension == dim)
        {
            result.push_back(pair);
        }
    }
    if (contract.level >= core::DeterminismLevel::STRICT)
    {
        std::ranges::sort(result, {}, &Pair::birth);
    }
    return errors::ErrorResult<Vector<Pair>>::success(Vector<Pair>(result));
}
errors::ErrorResult<Vector<int>>
Diagram::computeBettiNumbers(const core::DeterminismContract &contract) const
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<Vector<int>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }
    Vector<int> betti;
    ::std::set<int> dimension_set;
    for (const auto &pair : pairs_)
    {
        if (!validBettiPair(pair))
        {
            return errors::ErrorResult<Vector<int>>::error(
                errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
        dimension_set.insert(pair.dimension);
    }
    for (int dim : dimension_set)
    {
        int essential = 0;
        for (const auto &pair : pairs_)
        {
            if (pair.dimension == dim)
            {
                if (pair.death < 0 || std::isinf(pair.death))
                {
                    essential++;
                }
            }
        }
        betti.push_back(essential);
    }
    return errors::ErrorResult<Vector<int>>::success(Vector<int>(betti));
}
} // namespace nerve::persistence
