#include "nerve/core/rng/rng.hpp"

#include <utility>

namespace nerve::core
{

RNG DeterministicRNGFactory::createFromContract(const DeterminismContract &contract)
{
    return RNG(contract);
}

std::vector<RNG> DeterministicRNGFactory::createSplitGenerators(const DeterminismContract &contract,
                                                                Size count)
{
    std::vector<RNG> generators;
    generators.reserve(count);
    RNG baseRng(contract);
    for (Size i = 0; i < count; ++i)
    {
        RNG split_rng = baseRng.split();
        generators.push_back(std::move(split_rng));
    }
    return generators;
}

bool DeterministicRNGFactory::validateDeterminism(const RNG &rng,
                                                  const DeterminismContract &contract)
{
    if (contract.enable_deterministic_random && !rng.isDeterministic())
    {
        return false;
    }
    auto metadata = rng.getDeterminismMetadata();
    if (contract.params_hash_valid && metadata.params_hash != contract.params_hash)
    {
        return false;
    }
    if (contract.level > metadata.achieved_level)
    {
        return false;
    }
    return true;
}

std::vector<std::string>
DeterministicRNGFactory::getDeterminismWarnings(const RNG &rng, const DeterminismContract &contract)
{
    std::vector<std::string> warnings;
    auto metadata = rng.getDeterminismMetadata();
    if (!rng.isDeterministic() && contract.enable_deterministic_random)
    {
        warnings.push_back("RNG is not deterministic despite contract requirement");
    }
    if (contract.level > metadata.achieved_level)
    {
        warnings.push_back("Achieved determinism level lower than requested");
    }
    if (!metadata.warnings.empty())
    {
        warnings.insert(warnings.end(), metadata.warnings.begin(), metadata.warnings.end());
    }
    return warnings;
}

RNG &getGlobalRng()
{
    static RNG globalRng;
    return globalRng;
}

} // namespace nerve::core
