
#include "nerve/core/rng/rng.hpp"
#include "nerve/core/sha256.hpp"
#include "rng_wire_format.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>

using nerve::core::SHA256_CTX;
using nerve::core::sha256Final;
using nerve::core::sha256Init;
using nerve::core::sha256Update;

namespace nerve::core
{

namespace
{

[[nodiscard]] constexpr Size maxIndexableCount() noexcept
{
    return static_cast<Size>(std::numeric_limits<Index>::max()) + Size{1};
}

void ensureIndexableCount(Size count, const char *context)
{
    if (count > maxIndexableCount())
    {
        throw std::length_error(context);
    }
}

[[nodiscard]] Index toIndex(Size value, const char *context)
{
    if (value > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<Index>(value);
}

[[nodiscard]] double checkedDistance(double value, const char *context)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(context);
    }
    return value;
}

} // namespace

RNG::RNG()
    : seed_(makeUnseededSeed())
    , generator_(seed_)
{
    determinism_contract_.setRngSeed(seed_);
    initializeDistributions();
    is_deterministic_ = false;
    updateDeterminismMetadata();
}

RNG::RNG(uint64_t seed)
    : seed_(seed)
    , generator_(seed)
{
    determinism_contract_.setRngSeed(seed_);
    initializeDistributions();
    is_deterministic_ = true;
    updateDeterminismMetadata();
}
RNG::RNG(const DeterminismContract &contract)
    : determinism_contract_(contract)
{
    initializeFromContract(contract);
    updateDeterminismMetadata();
}
[[nodiscard]] uint64_t RNG::seed() const noexcept
{
    return seed_;
}
void RNG::seed(uint64_t new_seed)
{
    seed_ = new_seed;
    generator_.seed(new_seed);
    determinism_contract_.setRngSeed(new_seed);
    initializeDistributions();
    is_deterministic_ = true;
    determinism_metadata_.warnings.clear();
    updateDeterminismMetadata();
}
void RNG::seed(const DeterminismContract &contract)
{
    determinism_contract_ = contract;
    initializeFromContract(contract);
    updateDeterminismMetadata();
}
DeterminismMetadata RNG::getDeterminismMetadata() const
{
    return determinism_metadata_;
}
[[nodiscard]] bool RNG::isDeterministic() const noexcept
{
    return is_deterministic_;
}
[[nodiscard]] double RNG::uniform()
{
    return uniform_dist_(generator_);
}
[[nodiscard]] double RNG::uniform(double min, double max)
{
    if (!(min < max) || !std::isfinite(min) || !std::isfinite(max))
    {
        throw std::invalid_argument("Uniform range must be finite and ordered");
    }
    std::uniform_real_distribution<double> dist(min, max);
    return dist(generator_);
}
[[nodiscard]] Index RNG::uniformInt(Index min, Index max)
{
    if (min > max)
    {
        throw std::invalid_argument("Integer range minimum cannot exceed maximum");
    }
    std::uniform_int_distribution<Index> dist(min, max);
    return dist(generator_);
}
double RNG::normal(double mean, double stddev)
{
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0)
    {
        throw std::invalid_argument("Normal distribution parameters must be finite");
    }
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator_);
}
double RNG::exponential(double lambda)
{
    if (!std::isfinite(lambda) || lambda <= 0.0)
    {
        throw std::invalid_argument("Exponential rate must be finite and positive");
    }
    std::exponential_distribution<double> dist(lambda);
    return dist(generator_);
}
std::vector<Index> RNG::subsample(Size n, Size k)
{
    if (k > n)
    {
        throw std::invalid_argument("Sample size cannot exceed population size");
    }
    if (k == 0)
    {
        return {};
    }
    ensureIndexableCount(n, "Population size exceeds RNG index range");
    if (k == n)
    {
        std::vector<Index> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    if (k < n / 2)
    {
        std::vector<Index> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        for (Size i = 0; i < k; ++i)
        {
            const Index j = uniformInt(toIndex(i, "Sample index exceeds RNG index range"),
                                       toIndex(n - 1, "Population size exceeds RNG index range"));
            std::swap(indices[i], indices[static_cast<Size>(j)]);
        }
        indices.resize(k);
        return indices;
    }
    else
    {
        return reservoirSample(n, k);
    }
}
std::vector<Index> RNG::bootstrap(Size n)
{
    std::vector<Index> indices;
    if (n == 0)
    {
        return indices;
    }
    ensureIndexableCount(n, "Population size exceeds RNG index range");
    indices.reserve(n);
    for (Size i = 0; i < n; ++i)
    {
        indices.push_back(uniformInt(0, toIndex(n - 1, "Population size exceeds RNG index range")));
    }
    return indices;
}
Vector<Index> RNG::weightedSubsample(ConstSpan<double> weights, Size k)
{
    if (weights.empty() || k == 0)
        return {};
    Size n = weights.size();
    if (k > n)
    {
        throw std::invalid_argument("Sample size cannot exceed population size");
    }
    ensureIndexableCount(n, "Weight count exceeds RNG index range");
    std::vector<double> cumulative(weights.size());
    double running_total = 0.0;
    for (double weight : weights)
    {
        if (!std::isfinite(weight) || weight < 0.0)
        {
            throw std::invalid_argument("Weights must be finite and non-negative");
        }
        running_total += weight;
        if (!std::isfinite(running_total))
        {
            throw std::invalid_argument("Weight total must be finite");
        }
    }
    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());
    double total_weight = cumulative.back();
    if (total_weight <= 0.0)
    {
        throw std::invalid_argument("At least one weight must be positive");
    }
    Vector<Index> selected;
    selected.reserve(k);
    for (Size i = 0; i < k; ++i)
    {
        double r = uniform(0.0, total_weight);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        if (it == cumulative.end())
        {
            it = std::prev(cumulative.end());
        }
        Index idx = static_cast<Index>(std::distance(cumulative.begin(), it));
        selected.push_back(idx);
    }
    return selected;
}
Vector<Index> RNG::stratifiedSubsample(ConstSpan<Size> stratum_sizes, Size k)
{
    Size total_size = 0;
    for (Size stratum_size : stratum_sizes)
    {
        if (stratum_size > std::numeric_limits<Size>::max() - total_size)
        {
            throw std::length_error("Stratified population size overflow");
        }
        total_size += stratum_size;
    }
    if (k > total_size)
    {
        throw std::invalid_argument("Sample size cannot exceed total population size");
    }
    if (k == 0)
    {
        return {};
    }
    ensureIndexableCount(total_size, "Stratified population exceeds RNG index range");
    std::vector<Index> result;
    result.reserve(k);
    Size current_idx = 0;
    Size remaining_samples = k;
    Size remaining_population = total_size;
    for (Size stratum_size : stratum_sizes)
    {
        if (remaining_samples == 0)
            break;
        if (stratum_size == 0)
        {
            continue;
        }
        Size stratum_samples = static_cast<Size>(
            std::round(static_cast<double>(stratum_size) * static_cast<double>(remaining_samples) /
                       static_cast<double>(remaining_population)));
        stratum_samples = std::min(stratum_samples, stratum_size);
        stratum_samples = std::min(stratum_samples, remaining_samples);
        std::vector<Index> selected = subsample(stratum_size, stratum_samples);
        for (Index idx : selected)
        {
            const Size global_index = current_idx + static_cast<Size>(idx);
            result.push_back(toIndex(global_index, "Stratified sample index exceeds RNG range"));
        }
        current_idx += stratum_size;
        remaining_samples -= stratum_samples;
        remaining_population -= stratum_size;
    }
    return result;
}
std::vector<Index> RNG::landmarkFarthest(Size n, Size k,
                                         const std::function<double(Index, Index)> &distance)
{
    if (k == 0)
    {
        return {};
    }
    if (n == 0)
    {
        throw std::invalid_argument("Population cannot be empty");
    }
    if (k > n)
    {
        throw std::invalid_argument("Number of landmarks cannot exceed population size");
    }
    ensureIndexableCount(n, "Landmark population exceeds RNG index range");
    std::vector<Index> landmarks;
    std::vector<double> minDistances(n, std::numeric_limits<double>::infinity());
    Index first = uniformInt(0, toIndex(n - 1, "Landmark population exceeds RNG index range"));
    landmarks.push_back(first);
    minDistances[first] = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        if (static_cast<Index>(i) != first)
        {
            minDistances[i] = checkedDistance(distance(first, static_cast<Index>(i)),
                                              "Landmark distance must be finite and non-negative");
        }
    }
    for (Size iter = 1; iter < k; ++iter)
    {
        auto max_it = std::ranges::max_element(minDistances);
        Index farthest = static_cast<Index>(std::distance(minDistances.begin(), max_it));
        landmarks.push_back(farthest);
        minDistances[farthest] = 0.0;
        for (Size i = 0; i < n; ++i)
        {
            if (minDistances[i] > 0.0)
            {
                double new_dist =
                    checkedDistance(distance(farthest, static_cast<Index>(i)),
                                    "Landmark distance must be finite and non-negative");
                minDistances[i] = std::min(minDistances[i], new_dist);
            }
        }
    }
    return landmarks;
}
std::vector<Index> RNG::landmarkKcenter(Size n, Size k,
                                        const std::function<double(Index, Index)> &distance)
{
    if (k == 0)
    {
        return {};
    }
    if (n == 0)
    {
        throw std::invalid_argument("Population cannot be empty");
    }
    if (k > n)
    {
        throw std::invalid_argument("Number of centers cannot exceed population size");
    }
    ensureIndexableCount(n, "K-center population exceeds RNG index range");
    std::vector<Index> centers;
    std::vector<double> minDistances(n, std::numeric_limits<double>::infinity());
    Index first = uniformInt(0, toIndex(n - 1, "K-center population exceeds RNG index range"));
    centers.push_back(first);
    for (Size iter = 1; iter < k; ++iter)
    {
        for (Size i = 0; i < n; ++i)
        {
            for (Index center : centers)
            {
                double dist = checkedDistance(distance(center, static_cast<Index>(i)),
                                              "K-center distance must be finite and non-negative");
                minDistances[i] = std::min(minDistances[i], dist);
            }
        }
        auto max_it = std::max_element(minDistances.begin(), minDistances.end());
        Index new_center = static_cast<Index>(std::distance(minDistances.begin(), max_it));
        centers.push_back(new_center);
        minDistances[new_center] = 0.0;
    }
    return centers;
}
std::vector<Index> RNG::importanceSample(ConstSpan<double> importance, Size k)
{
    return weightedSubsample(importance, k);
}
void RNG::initializeDistributions()
{
    uniform_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    normal_dist_ = std::normal_distribution<double>(0.0, 1.0);
    exp_dist_ = std::exponential_distribution<double>(1.0);
}
uint64_t RNG::makeUnseededSeed() noexcept
{
    try
    {
        std::random_device rd;
        const uint64_t hi = static_cast<uint64_t>(rd()) << 32;
        const uint64_t lo = static_cast<uint64_t>(rd());
        return hi ^ lo;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        const auto now = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto addr = reinterpret_cast<uintptr_t>(&now);
        return now ^ (static_cast<uint64_t>(addr) << 1);
    }
}
void RNG::initializeFromContract(const DeterminismContract &contract)
{
    auto violations = DeterminismEnforcer::getContractViolations(contract);
    const bool contract_satisfied = violations.empty();
    if (!contract_satisfied)
    {
        if (contract.fail_on_non_deterministic)
        {
            throw std::runtime_error(
                "Cannot satisfy determinism contract for RNG: " +
                (violations.empty() ? std::string("unknown violation") : violations[0]));
        }
    }
    if (contract.rng_seed_provided)
    {
        uint64_t seed64 = 0;
        for (size_t i = 0; i < sizeof(uint64_t) && i < contract.rng_seed.size(); ++i)
        {
            seed64 |= (static_cast<uint64_t>(contract.rng_seed[i]) << (i * 8));
        }
        seed_ = seed64;
        generator_.seed(seed64);
    }
    else
    {
        seed_ = makeUnseededSeed();
        generator_.seed(seed_);
    }
    initializeDistributions();
    is_deterministic_ = contract.enable_deterministic_random && contract.rng_seed_provided &&
                        (contract.level >= DeterminismLevel::BASIC);
    determinism_metadata_.achieved_level =
        is_deterministic_ ? (contract_satisfied ? contract.level : DeterminismLevel::BASIC)
                          : DeterminismLevel::NONE;
    determinism_metadata_.rng_seed_used = contract.rng_seed;
    determinism_metadata_.params_hash = contract.params_hash;
    determinism_metadata_.warnings = std::move(violations);
}
void RNG::updateDeterminismMetadata()
{
    determinism_metadata_.was_deterministic = is_deterministic_;
    determinism_metadata_.rng_seed_used = determinism_contract_.rng_seed;
    determinism_metadata_.params_hash = determinism_contract_.params_hash;
    auto state = getState();
    std::vector<uint8_t> state_bytes;
    for (uint64_t s : state)
    {
        detail::appendUint64LittleEndian(state_bytes, s);
    }
    SHA256_CTX ctx;
    sha256Init(&ctx);
    sha256Update(&ctx, state_bytes.data(), state_bytes.size());
    sha256Final(&ctx, determinism_metadata_.result_checksum.data());
}
std::vector<Index> RNG::reservoirSample(Size n, Size k)
{
    if (k == 0)
    {
        return {};
    }
    std::vector<Index> reservoir(k);
    for (Size i = 0; i < k; ++i)
    {
        reservoir[i] = static_cast<Index>(i);
    }
    for (Size i = k; i < n; ++i)
    {
        const Index j = uniformInt(0, toIndex(i - 1, "Reservoir sample index exceeds RNG range"));
        if (static_cast<Size>(j) < k)
        {
            reservoir[static_cast<Size>(j)] = static_cast<Index>(i);
        }
    }
    return reservoir;
}
} // namespace nerve::core
