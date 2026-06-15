
#pragma once
#include "determinism_contract.hpp"
#include "nerve/core_types.hpp"

#include <concepts>
#include <functional>
#include <random>
#include <ranges>
#include <vector>

namespace nerve::core
{

class RNG
{
public:
    RNG();
    explicit RNG(uint64_t seed);
    explicit RNG(const DeterminismContract &contract);
    [[nodiscard]] uint64_t seed() const noexcept;
    void seed(uint64_t new_seed);
    void seed(const DeterminismContract &contract);
    [[nodiscard]] DeterminismMetadata getDeterminismMetadata() const;
    [[nodiscard]] bool isDeterministic() const noexcept;
    [[nodiscard]] double uniform();
    [[nodiscard]] double uniform(double min, double max);
    [[nodiscard]] Index uniformInt(Index min, Index max);
    [[nodiscard]] double normal(double mean = 0.0, double stddev = 1.0);
    [[nodiscard]] double exponential(double lambda = 1.0);
    template <std::ranges::range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Index>
    Vector<Index> subsampleFromRange(const R &range, Size k);
    Vector<Index> subsample(Size n, Size k);
    Vector<Index> bootstrap(Size n);
    Vector<Index> weightedSubsample(ConstSpan<double> weights, Size k);
    Vector<Index> stratifiedSubsample(ConstSpan<Size> stratum_sizes, Size k);
    std::vector<Index> landmarkFarthest(Size n, Size k,
                                        const std::function<double(Index, Index)> &distance);
    std::vector<Index> landmarkKcenter(Size n, Size k,
                                       const std::function<double(Index, Index)> &distance);
    std::vector<Index> importanceSample(ConstSpan<double> importance, Size k);
    std::vector<double> haltonSequence(Size n, Size base = 2);
    std::vector<double> sobolSequence(Size n, Size dimension = 1);
    void jumpAhead(Size steps);
    RNG split();
    std::vector<uint64_t> getState() const;
    void setState(const ConstSpan<uint64_t> &state);

private:
    uint64_t seed_;
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
    std::exponential_distribution<double> exp_dist_;
    DeterminismContract determinism_contract_;
    DeterminismMetadata determinism_metadata_;
    bool is_deterministic_ = false;
    static uint64_t makeUnseededSeed() noexcept;
    void initializeDistributions();
    void initializeFromContract(const DeterminismContract &contract);
    void updateDeterminismMetadata();
    std::vector<Index> reservoirSample(Size n, Size k);
    double halton(Size index, Size base);
};
RNG &getGlobalRng();
class ScopedRNG
{
public:
    explicit ScopedRNG(uint64_t seed)
        : old_rng_(getGlobalRng())
        , local_rng_(seed)
    {}
    explicit ScopedRNG(const DeterminismContract &contract)
        : old_rng_(getGlobalRng())
        , local_rng_(contract)
    {}
    ~ScopedRNG() { getGlobalRng() = old_rng_; }
    RNG &get() { return local_rng_; }

private:
    RNG old_rng_;
    RNG local_rng_;
};
class DeterministicRNGFactory
{
public:
    static RNG createFromContract(const DeterminismContract &contract);
    static std::vector<RNG> createSplitGenerators(const DeterminismContract &contract, Size count);
    static bool validateDeterminism(const RNG &rng, const DeterminismContract &contract);
    static std::vector<std::string> getDeterminismWarnings(const RNG &rng,
                                                           const DeterminismContract &contract);
};
} // namespace nerve::core
