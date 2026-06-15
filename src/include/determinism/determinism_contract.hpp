#pragma once
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::determinism
{

struct VectorIndexHash
{
    std::size_t operator()(const std::vector<nerve::Index> &key) const noexcept
    {
        std::size_t hash = 1469598103934665603ULL;
        for (nerve::Index value : key)
        {
            hash ^= static_cast<std::size_t>(value);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

template <typename T>
concept ValidDeterminismLevel = std::integral<T> && requires(T l) {
    { l } -> std::convertible_to<int>;
    requires l >= 0 && l <= 3;
};

enum class DeterminismLevel
{
    STRONG,  // Run multiple times -> always same result
    WEAK,    // Same binary + same hardware + same thread count -> same result
    NONE,    // GPU or parallel with floating-point reduction: not guaranteed
    UNKNOWN, // Unknown: needs analysis
};

struct CanonicalOrder
{
    [[nodiscard]] bool operator()(const nerve::Simplex &a, const nerve::Simplex &b) const noexcept
    {
        if (a.value != b.value)
            return a.value < b.value;
        if (a.vertices.size() != b.vertices.size())
            return a.vertices.size() < b.vertices.size();
        return a.vertices < b.vertices;
    }
};

nerve::error::Result<void> canonicalizeFiltration(std::vector<nerve::Simplex> &filtration)
{
    std::sort(filtration.begin(), filtration.end(), CanonicalOrder{});

    std::unordered_map<std::vector<nerve::Index>, nerve::Size, VectorIndexHash> pos;
    pos.reserve(filtration.size());

    for (nerve::Size i = 0; i < filtration.size(); ++i)
    {
        const auto &sigma = filtration[i];
        pos[sigma.vertices] = i;

        if (sigma.vertices.size() >= 2)
        {
            for (nerve::Size face_idx = 0; face_idx < sigma.vertices.size(); ++face_idx)
            {
                std::vector<nerve::Index> face;
                for (nerve::Size p = 0; p < sigma.vertices.size(); ++p)
                    if (p != face_idx)
                        face.push_back(sigma.vertices[p]);

                auto it = pos.find(face);
                if (it == pos.end() || it->second >= i)
                    return nerve::error::Result<void>::err(
                        nerve::error::TDAErrorCode::NonMonotoneFiltration,
                        "canonical ordering violates face constraint at simplex " +
                            std::to_string(i));
            }
        }
    }
    return nerve::error::Result<void>::ok();
}

struct GPUDeterminismConfig
{
    // Strategy 1: don't use atomics at all (sort-then-reduce)

    // Strategy 2: use double precision to reduce impact of ordering
    bool force_fp64 = false;

    DeterminismLevel effectiveLevel() const noexcept
    {
        if (avoid_atomics)
            return DeterminismLevel::WEAK;
        return DeterminismLevel::NONE;
    }
};

// Run computation N times, verify all results identical
template <typename ComputeFn>
nerve::error::Result<void> verifyDeterministic(ComputeFn &&fn, int n_runs = 10,
                                               std::string_view algorithm_name = "unknown")
{
    auto first = fn();
    if (!first.isOk())
    {
        return nerve::error::Result<void>::err(
            static_cast<nerve::error::TDAErrorCode>(first.error().value()),
            std::string(first.detail()), first.where());
    }

    for (int run = 1; run < n_runs; ++run)
    {
        auto result = fn();
        if (!result.isOk())
        {
            return nerve::error::Result<void>::err(
                static_cast<nerve::error::TDAErrorCode>(result.error().value()),
                std::string(result.detail()), result.where());
        }

        // Compare  --  requires operator== on ResultT::value_type
        if (result.value() != first.value())
        {
            return nerve::error::Result<void>::err(
                nerve::error::TDAErrorCode::DeterminismViolation,
                std::string(algorithm_name) + " produced different results on run " +
                    std::to_string(run) + " of " + std::to_string(n_runs) +
                    ". This algorithm is NOT deterministic.");
        }
    }
    return nerve::error::Result<void>::ok();
}

class DeterministicRNG
{
public:
    explicit DeterministicRNG(uint64_t seed = 42)
        : rng_(seed)
    {}

    // Generate deterministic sequence
    template <typename T>
    T generate()
    {
        if constexpr (std::is_integral_v<T>)
        {
            return static_cast<T>(rng_());
        }
        else
        {
            std::uniform_real_distribution<T> dist(0.0, 1.0);
            return dist(rng_);
        }
    }

    // Get reproducible permutation
    std::vector<nerve::Size> permutation(nerve::Size n)
    {
        std::vector<nerve::Size> perm(n);
        for (nerve::Size i = 0; i < n; ++i)
            perm[i] = i;

        std::shuffle(perm.begin(), perm.end(), rng_);
        return perm;
    }

private:
    std::mt19937_64 rng_;
};

struct DeterministicHash
{
    std::size_t operator()(const std::vector<nerve::Index> &key) const noexcept
    {
        // FNV-1a hash - deterministic across platforms
        std::size_t hash = 1469598103934665603ULL;
        for (nerve::Index x : key)
        {
            hash ^= static_cast<std::size_t>(x);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

template <typename K, typename V>
using DeterministicMap = std::unordered_map<K, V, DeterministicHash>;

struct DeterminismReport
{
    DeterminismLevel level = DeterminismLevel::UNKNOWN;
    std::string algorithm_name;
    bool passed_tests = false;
    std::vector<std::string> warnings;
    std::vector<std::string> recommendations;

    void addWarning(const std::string &warning) { warnings.push_back(warning); }

    void addRecommendation(const std::string &recommendation)
    {
        recommendations.push_back(recommendation);
    }
};

// Analyze an algorithm for determinism
template <typename Algorithm>
class DeterminismAnalyzer
{
public:
    static DeterminismReport analyze(const std::string &name)
    {
        DeterminismReport report;
        report.algorithm_name = name;

        // Check if algorithm uses floating-point parallel reduction
        if constexpr (requires { Algorithm::uses_parallel_fp_reduction(); })
        {
            if (Algorithm::uses_parallel_fp_reduction())
            {
                report.level = DeterminismLevel::NONE;
                report.addWarning("Algorithm uses parallel floating-point reduction");
                report.addRecommendation(
                    "Use deterministic reduction strategy or accept non-determinism");
            }
        }

        // Check if algorithm uses GPU atomics
        if constexpr (requires { Algorithm::uses_gpu_atomics(); })
        {
            if (Algorithm::uses_gpu_atomics())
            {
                report.level = DeterminismLevel::NONE;
                report.addWarning("Algorithm uses GPU atomic operations");
                report.addRecommendation("Use sort-then-reduce strategy or CPU implementation");
            }
        }

        // Default to weak determinism for CPU algorithms
        if (report.level == DeterminismLevel::UNKNOWN)
        {
            report.level = DeterminismLevel::WEAK;
            report.addRecommendation("Use canonical filtration ordering for strong determinism");
        }

        return report;
    }
};

// Wraps any reduction algorithm to make it deterministic
template <typename BaseAlgorithm>
class DeterministicWrapper
{
public:
    using Config = typename BaseAlgorithm::Config;

    explicit DeterministicWrapper(Config cfg = {})
        : algorithm_(cfg)
    {}

    nerve::error::Result<std::vector<nerve::Pair>> compute(std::vector<nerve::Simplex> filtration)
    {
        // Canonicalize filtration order
        auto canon_result = canonicalizeFiltration(filtration);
        if (!canon_result.isOk())
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                static_cast<nerve::error::TDAErrorCode>(canon_result.error().value()),
                std::string(canon_result.detail()), canon_result.where());
        }

        // Run algorithm
        auto result = algorithm_.compute(filtration);
        if (!result.isOk())
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                static_cast<nerve::error::TDAErrorCode>(result.error().value()),
                std::string(result.detail()), result.where());
        }

        // Sort output for consistency
        auto pairs = result.value();
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            if (a.dimension != b.dimension)
                return a.dimension < b.dimension;
            if (a.birth != b.birth)
                return a.birth < b.birth;
            return a.death < b.death;
        });

        return nerve::error::Result<std::vector<nerve::Pair>>::ok(pairs);
    }

private:
    BaseAlgorithm algorithm_;
};

} // namespace nerve::determinism
