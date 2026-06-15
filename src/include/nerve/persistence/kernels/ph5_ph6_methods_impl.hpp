
#pragma once

#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>
#include <vector>

namespace nerve::persistence
{

namespace detail
{

template <typename Coord>
std::optional<double> coordToDouble(const Coord &value)
{
    if constexpr (std::is_arithmetic_v<std::remove_cvref_t<Coord>>)
    {
        return static_cast<double>(value);
    }
    else if constexpr (requires { value.value(); })
    {
        return static_cast<double>(value.value());
    }
    return std::nullopt;
}

inline void appendU64(std::vector<uint8_t> &data, uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        data.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFU));
    }
}

inline std::array<uint8_t, 32> hashBytes256(const std::vector<uint8_t> &data)
{
    constexpr std::array<uint64_t, 4> offsets{
        1469598103934665603ULL,
        1099511628211ULL,
        7809847782465536322ULL,
        9659303129496669493ULL,
    };

    std::array<uint8_t, 32> digest{};
    for (size_t lane = 0; lane < offsets.size(); ++lane)
    {
        uint64_t h = offsets[lane];
        for (uint8_t byte : data)
        {
            h ^= static_cast<uint64_t>(byte);
            h *= 1099511628211ULL;
        }
        std::memcpy(digest.data() + lane * 8, &h, sizeof(h));
    }
    return digest;
}

} // namespace detail

template <typename PointType, typename Scalar>
PH5PH6Engine<PointType, Scalar>::PH5PH6Engine(const Config &config)
    : config_(config)
{
    determinism_contract_ =
        core::DeterminismEnforcer::createContract(config_.determinism_level, "PH5PH6");
    determinism_contract_.enable_checksum_validation = config_.enable_checksum_validation;
    determinism_contract_.computation_id = config_.computation_id;
    determinism_contract_.setRngSeed(config_.algorithm_seeds[0]);
    determinism_contract_.rng_seed_provided = true;
    algorithm_seeds_ = config_.algorithm_seeds;
}
template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::ResultType>
PH5PH6Engine<PointType, Scalar>::computePersistenceCohomology(const PointContainer &points,
                                                              size_t max_dimension)
{
    return runCanonical(points, max_dimension, ::nerve::persistence::computePersistenceCohomology);
}

template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::ResultType>
PH5PH6Engine<PointType, Scalar>::computePersistenceCompressedWitness(const PointContainer &points,
                                                                     size_t max_dimension)
{
    return runCanonical(points, max_dimension, ::nerve::persistence::compute);
}

template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::ResultType>
PH5PH6Engine<PointType, Scalar>::computePersistenceBlockSparse(const PointContainer &points,
                                                               size_t max_dimension)
{
    return runCanonical(points, max_dimension, ::nerve::persistence::compute);
}

template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::ResultType>
PH5PH6Engine<PointType, Scalar>::computePersistenceHybrid(const PointContainer &points,
                                                          size_t max_dimension)
{
    return runCanonical(points, max_dimension, ::nerve::persistence::compute);
}

template <typename PointType, typename Scalar>
typename PH5PH6Engine<PointType, Scalar>::ComputationMetrics
PH5PH6Engine<PointType, Scalar>::getComputationMetrics() const
{
    return metrics_;
}

template <typename PointType, typename Scalar>
std::vector<std::pair<std::string, errors::ErrorCode>>
PH5PH6Engine<PointType, Scalar>::getErrorLog() const
{
    return error_log_;
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::hasErrors() const
{
    return !error_log_.empty();
}

template <typename PointType, typename Scalar>
void PH5PH6Engine<PointType, Scalar>::clearErrorLog()
{
    error_log_.clear();
}

template <typename PointType, typename Scalar>
std::string PH5PH6Engine<PointType, Scalar>::getLastError() const
{
    if (error_log_.empty())
    {
        return "No errors recorded";
    }
    return error_log_.back().first;
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::validateDeterministicResult(const PointContainer &points,
                                                                  size_t max_dimension)
{
    auto first = computePersistenceCohomology(points, max_dimension);
    auto second = computePersistenceCohomology(points, max_dimension);
    if (!first || !second || first->size() != second->size())
    {
        return false;
    }

    for (size_t i = 0; i < first->size(); ++i)
    {
        if ((*first)[i].first != (*second)[i].first)
        {
            return false;
        }
        if (std::abs((*first)[i].second - (*second)[i].second) > config_.numerical_tolerance)
        {
            return false;
        }
    }
    return true;
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::runStabilityTest(const PointContainer &points,
                                                       size_t max_dimension, size_t num_runs)
{
    if (num_runs == 0)
    {
        return false;
    }

    std::optional<ResultType> reference;
    for (size_t i = 0; i < num_runs; ++i)
    {
        auto result = computePersistenceCohomology(points, max_dimension);
        if (!result)
        {
            return false;
        }
        if (!reference)
        {
            reference = std::move(result);
        }
        else if (*reference != *result)
        {
            return false;
        }
    }
    return true;
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::setDeterminismContract(
    const core::DeterminismContract &contract)
{
    if (!contract.isValid())
    {
        recordError("Invalid determinism contract", errors::ErrorCode::E30_DET_MISMATCH);
        return false;
    }

    determinism_contract_ = contract;
    config_.determinism_level = contract.level;
    config_.enable_checksum_validation = contract.enable_checksum_validation;
    config_.computation_id = contract.computation_id;
    return true;
}

template <typename PointType, typename Scalar>
core::DeterminismContract PH5PH6Engine<PointType, Scalar>::getDeterminismContract() const
{
    return determinism_contract_;
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::validateDeterminismContract() const
{
    return determinism_contract_.isValid();
}

template <typename PointType, typename Scalar>
std::array<uint8_t, 32>
PH5PH6Engine<PointType, Scalar>::computeResultChecksum(const ResultType &result) const
{
    std::vector<uint8_t> data;
    data.reserve(result.size() * 24);

    for (const auto &pair : result)
    {
        detail::appendU64(data, static_cast<uint64_t>(pair.first.size()));
        for (size_t v : pair.first)
        {
            detail::appendU64(data, static_cast<uint64_t>(v));
        }
        const double value = static_cast<double>(pair.second);
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(value));
        detail::appendU64(data, bits);
    }
    return detail::hashBytes256(data);
}

template <typename PointType, typename Scalar>
bool PH5PH6Engine<PointType, Scalar>::validateResultChecksum(
    const ResultType &result, const std::array<uint8_t, 32> &expected_checksum) const
{
    return computeResultChecksum(result) == expected_checksum;
}

template <typename PointType, typename Scalar>
void PH5PH6Engine<PointType, Scalar>::computeParamsHash(const PointContainer &points,
                                                        size_t max_dimension)
{
    std::vector<uint8_t> data;
    data.reserve(points.size() * 16);
    detail::appendU64(data, static_cast<uint64_t>(points.size()));
    detail::appendU64(data, static_cast<uint64_t>(max_dimension));
    for (uint64_t seed : algorithm_seeds_)
    {
        detail::appendU64(data, seed);
    }

    for (const auto &point : points)
    {
        if constexpr (requires {
                          point.size();
                          point[0];
                      })
        {
            detail::appendU64(data, static_cast<uint64_t>(point.size()));
            for (size_t i = 0; i < point.size(); ++i)
            {
                const std::optional<double> converted = detail::coordToDouble(point[i]);
                uint64_t bits = 0;
                if (converted.has_value())
                {
                    const double value = converted.value();
                    std::memcpy(&bits, &value, sizeof(value));
                }
                else
                {
                    bits = 0xFFFFFFFFFFFFFFFFULL;
                }
                detail::appendU64(data, bits);
            }
        }
    }

    current_params_hash_ = detail::hashBytes256(data);
    determinism_contract_.params_hash = current_params_hash_;
    determinism_contract_.params_hash_valid = true;
    metrics_.params_hash = current_params_hash_;
}

template <typename PointType, typename Scalar>
std::array<uint8_t, 32> PH5PH6Engine<PointType, Scalar>::getParamsHash() const
{
    return current_params_hash_;
}

template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::FlattenedPoints>
PH5PH6Engine<PointType, Scalar>::flattenPoints(const PointContainer &points,
                                               std::string &error_message)
{
    if (points.empty())
    {
        error_message = "Empty input point set";
        return std::nullopt;
    }

    if constexpr (!(requires {
                      points[0].size();
                      points[0][0];
                  }))
    {
        error_message = "Point type must provide size() and operator[]";
        return std::nullopt;
    }

    const Size point_dim = static_cast<Size>(points.front().size());
    if (point_dim == 0)
    {
        error_message = "Point dimension cannot be zero";
        return std::nullopt;
    }

    FlattenedPoints flattened;
    flattened.point_dim = point_dim;
    flattened.num_points = static_cast<Size>(points.size());
    flattened.values.reserve(points.size() * point_dim);

    for (const auto &point : points)
    {
        if (static_cast<Size>(point.size()) != point_dim)
        {
            error_message = "Inconsistent point dimensions";
            return std::nullopt;
        }
        for (size_t d = 0; d < point_dim; ++d)
        {
            const std::optional<double> converted = detail::coordToDouble(point[d]);
            if (!converted.has_value())
            {
                error_message = "Point coordinate type must be arithmetic or expose value()";
                return std::nullopt;
            }
            const double value = converted.value();
            if (!std::isfinite(value))
            {
                error_message = "Point coordinates must be finite";
                return std::nullopt;
            }
            flattened.values.push_back(value);
        }
    }
    return flattened;
}

} // namespace nerve::persistence
