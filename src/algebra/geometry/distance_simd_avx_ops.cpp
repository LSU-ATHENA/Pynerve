#include "nerve/algebra/simd_distance_avx.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nerve::algebra
{

namespace
{
bool squareSizeOverflows(std::size_t n)
{
    return n != 0 && n > std::numeric_limits<std::size_t>::max() / n;
}

bool pairCountOverflows(std::size_t n)
{
    return n > 1 && n > std::numeric_limits<std::size_t>::max() / (n - 1);
}

bool elementCountOverflows(std::size_t rows, std::size_t dimension)
{
    return dimension != 0 && rows > std::numeric_limits<std::size_t>::max() / dimension;
}

bool vectorCapacityExceeded(std::size_t count)
{
    return count > std::vector<double>().max_size();
}

bool valuesAreFinite(const double *values, std::size_t count)
{
    return std::all_of(values, values + count, [](double value) { return std::isfinite(value); });
}

double checkedDistanceResult(double value)
{
    if (!std::isfinite(value))
        throw std::overflow_error("SIMD distance overflow");
    return value;
}

errors::ErrorResult<std::vector<double>> distanceOverflowError(const std::overflow_error &error)
{
    return errors::ErrorResult<std::vector<double>>::error(errors::ErrorCode::E20_NUM_NAN,
                                                           error.what());
}
} // anonymous namespace

SIMDCalculator::SIMDCalculator()
    : SIMDDistanceCalculator()
{
    nerve::simd::simd_init();
}

errors::ErrorResult<std::vector<double>>
SIMDCalculator::batchEuclideanDistances(const double *query_point, const double *target_points,
                                        std::size_t num_targets, std::size_t dimension,
                                        const core::DeterminismContract &contract)
{
    if (!query_point || !target_points || dimension == 0)
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    if (num_targets == 0)
        return errors::ErrorResult<std::vector<double>>::success(std::vector<double>{});
    if (elementCountOverflows(num_targets, dimension))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    if (vectorCapacityExceeded(num_targets))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    if (!valuesAreFinite(query_point, dimension) ||
        !valuesAreFinite(target_points, num_targets * dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    std::vector<double> results(num_targets);
    constexpr std::size_t batch_size = 4;
    const std::size_t num_batches = num_targets / batch_size;

    try
    {
        for (std::size_t batch = 0; batch < num_batches; ++batch)
        {
            std::size_t offset = batch * batch_size;
            for (int i = 0; i < 4; ++i)
            {
                const double *t =
                    target_points + (offset + static_cast<std::size_t>(i)) * dimension;
                results[offset + static_cast<std::size_t>(i)] =
                    checkedDistanceResult(nerve::simd::simd_euclidean(query_point, t, dimension));
            }
        }
        for (std::size_t i = num_batches * batch_size; i < num_targets; ++i)
        {
            const double *target = target_points + i * dimension;
            results[i] =
                checkedDistanceResult(nerve::simd::simd_euclidean(query_point, target, dimension));
        }
    }
    catch (const std::overflow_error &error)
    {
        return distanceOverflowError(error);
    }

    return errors::ErrorResult<std::vector<double>>::success(std::move(results));
}

errors::ErrorResult<std::vector<double>>
SIMDCalculator::computeDistanceMatrix(const double *points, std::size_t num_points,
                                      std::size_t dimension,
                                      const core::DeterminismContract &contract)
{
    if (!points || num_points == 0 || dimension == 0 || squareSizeOverflows(num_points) ||
        elementCountOverflows(num_points, dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const std::size_t point_value_count = num_points * dimension;
    const std::size_t matrix_size = num_points * num_points;
    if (vectorCapacityExceeded(matrix_size))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    if (!valuesAreFinite(points, point_value_count))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);

    std::vector<double> distanceMatrix(matrix_size);

    try
    {
        for (std::size_t i = 0; i < num_points; ++i)
        {
            const double *pi = points + i * dimension;
            distanceMatrix[i * num_points + i] = 0.0;

            for (std::size_t j = i + 1; j < num_points; ++j)
            {
                const double *pj = points + j * dimension;
                double dist = checkedDistanceResult(nerve::simd::simd_euclidean(pi, pj, dimension));
                distanceMatrix[i * num_points + j] = dist;
                distanceMatrix[j * num_points + i] = dist;
            }
        }
    }
    catch (const std::overflow_error &error)
    {
        return distanceOverflowError(error);
    }

    return errors::ErrorResult<std::vector<double>>::success(std::move(distanceMatrix));
}

errors::ErrorResult<std::vector<double>>
SIMDCalculator::computeCompressedMatrix(const double *points, std::size_t num_points,
                                        std::size_t dimension,
                                        const core::DeterminismContract &contract)
{
    if (!points || num_points == 0 || dimension == 0 || pairCountOverflows(num_points) ||
        elementCountOverflows(num_points, dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const std::size_t point_value_count = num_points * dimension;
    const std::size_t compressed_size = num_points * (num_points - 1) / 2;
    if (vectorCapacityExceeded(compressed_size))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    if (!valuesAreFinite(points, point_value_count))
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);

    std::vector<double> compressedMatrix(compressed_size);

    try
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < num_points; ++i)
        {
            const double *pi = points + i * dimension;
            for (std::size_t j = i + 1; j < num_points; ++j)
            {
                const double *pj = points + j * dimension;
                compressedMatrix[index++] =
                    checkedDistanceResult(nerve::simd::simd_euclidean(pi, pj, dimension));
            }
        }
    }
    catch (const std::overflow_error &error)
    {
        return distanceOverflowError(error);
    }

    return errors::ErrorResult<std::vector<double>>::success(std::move(compressedMatrix));
}

} // namespace nerve::algebra
