#include "nerve/algebra/simd_distance_avx.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nerve::algebra
{

namespace
{
bool squareSizeOverflows(Size n)
{
    return n != 0 && n > std::numeric_limits<Size>::max() / n;
}

bool pairCountOverflows(Size n)
{
    return n > 1 && n > std::numeric_limits<Size>::max() / (n - 1);
}

bool elementCountOverflows(Size rows, Size dimension)
{
    return dimension != 0 && rows > std::numeric_limits<Size>::max() / dimension;
}

bool vectorCapacityExceeded(Size count)
{
    return count > std::vector<double>().max_size();
}

bool valuesAreFinite(const double *values, Size count)
{
    return std::all_of(values, values + count, [](double value) { return std::isfinite(value); });
}

bool strictContractUnsatisfied(const core::DeterminismContract &contract)
{
    return contract.level == core::DeterminismLevel::STRICT &&
           !core::DeterminismEnforcer::canSatisfyContract(contract);
}

double checkedDistanceResult(double value)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error("SIMD distance overflow");
    }
    return value;
}

errors::ErrorResult<std::vector<double>> distanceOverflowError(const std::overflow_error &error)
{
    return errors::ErrorResult<std::vector<double>>::error(errors::ErrorCode::E20_NUM_NAN,
                                                           error.what());
}
} // namespace

EnhancedSIMDCalculator::EnhancedSIMDCalculator()
    : SIMDDistanceCalculator()
{
    detectCapabilities();
}

void EnhancedSIMDCalculator::detectCapabilities()
{
#ifdef __AVX512F__
    if (hasAvx512())
    {
        distance_function_ = euclideanAvx512Unrolled;
        return;
    }
#endif

#ifdef __AVX2__
    if (hasAvx2())
    {
        distance_function_ = euclideanAvx2Unrolled;
        return;
    }
#endif

#ifdef __SSE4_1__
    bool has_sse41_runtime = true;
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    has_sse41_runtime = __builtin_cpu_supports("sse4.1");
#endif
#endif
    if (has_sse41_runtime)
    {
        distance_function_ = euclideanSse4Simd;
        return;
    }
#endif

    distance_function_ = euclideanScalar;
}

errors::ErrorResult<std::vector<double>> EnhancedSIMDCalculator::batchEuclideanDistances(
    const double *query_point, const double *target_points, Size num_targets, Size dimension,
    const core::DeterminismContract &contract)
{
    if (!query_point || !target_points || dimension == 0)
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    if (num_targets == 0)
    {
        return errors::ErrorResult<std::vector<double>>::success(std::vector<double>{});
    }
    if (strictContractUnsatisfied(contract))
    {
        return errors::ErrorResult<std::vector<double>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    if (elementCountOverflows(num_targets, dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const Size target_value_count = num_targets * dimension;
    if (vectorCapacityExceeded(num_targets))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    if (!valuesAreFinite(query_point, dimension) ||
        !valuesAreFinite(target_points, target_value_count))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }

    std::vector<double> results(num_targets);
    constexpr Size batch_size = 4;
    const Size num_batches = num_targets / batch_size;

    try
    {
        for (Size batch = 0; batch < num_batches; ++batch)
        {
            Size offset = batch * batch_size;

            const double *targets[4];
            for (int i = 0; i < 4; ++i)
            {
                targets[i] = target_points + (offset + i) * dimension;
            }

            if (hasAvx512())
            {
                batchCompute4Avx512(query_point, targets, results.data() + offset, dimension);
            }
            else if (hasAvx2())
            {
                batchCompute4Avx2(query_point, targets, results.data() + offset, dimension);
            }
            else
            {
                for (int i = 0; i < 4; ++i)
                {
                    results[offset + i] = distance_function_(query_point, targets[i], dimension);
                }
            }
        }

        for (Size i = num_batches * batch_size; i < num_targets; ++i)
        {
            const double *target = target_points + i * dimension;
            results[i] = distance_function_(query_point, target, dimension);
        }
    }
    catch (const std::overflow_error &error)
    {
        return distanceOverflowError(error);
    }

    return errors::ErrorResult<std::vector<double>>::success(std::move(results));
}

errors::ErrorResult<std::vector<double>>
EnhancedSIMDCalculator::computeDistanceMatrix(const double *points, Size num_points, Size dimension,
                                              const core::DeterminismContract &contract)
{
    if (!points || num_points == 0 || dimension == 0 || squareSizeOverflows(num_points) ||
        elementCountOverflows(num_points, dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const Size point_value_count = num_points * dimension;
    const Size matrix_size = num_points * num_points;
    if (vectorCapacityExceeded(matrix_size))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    if (!valuesAreFinite(points, point_value_count))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    if (strictContractUnsatisfied(contract))
    {
        return errors::ErrorResult<std::vector<double>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    std::vector<double> distanceMatrix(matrix_size);

    try
    {
        for (Size i = 0; i < num_points; ++i)
        {
            const double *point_i = points + i * dimension;

            distanceMatrix[i * num_points + i] = 0.0;

            for (Size j = i + 1; j < num_points; ++j)
            {
                const double *point_j = points + j * dimension;
                double dist = distance_function_(point_i, point_j, dimension);

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
EnhancedSIMDCalculator::computeCompressedMatrix(const double *points, Size num_points,
                                                Size dimension,
                                                const core::DeterminismContract &contract)
{
    if (!points || num_points == 0 || dimension == 0 || pairCountOverflows(num_points) ||
        elementCountOverflows(num_points, dimension))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    const Size point_value_count = num_points * dimension;
    const Size compressed_size = num_points * (num_points - 1) / 2;
    if (vectorCapacityExceeded(compressed_size))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    if (!valuesAreFinite(points, point_value_count))
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    if (strictContractUnsatisfied(contract))
    {
        return errors::ErrorResult<std::vector<double>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    std::vector<double> compressedMatrix(compressed_size);

    try
    {
        Size index = 0;
        for (Size i = 0; i < num_points; ++i)
        {
            const double *point_i = points + i * dimension;

            for (Size j = i + 1; j < num_points; ++j)
            {
                const double *point_j = points + j * dimension;
                compressedMatrix[index++] = distance_function_(point_i, point_j, dimension);
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
