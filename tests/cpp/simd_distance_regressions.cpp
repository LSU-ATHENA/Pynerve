#include "nerve/algebra/simd_distance_avx.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

nerve::Size squareSideAboveVectorCapacity()
{
    const nerve::Size capacity = std::vector<double>().max_size();
    nerve::Size side = static_cast<nerve::Size>(std::sqrt(static_cast<long double>(capacity))) + 1;
    while (side <= capacity / side)
    {
        ++side;
    }
    return side;
}

nerve::Size compressedSideAboveVectorCapacity()
{
    const nerve::Size capacity = std::vector<double>().max_size();
    nerve::Size side =
        static_cast<nerve::Size>(
            (1.0L + std::sqrt(1.0L + 8.0L * static_cast<long double>(capacity))) / 2.0L) +
        1;
    while (side > 1 && side <= std::numeric_limits<nerve::Size>::max() / (side - 1) &&
           (side * (side - 1)) / 2 <= capacity)
    {
        ++side;
    }
    return side;
}

void assertResourceLimit(const nerve::errors::ErrorResult<std::vector<double>> &result)
{
    assert(result.isError());
    assert(result.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);
}

} // namespace

int main()
{
    nerve::algebra::SIMDCalculator calculator;
    nerve::algebra::SIMDDistanceCalculator base_calculator;

    const double points[] = {0.0, 0.0, 3.0, 4.0};
    const auto batch = calculator.batchEuclideanDistances(points, points + 2, 1, 2);
    assert(batch.isSuccess());
    assert(batch.value().size() == 1);
    assert(batch.value()[0] == 5.0);

    const auto matrix = calculator.computeDistanceMatrix(points, 2, 2);
    assert(matrix.isSuccess());
    assert(matrix.value().size() == 4);
    assert(matrix.value()[1] == 5.0);
    assert(matrix.value()[2] == 5.0);

    const auto compressed = calculator.computeCompressedMatrix(points, 2, 2);
    assert(compressed.isSuccess());
    assert(compressed.value().size() == 1);
    assert(compressed.value()[0] == 5.0);

    const double invalid_points[] = {0.0, std::numeric_limits<double>::infinity()};
    assert(!calculator.batchEuclideanDistances(points, invalid_points, 1, 2).isSuccess());
    assert(!calculator.computeDistanceMatrix(invalid_points, 1, 2).isSuccess());
    assert(!calculator.computeCompressedMatrix(invalid_points, 1, 2).isSuccess());

    const double zero_point[] = {0.0};
    const double huge_point[] = {std::numeric_limits<double>::max()};
    bool rejected_euclidean_overflow = false;
    try
    {
        (void)base_calculator.euclideanDistance(zero_point, huge_point, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_euclidean_overflow = true;
    }
    assert(rejected_euclidean_overflow);

    const double overflow_batch_points[] = {0.0, std::numeric_limits<double>::max()};
    bool rejected_batch_overflow = false;
    try
    {
        (void)base_calculator.batchEuclideanDistances(overflow_batch_points, 2, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_batch_overflow = true;
    }
    assert(rejected_batch_overflow);

    const double negative_huge_point[] = {-std::numeric_limits<double>::max()};
    bool rejected_manhattan_overflow = false;
    try
    {
        (void)base_calculator.manhattanDistance(huge_point, negative_huge_point, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_manhattan_overflow = true;
    }
    assert(rejected_manhattan_overflow);

    const double huge_cosine_point[] = {std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max()};
    bool rejected_cosine_overflow = false;
    try
    {
        (void)base_calculator.cosineDistance(huge_cosine_point, huge_cosine_point, 2);
    }
    catch (const std::overflow_error &)
    {
        rejected_cosine_overflow = true;
    }
    assert(rejected_cosine_overflow);

    assert(calculator.batchEuclideanDistances(zero_point, huge_point, 1, 1).isError());
    assert(calculator.computeDistanceMatrix(overflow_batch_points, 2, 1).isError());
    assert(calculator.computeCompressedMatrix(overflow_batch_points, 2, 1).isError());

    const double scalar_point[] = {0.0};
    const nerve::Size vector_capacity = std::vector<double>().max_size();
    if (vector_capacity < std::numeric_limits<nerve::Size>::max())
    {
        assertResourceLimit(
            calculator.batchEuclideanDistances(scalar_point, scalar_point, vector_capacity + 1, 1));
    }

    assertResourceLimit(
        calculator.computeDistanceMatrix(scalar_point, squareSideAboveVectorCapacity(), 1));
    assertResourceLimit(
        calculator.computeCompressedMatrix(scalar_point, compressedSideAboveVectorCapacity(), 1));

    assert(!calculator
                .computeDistanceMatrix(scalar_point,
                                       std::numeric_limits<nerve::Size>::max() / 2 + 1, 3)
                .isSuccess());
    assert(!calculator
                .computeCompressedMatrix(scalar_point,
                                         std::numeric_limits<nerve::Size>::max() / 2 + 1, 1)
                .isSuccess());

    return 0;
}
