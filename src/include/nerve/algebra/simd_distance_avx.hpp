
#pragma once
#include "nerve/algebra/simd_distance.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/errors/errors.hpp"

#include <vector>

namespace nerve::algebra
{

class EnhancedSIMDCalculator : public SIMDDistanceCalculator
{
public:
    EnhancedSIMDCalculator();

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    batchEuclideanDistances(const double *query_point, const double *target_points,
                            Size num_targets, Size dimension,
                            const core::DeterminismContract &contract = {});

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    computeDistanceMatrix(const double *points, Size num_points, Size dimension,
                          const core::DeterminismContract &contract = {});

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    computeCompressedMatrix(const double *points, Size num_points, Size dimension,
                            const core::DeterminismContract &contract = {});

private:
    using DistanceFn = double (*)(const double *, const double *, Size);
    DistanceFn distance_function_;

    void detectCapabilities();

    static double euclideanAvx512Unrolled(const double *a, const double *b, Size dim);
    static double euclideanAvx2Unrolled(const double *a, const double *b, Size dim);
    static double euclideanSse4Simd(const double *a, const double *b, Size dim);
    static double euclideanScalar(const double *a, const double *b, Size dim);

    void batchCompute4Avx2(const double *query, const double *const *targets, double *results,
                           Size dimension);

    void batchCompute4Avx512(const double *query, const double *const *targets, double *results,
                             Size dimension);
};

} // namespace nerve::algebra
