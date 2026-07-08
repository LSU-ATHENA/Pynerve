#pragma once
#include "nerve/algebra/simd_distance.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <vector>

namespace nerve::algebra
{

class SIMDCalculator : public SIMDDistanceCalculator
{
public:
    SIMDCalculator();

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    batchEuclideanDistances(const double *query_point, const double *target_points,
                            std::size_t num_targets, std::size_t dimension,
                            const core::DeterminismContract &contract = {});

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    computeDistanceMatrix(const double *points, std::size_t num_points, std::size_t dimension,
                          const core::DeterminismContract &contract = {});

    [[nodiscard]] errors::ErrorResult<std::vector<double>>
    computeCompressedMatrix(const double *points, std::size_t num_points, std::size_t dimension,
                            const core::DeterminismContract &contract = {});
};

} // namespace nerve::algebra
