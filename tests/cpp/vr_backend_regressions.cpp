#include "nerve/persistence/vr/vr_algorithm_selector_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"
#include "nerve/persistence/vr/vr_lazy_witness_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &lhs, const Pair &rhs) {
        return std::tuple(lhs.dimension, lhs.birth, lhs.death, lhs.birth_index, lhs.death_index) <
               std::tuple(rhs.dimension, rhs.birth, rhs.death, rhs.birth_index, rhs.death_index);
    });
    return pairs;
}

void assert_same_pairs(std::vector<Pair> lhs, std::vector<Pair> rhs)
{
    lhs = canonical(std::move(lhs));
    rhs = canonical(std::move(rhs));
    assert(lhs.size() == rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        assert(lhs[i].dimension == rhs[i].dimension);
        assert(std::abs(lhs[i].birth - rhs[i].birth) < 1e-12);
        if (std::isinf(lhs[i].death) || std::isinf(rhs[i].death))
        {
            assert(std::isinf(lhs[i].death) && std::isinf(rhs[i].death));
        }
        else
        {
            assert(std::abs(lhs[i].death - rhs[i].death) < 1e-12);
        }
    }
}

std::vector<Pair> run_vr(const std::vector<double> &points, std::size_t point_dim, VRConfig config)
{
    nerve::core::BufferView<const double> view(points.data(), points.size());
    return nerve::persistence::computeVrPersistenceFast(view, point_dim, config);
}

} // namespace

int main()
{
    {
        const std::vector<double> points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        };
        VRConfig exact_config;
        exact_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        exact_config.max_radius = 2.0;
        exact_config.max_dim = 1;

        VRConfig simd_config = exact_config;
        simd_config.algorithm = VRAlgorithmSelection::FAST_SIMD;
        assert_same_pairs(run_vr(points, 2, exact_config), run_vr(points, 2, simd_config));

        exact_config.max_dim = 2;
        simd_config = exact_config;
        simd_config.algorithm = VRAlgorithmSelection::FAST_SIMD;
        assert_same_pairs(run_vr(points, 2, exact_config), run_vr(points, 2, simd_config));
    }

    {
        std::vector<double> dense_points(66 * 2, 0.0);
        VRConfig exact_config;
        exact_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        exact_config.max_radius = 1.0;
        exact_config.max_dim = 0;

        VRConfig simd_config = exact_config;
        simd_config.algorithm = VRAlgorithmSelection::FAST_SIMD;
        assert_same_pairs(run_vr(dense_points, 2, exact_config),
                          run_vr(dense_points, 2, simd_config));
    }

    {
        std::vector<double> medium_points(1024 * 2, 0.0);
        const std::vector<double> square{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        };
        std::copy(square.begin(), square.end(), medium_points.begin());
        for (std::size_t i = 4; i < 1024; ++i)
        {
            medium_points[i * 2] = 10000.0 + static_cast<double>(i) * 10.0;
            medium_points[i * 2 + 1] = -10000.0;
        }

        VRConfig exact_config;
        exact_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        exact_config.max_radius = 2.0;
        exact_config.max_dim = 1;

        nerve::core::BufferView<const double> view(medium_points.data(), medium_points.size());
        const auto exact = run_vr(medium_points, 2, exact_config);
        const auto medium =
            nerve::persistence::computeVrPersistenceMediumHybrid(view, 2, exact_config);
        assert_same_pairs(exact, medium);
    }

    {
        const std::vector<double> valid_points{
            0.0,
            0.0,
            1.0,
            0.0,
        };
        nerve::core::BufferView<const double> valid_view(valid_points.data(), valid_points.size());

        VRConfig invalid_config;
        invalid_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        invalid_config.max_radius = std::numeric_limits<double>::quiet_NaN();
        assert(run_vr(valid_points, 2, invalid_config).empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceMediumHybrid(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceLargeWitness(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceFastResult(valid_view, 2, invalid_config)
                   .isError());

        invalid_config.max_radius = -1.0;
        assert(run_vr(valid_points, 2, invalid_config).empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceMediumHybrid(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceLargeWitness(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceFastResult(valid_view, 2, invalid_config)
                   .isError());

        invalid_config.max_radius = 1.0;
        invalid_config.max_dim = std::numeric_limits<std::size_t>::max();
        assert(run_vr(valid_points, 2, invalid_config).empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceMediumHybrid(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceLargeWitness(valid_view, 2, invalid_config)
                   .empty());
        assert(nerve::persistence::computeVrPersistenceFastResult(valid_view, 2, invalid_config)
                   .isError());

        const std::vector<double> nonfinite_points{
            0.0,
            0.0,
            std::numeric_limits<double>::infinity(),
            0.0,
        };
        nerve::core::BufferView<const double> nonfinite_view(nonfinite_points.data(),
                                                             nonfinite_points.size());
        invalid_config.max_dim = 1;
        assert(run_vr(nonfinite_points, 2, invalid_config).empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(nonfinite_view, 2, invalid_config)
                   .empty());
        assert(
            nerve::persistence::computeVrPersistenceMediumHybrid(nonfinite_view, 2, invalid_config)
                .empty());
        assert(
            nerve::persistence::computeVrPersistenceLargeWitness(nonfinite_view, 2, invalid_config)
                .empty());
        assert(nerve::persistence::computeVrPersistenceFastResult(nonfinite_view, 2, invalid_config)
                   .isError());

        assert(nerve::persistence::computeVrPersistenceLargeWitness(valid_view, 0, invalid_config)
                   .empty());
    }

    {
        const std::vector<double> points{
            100.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0,
        };
        const std::vector<std::size_t> landmarks{1, 2, 3};
        nerve::persistence::LazyWitnessComplex witness(points, 2, landmarks, 1, 5.0);
        nerve::algebra::SimplicialComplex complex;
        witness.buildComplex(complex);

        const auto triangles = complex.simplicesOfDimension(2);
        assert(triangles.size() == 1);
        const nerve::algebra::Simplex triangle({1, 2, 3});
        assert(std::abs(complex.getFiltration(triangle) - 5.0) < 1e-12);

        nerve::persistence::LazyWitnessComplex invalid_witness(points, 0, landmarks, 1, 5.0);
        nerve::algebra::SimplicialComplex invalid_complex;
        invalid_witness.buildComplex(invalid_complex);
        assert(invalid_complex.getSimplices().empty());
    }

    {
        const auto empty_config = nerve::persistence::getOptimalWitnessConfig(0, 2);
        assert(empty_config.num_landmarks == 0);

        const auto invalid_dim_config = nerve::persistence::getOptimalWitnessConfig(10, 0);
        assert(invalid_dim_config.num_landmarks == 0);

        const auto tiny_config = nerve::persistence::getOptimalWitnessConfig(3, 2);
        assert(tiny_config.num_landmarks == 3);
        assert(tiny_config.strategy == nerve::persistence::LandmarkSelectionStrategy::MAXMIN);
        assert(tiny_config.approximation_factor == 3.0);

        const auto high_dim_config = nerve::persistence::getOptimalWitnessConfig(20000, 4);
        assert(high_dim_config.num_landmarks == 2000);
        assert(high_dim_config.strategy == nerve::persistence::LandmarkSelectionStrategy::DENSITY);

        const auto empty_bounds = nerve::persistence::computeWitnessApproximationBounds(0, 10, 1.0);
        assert(empty_bounds.num_landmarks_used == 0);
        assert(empty_bounds.estimated_coverage == 0.0);
        assert(std::isfinite(empty_bounds.birth_factor));
        assert(std::isfinite(empty_bounds.death_factor));
        assert(std::isfinite(empty_bounds.persistence_factor));

        const auto clamped_bounds =
            nerve::persistence::computeWitnessApproximationBounds(1000, 2500, 1.0);
        assert(clamped_bounds.num_landmarks_used == 1000);
        assert(clamped_bounds.estimated_coverage == 1.0);
        assert(clamped_bounds.birth_factor == 3.0);
        assert(clamped_bounds.death_factor == 3.0);
        assert(clamped_bounds.persistence_factor == 3.0);

        const auto invalid_radius_bounds = nerve::persistence::computeWitnessApproximationBounds(
            1000, 50, std::numeric_limits<double>::quiet_NaN());
        assert(invalid_radius_bounds.num_landmarks_used == 50);
        assert(invalid_radius_bounds.estimated_coverage == 0.0);
        assert(std::isfinite(invalid_radius_bounds.birth_factor));
        assert(std::isfinite(invalid_radius_bounds.death_factor));
        assert(std::isfinite(invalid_radius_bounds.persistence_factor));
    }

    {
        const auto nan_memory = nerve::persistence::estimateMemoryUsage(
            4096, 2, std::numeric_limits<double>::quiet_NaN());
        const auto zero_memory = nerve::persistence::estimateMemoryUsage(4096, 2, 0.0);
        assert(nan_memory == zero_memory);

        const auto infinite_memory = nerve::persistence::estimateMemoryUsage(
            100000, 2, std::numeric_limits<double>::infinity());
        const auto unit_memory = nerve::persistence::estimateMemoryUsage(100000, 2, 1.0);
        assert(infinite_memory == unit_memory);

        const auto nan_recommendation = nerve::persistence::recommendAlgorithm(
            std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<double>::quiet_NaN(), false);
        assert(std::isfinite(nan_recommendation.estimated_time_seconds));
        assert(nan_recommendation.memory_estimate_mb > 0);
        assert(std::isfinite(nan_recommendation.approximation_factor));
    }

    return 0;
}
