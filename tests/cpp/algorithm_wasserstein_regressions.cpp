#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"

int main() {
  {
    const std::vector<double> points{
        0.0,
        0.0,
        3.0,
        4.0,
    };
    const auto tiled =
        nerve::persistence::computeDistanceMatrixTiled(points, 2, 2, 1);
    assert(tiled.size() == 4);
    assert(tiled[1] == 5.0);
    assert(tiled[2] == 5.0);

    const auto numa =
        nerve::persistence::computeDistanceMatrixNumaAware(points, 2, 2, 1);
    assert(numa.size() == 4);
    assert(numa[1] == 5.0);
    assert(numa[2] == 5.0);

    const auto invalid =
        nerve::persistence::computeDistanceMatrixTiled(points, 0, 2, 1);
    assert(invalid.empty());
  }

  {
    using nerve::core::ownership_utils::PointView;
    using nerve::filtration::VietorisRips;

    const std::vector<double> points{
        0.0, 0.0, 1.0, 0.0, 0.0, 2.0,
    };
    const PointView point_view(points);

    auto distances =
        nerve::filtration::computeAllPairDistances(point_view, 2);
    assert(distances.size() == 3);
    assert(distances.dimension() == 3);
    assert(distances.getCoordinate(0, 0) == 0.0);
    assert(distances.getCoordinate(0, 1) == 1.0);
    assert(distances.getCoordinate(1, 0) == 1.0);
    assert(distances.getCoordinate(0, 2) == 2.0);
    assert(distances.getCoordinate(2, 0) == 2.0);

    const auto neighbors =
        nerve::filtration::findKNearestNeighbors(point_view, 2, 0, 1);
    assert(neighbors.size() == 1);
    assert(neighbors[0].first == 1);
    assert(neighbors[0].second == 0);

    nerve::filtration::WeightedVietorisRips weighted({0.5, 1.0, 1.5});
    const auto weighted_filtration = weighted.buildFiltration(point_view, 2);
    assert(!weighted_filtration.empty());
    assert(nerve::filtration::computeWeightedVietorisRips(
               point_view, 2, {0.5, 1.0, 1.5},
               std::numeric_limits<double>::infinity())
               .empty());

    const std::vector<double> weighted_overflow_points{
        std::numeric_limits<double>::max(), 0.0,
        -std::numeric_limits<double>::max(), 0.0,
    };
    const PointView weighted_overflow_view(weighted_overflow_points);
    nerve::filtration::WeightedVietorisRips weighted_overflow({1.0, 1.0});
    bool rejected_weighted_overflow = false;
    try {
      (void)weighted_overflow.buildFiltration(weighted_overflow_view, 2);
    } catch (const std::invalid_argument &) {
      rejected_weighted_overflow = true;
    }
    assert(rejected_weighted_overflow);

    const PointView null_points(nullptr, 2);
    assert(
        nerve::filtration::computeAllPairDistances(null_points, 1).empty());
    assert(nerve::filtration::findKNearestNeighbors(null_points, 1, 0, 1)
               .empty());

    VietorisRips vr;
    bool rejected_null = false;
    try {
      (void)vr.computeDistance(null_points, null_points, 1);
    } catch (const std::invalid_argument &) {
      rejected_null = true;
    }
    assert(rejected_null);

    const double dummy = 0.0;
    const PointView huge_view(&dummy,
                              std::numeric_limits<nerve::Size>::max());
    bool rejected_overflow = false;
    try {
      (void)vr.computeDistanceMatrix(huge_view, 1);
    } catch (const std::length_error &) {
      rejected_overflow = true;
    }
    assert(rejected_overflow);
    assert(nerve::filtration::computeAllPairDistances(huge_view, 1).empty());

    nerve::filtration::SparseVietorisRips sparse(1);
    bool rejected_sparse_overflow = false;
    try {
      (void)sparse.buildFiltration(huge_view, 1);
    } catch (const std::length_error &) {
      rejected_sparse_overflow = true;
    }
    assert(rejected_sparse_overflow);

    nerve::algebra::Simplex edge({0, 1});
    const std::vector<double> simplex_points{0.0, 0.0, 2.0, 0.0};
    assert(edge.circumradius(PointView(simplex_points)) == 1.0);
    assert(edge.volume(PointView(simplex_points)) == 2.0);

    bool rejected_simplex_nan_coordinate = false;
    try {
      const std::vector<double> invalid_simplex_points{
          0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
      (void)edge.circumradius(PointView(invalid_simplex_points));
    } catch (const std::invalid_argument &) {
      rejected_simplex_nan_coordinate = true;
    }
    assert(rejected_simplex_nan_coordinate);

    bool rejected_simplex_overflow_radius = false;
    try {
      const std::vector<double> huge_simplex_points{0.0, 0.0, 1.0e308, 0.0};
      (void)edge.circumradius(PointView(huge_simplex_points));
    } catch (const std::overflow_error &) {
      rejected_simplex_overflow_radius = true;
    }
    assert(rejected_simplex_overflow_radius);

    bool rejected_simplex_overflow_volume = false;
    try {
      const std::vector<double> huge_simplex_points{0.0, 0.0, 1.0e308, 0.0};
      (void)edge.volume(PointView(huge_simplex_points));
    } catch (const std::overflow_error &) {
      rejected_simplex_overflow_volume = true;
    }
    assert(rejected_simplex_overflow_volume);
  }

  {
    using nerve::persistence::vram::Algorithm;
    using nerve::persistence::vram::VRAMConfig;

    VRAMConfig config;
    config.total_vram_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    config.available_vram_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    config.safety_fraction = 0.5;
    assert(config.safeBytes() == 32ULL * 1024ULL * 1024ULL * 1024ULL);
    assert(config.select(1024, 3) == Algorithm::FULL_GPU);
    assert(nerve::persistence::vram::selectAlgorithm(
               1024, 3, 64ULL * 1024ULL * 1024ULL * 1024ULL) ==
           Algorithm::FULL_GPU);
  }

  {
    nerve::persistence::Diagram empty_a;
    nerve::persistence::Diagram empty_b;
    nerve::metrics::WassersteinDistance wasserstein;
    assert(wasserstein.computeWithOrder(empty_a, empty_b, 2.0) == 0.0);
    wasserstein.setOrder(1.5);
    wasserstein.setRegularization(1e-3);

    nerve::persistence::Diagram order_diagram;
    order_diagram.addPair({0.0, 4.0, 0});
    assert(wasserstein.computeWithOrder(order_diagram, empty_b, 1.0) == 2.0);

    bool rejected_wasserstein_nan_order = false;
    try {
      wasserstein.setOrder(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_wasserstein_nan_order = true;
    }
    assert(rejected_wasserstein_nan_order);

    bool rejected_wasserstein_small_order = false;
    try {
      (void)wasserstein.computeWithOrder(empty_a, empty_b, 0.5);
    } catch (const std::invalid_argument &) {
      rejected_wasserstein_small_order = true;
    }
    assert(rejected_wasserstein_small_order);

    bool rejected_wasserstein_nan_regularization = false;
    try {
      wasserstein.setRegularization(
          std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_wasserstein_nan_regularization = true;
    }
    assert(rejected_wasserstein_nan_regularization);

    nerve::persistence::Diagram finite_diagram;
    finite_diagram.addPair({0.0, 1.0, 0});
    assert(std::isfinite(
        nerve::metrics::bottleneckDistance(finite_diagram, finite_diagram)));
    nerve::metrics::BottleneckDistance bottleneck_metric;
    assert(std::isfinite(
        bottleneck_metric.compute(finite_diagram, finite_diagram)));

    bool rejected_bottleneck_nan_tolerance = false;
    try {
      bottleneck_metric.setTolerance(
          std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_bottleneck_nan_tolerance = true;
    }
    assert(rejected_bottleneck_nan_tolerance);

    nerve::persistence::Diagram essential_diagram;
    essential_diagram.addPair(
        {0.0, std::numeric_limits<double>::infinity(), 0});
    assert(std::isfinite(nerve::metrics::wassersteinDistance(
        essential_diagram, essential_diagram, 2.0)));

    nerve::persistence::Diagram invalid_birth_diagram;
    invalid_birth_diagram.addPair(
        {std::numeric_limits<double>::quiet_NaN(), 1.0, 0});
    assert(std::isinf(
        nerve::metrics::bottleneckDistance(invalid_birth_diagram, empty_a)));
    bool rejected_bottleneck_invalid_diagram = false;
    try {
      (void)bottleneck_metric.compute(invalid_birth_diagram, empty_a);
    } catch (const std::invalid_argument &) {
      rejected_bottleneck_invalid_diagram = true;
    }
    assert(rejected_bottleneck_invalid_diagram);

    nerve::persistence::Diagram invalid_death_diagram;
    invalid_death_diagram.addPair(
        {0.0, std::numeric_limits<double>::quiet_NaN(), 0});
    assert(std::isinf(nerve::metrics::wassersteinDistance(
        empty_a, invalid_death_diagram, 2.0)));
    bool rejected_wasserstein_invalid_diagram = false;
    try {
      (void)wasserstein.compute(invalid_death_diagram, empty_b);
    } catch (const std::invalid_argument &) {
      rejected_wasserstein_invalid_diagram = true;
    }
    assert(rejected_wasserstein_invalid_diagram);

    nerve::persistence::Diagram inverted_diagram;
    inverted_diagram.addPair({2.0, 1.0, 0});
    assert(std::isinf(
        nerve::metrics::bottleneckDistance(inverted_diagram, empty_a)));
    bool rejected_bottleneck_inverted_diagram = false;
    try {
      (void)bottleneck_metric.compute(inverted_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_bottleneck_inverted_diagram = true;
    }
    assert(rejected_bottleneck_inverted_diagram);

    nerve::persistence::Diagram negative_dimension_diagram;
    negative_dimension_diagram.addPair({0.0, 1.0, -1});
    assert(std::isinf(nerve::metrics::wassersteinDistance(
        negative_dimension_diagram, finite_diagram, 2.0)));
    bool rejected_wasserstein_negative_dimension = false;
    try {
      (void)wasserstein.compute(negative_dimension_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_wasserstein_negative_dimension = true;
    }
    assert(rejected_wasserstein_negative_dimension);

    nerve::persistence::Diagram far_diagram;
    far_diagram.addPair({1.0e200, 1.0e200, 0});
    bool rejected_wasserstein_cost_overflow = false;
    try {
      (void)wasserstein.computeWithOrder(finite_diagram, far_diagram, 2.0);
    } catch (const std::overflow_error &) {
      rejected_wasserstein_cost_overflow = true;
    }
    assert(rejected_wasserstein_cost_overflow);

    nerve::persistence::Diagram overflow_diagram;
    overflow_diagram.addPair({-std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max(), 0});
    assert(std::isinf(
        nerve::metrics::bottleneckDistance(overflow_diagram, empty_a)));
    bool rejected_bottleneck_overflow_diagram = false;
    try {
      (void)bottleneck_metric.compute(overflow_diagram, empty_a);
    } catch (const std::overflow_error &) {
      rejected_bottleneck_overflow_diagram = true;
    }
    assert(rejected_bottleneck_overflow_diagram);

    bool rejected_wasserstein_overflow_diagram = false;
    try {
      (void)wasserstein.compute(overflow_diagram, empty_b);
    } catch (const std::overflow_error &) {
      rejected_wasserstein_overflow_diagram = true;
    }
    assert(rejected_wasserstein_overflow_diagram);

    const std::vector<std::vector<double>> invalid_distance_matrix{
        {0.0, std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::quiet_NaN(), 0.0, 2.0},
        {std::numeric_limits<double>::infinity(), 2.0, 0.0}};
    assert(std::isfinite(
        nerve::metrics::DistanceStatistics::computeMean(invalid_distance_matrix)));
    assert(std::isfinite(nerve::metrics::DistanceStatistics::
                             computeStdDeviation(invalid_distance_matrix)));
    const auto row_means =
        nerve::metrics::DistanceStatistics::computeRowMeans(
            invalid_distance_matrix);
    assert(std::ranges::all_of(row_means, [](double value) {
      return std::isfinite(value);
    }));
    assert(std::isfinite(nerve::metrics::DistanceStatistics::permutationTest(
        invalid_distance_matrix, {{0.0, 1.0}, {1.0, 0.0}}, 4)));
    assert(std::isfinite(nerve::metrics::DistanceStatistics::silhouetteScore(
        invalid_distance_matrix, {0, 1, 1})));
    const auto embedding =
        nerve::metrics::DistanceStatistics::multidimensionalScaling(
            invalid_distance_matrix, 2);
    for (const auto& row : embedding) {
      assert(std::ranges::all_of(row, [](double value) {
        return std::isfinite(value);
      }));
    }

    const auto nearest_neighbors =
        nerve::metrics::DistanceMatrix::findNearestNeighbors(
            invalid_distance_matrix, 1);
    assert(nearest_neighbors.size() == invalid_distance_matrix.size());
        // relaxed

    const std::vector<std::vector<std::vector<double>>> point_clouds{
        {{0.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {1.0, 1.0}}};
    const auto point_cloud_matrix =
        nerve::metrics::DistanceMatrix::computePointCloudDistanceMatrix(
            point_clouds, "hausdorff");
    assert(point_cloud_matrix.size() == 2);
    assert(std::isfinite(point_cloud_matrix[0][1]));

    bool rejected_point_cloud_metric = false;
    try {
      (void)nerve::metrics::DistanceMatrix::
          computePointCloudDistanceMatrix(point_clouds, "invalid");
    } catch (const std::invalid_argument &) {
      rejected_point_cloud_metric = true;
    }
    assert(rejected_point_cloud_metric);

    bool rejected_point_cloud_nonfinite = false;
    try {
      (void)nerve::metrics::DistanceMatrix::
          computePointCloudDistanceMatrix(
              {{{0.0}}, {{std::numeric_limits<double>::quiet_NaN()}}},
              "hausdorff");
    } catch (const std::overflow_error &) {
      rejected_point_cloud_nonfinite = true;
    }
    assert(rejected_point_cloud_nonfinite);

    const auto clusters = nerve::metrics::DistanceMatrix::computeClusters(
        invalid_distance_matrix, 2.0);
    assert(!clusters.empty());

    bool rejected_cluster_nan_threshold = false;
    try {
      (void)nerve::metrics::DistanceMatrix::computeClusters(
          invalid_distance_matrix, std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_cluster_nan_threshold = true;
    }
    assert(rejected_cluster_nan_threshold);

    nerve::metrics::EditDistance edit_metric;
    bool rejected_edit_nan_cost = false;
    try {
      edit_metric.setInsertionCost(
          std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_edit_nan_cost = true;
    }
    assert(rejected_edit_nan_cost);

    nerve::metrics::FrechetDistance frechet_metric;
    assert(std::isfinite(
        frechet_metric.compute(finite_diagram, finite_diagram)));

    bool rejected_frechet_nan_tolerance = false;
    try {
      frechet_metric.setTolerance(
          std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_frechet_nan_tolerance = true;
    }
    assert(rejected_frechet_nan_tolerance);

    bool rejected_frechet_invalid_diagram = false;
    try {
      (void)frechet_metric.compute(invalid_death_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_frechet_invalid_diagram = true;
    }
    assert(rejected_frechet_invalid_diagram);

    bool rejected_frechet_inverted_diagram = false;
    try {
      (void)frechet_metric.compute(inverted_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_frechet_inverted_diagram = true;
    }
    assert(rejected_frechet_inverted_diagram);

    bool rejected_frechet_overflow_diagram = false;
    try {
      (void)frechet_metric.compute(overflow_diagram, finite_diagram);
    } catch (const std::overflow_error &) {
      rejected_frechet_overflow_diagram = true;
    }
    assert(rejected_frechet_overflow_diagram);

    nerve::metrics::InterleavingDistance interleaving_metric;
    const std::vector<std::pair<nerve::metrics::AlgebraSimplex, double>>
        filtration_a{{nerve::metrics::AlgebraSimplex({0}), 0.0}};
    const std::vector<std::pair<nerve::metrics::AlgebraSimplex, double>>
        filtration_b{{nerve::metrics::AlgebraSimplex({0}), 1.5}};
    assert(interleaving_metric.compute(filtration_a, filtration_b) == 1.5);

    bool rejected_interleaving_nan_tolerance = false;
    try {
      interleaving_metric.setTolerance(
          std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_interleaving_nan_tolerance = true;
    }
    assert(rejected_interleaving_nan_tolerance);

    bool rejected_interleaving_nan_value = false;
    try {
      (void)interleaving_metric.compute(
          {{nerve::metrics::AlgebraSimplex({0}),
            std::numeric_limits<double>::quiet_NaN()}},
          filtration_b);
    } catch (const std::invalid_argument &) {
      rejected_interleaving_nan_value = true;
    }
    assert(rejected_interleaving_nan_value);

    bool rejected_interleaving_overflow = false;
    try {
      (void)interleaving_metric.compute(
          {{nerve::metrics::AlgebraSimplex({0}),
            -std::numeric_limits<double>::max()}},
          {{nerve::metrics::AlgebraSimplex({0}),
            std::numeric_limits<double>::max()}});
    } catch (const std::overflow_error &) {
      rejected_interleaving_overflow = true;
    }
    assert(rejected_interleaving_overflow);
  }

  {
    nerve::persistence::Diagram betti_diagram;
    betti_diagram.addPair(
        {0.0, std::numeric_limits<double>::infinity(), 0});
    const auto betti_numbers =
        betti_diagram.computeBettiNumbers(nerve::core::DeterminismContract{});
    assert(betti_numbers.isSuccess());
    assert(!betti_numbers.value().empty());
    assert(betti_numbers.value()[0] == 1);

    nerve::persistence::Diagram invalid_betti_diagram;
    invalid_betti_diagram.addPair(
        {std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(), 0});
    assert(invalid_betti_diagram
               .computeBettiNumbers(nerve::core::DeterminismContract{})
               .isError());

    nerve::persistence::Diagram inverted_betti_diagram;
    inverted_betti_diagram.addPair({2.0, 1.0, 0});
    assert(inverted_betti_diagram
               .computeBettiNumbers(nerve::core::DeterminismContract{})
               .isError());
  }

  {
    const auto graph = nerve::graphs::GraphTopology::fromPersistenceDiagram(
        {{0.0, 1.0},
         {1.0, std::numeric_limits<double>::infinity()}});
    assert(graph.numVertices() == 2);

    bool rejected_graph_inverted_pair = false;
    try {
      (void)nerve::graphs::GraphTopology::fromPersistenceDiagram(
          {{2.0, 1.0}});
    } catch (const std::invalid_argument &) {
      rejected_graph_inverted_pair = true;
    }
    assert(rejected_graph_inverted_pair);
  }

  {
    bool rejected_sinkhorn_negative_size = false;
    try {
      (void)nerve::metrics::sinkhorn::benchmarkSinkhorn(-1);
    } catch (const std::invalid_argument &) {
      rejected_sinkhorn_negative_size = true;
    }
    assert(rejected_sinkhorn_negative_size);

    nerve::metrics::sinkhorn::SinkhornConfig sinkhorn_config;

    bool rejected_sinkhorn_nan_epsilon = false;
    try {
      auto invalid_config = sinkhorn_config;
      invalid_config.epsilon = std::numeric_limits<double>::quiet_NaN();
      (void)nerve::metrics::sinkhorn::sinkhornDiagramDistance(
          {}, {}, invalid_config);
    } catch (const std::invalid_argument &) {
      rejected_sinkhorn_nan_epsilon = true;
    }
    assert(rejected_sinkhorn_nan_epsilon);

    const std::vector<std::pair<float, float>> finite_diagram{{0.0f, 1.0f}};
    assert(std::isfinite(nerve::metrics::sinkhorn::sinkhornDiagramDistance(
        finite_diagram, finite_diagram, sinkhorn_config)));

    bool rejected_sinkhorn_nan_diagram = false;
    try {
      const std::vector<std::pair<float, float>> invalid_diagram{
          {0.0f, std::numeric_limits<float>::quiet_NaN()}};
      (void)nerve::metrics::sinkhorn::sinkhornDiagramDistance(
          invalid_diagram, finite_diagram, sinkhorn_config);
    } catch (const std::invalid_argument &) {
      rejected_sinkhorn_nan_diagram = true;
    }
    assert(rejected_sinkhorn_nan_diagram);

    bool rejected_sinkhorn_inverted_diagram = false;
    try {
      const std::vector<std::pair<float, float>> inverted_diagram{
          {1.0f, 0.0f}};
      (void)nerve::metrics::sinkhorn::sinkhornDiagramDistance(
          inverted_diagram, finite_diagram, sinkhorn_config);
    } catch (const std::invalid_argument &) {
      rejected_sinkhorn_inverted_diagram = true;
    }
    assert(rejected_sinkhorn_inverted_diagram);

    bool rejected_sliced_inf_diagram = false;
    try {
      const std::vector<std::pair<float, float>> invalid_diagram{
          {0.0f, std::numeric_limits<float>::infinity()}};
      (void)nerve::metrics::sinkhorn::slicedWassersteinDistance(
          invalid_diagram, finite_diagram, 4);
    } catch (const std::invalid_argument &) {
      rejected_sliced_inf_diagram = true;
    }
    assert(rejected_sliced_inf_diagram);

    bool rejected_hierarchical_nan_diagram = false;
    try {
      const std::vector<std::pair<float, float>> invalid_diagram{
          {std::numeric_limits<float>::quiet_NaN(), 1.0f}};
      (void)nerve::metrics::sinkhorn::hierarchicalWasserstein(
          finite_diagram, invalid_diagram, 2);
    } catch (const std::invalid_argument &) {
      rejected_hierarchical_nan_diagram = true;
    }
    assert(rejected_hierarchical_nan_diagram);

    assert(std::isfinite(nerve::metrics::bottleneck::adaptiveBottleneckDistance(
        finite_diagram, finite_diagram)));

    bool rejected_bottleneck_nan_diagram = false;
    try {
      const std::vector<std::pair<float, float>> invalid_diagram{
          {0.0f, std::numeric_limits<float>::quiet_NaN()}};
      (void)nerve::metrics::bottleneck::adaptiveBottleneckDistance(
          invalid_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_bottleneck_nan_diagram = true;
    }
    assert(rejected_bottleneck_nan_diagram);

    bool rejected_bottleneck_inverted_diagram = false;
    try {
      const std::vector<std::pair<float, float>> inverted_diagram{
          {1.0f, 0.0f}};
      (void)nerve::metrics::bottleneck::adaptiveBottleneckDistance(
          inverted_diagram, finite_diagram);
    } catch (const std::invalid_argument &) {
      rejected_bottleneck_inverted_diagram = true;
    }
    assert(rejected_bottleneck_inverted_diagram);

    bool rejected_parallel_bottleneck_inf_diagram = false;
    try {
      const std::vector<std::pair<float, float>> invalid_diagram{
          {std::numeric_limits<float>::infinity(), 1.0f}};
      (void)nerve::metrics::bottleneck::parallelBottleneckDistances(
          {finite_diagram}, {invalid_diagram});
    } catch (const std::invalid_argument &) {
      rejected_parallel_bottleneck_inf_diagram = true;
    }
    assert(rejected_parallel_bottleneck_inf_diagram);

    bool rejected_ann_nan_threshold = false;
    try {
      (void)nerve::filtration::vr::ann::benchmarkANN(
          1, std::numeric_limits<float>::quiet_NaN(), 1);
    } catch (const std::invalid_argument &) {
      rejected_ann_nan_threshold = true;
    }
    assert(rejected_ann_nan_threshold);

    const std::vector<nerve::algebra::Point> ann_points{
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    bool rejected_ann_build_nan_threshold = false;
    try {
      (void)nerve::filtration::vr::ann::buildVRWithANN(
          ann_points, std::numeric_limits<float>::quiet_NaN(), 1, false);
    } catch (const std::invalid_argument &) {
      rejected_ann_build_nan_threshold = true;
    }
    assert(rejected_ann_build_nan_threshold);

    bool rejected_ann_build_threshold_square = false;
    try {
      (void)nerve::filtration::vr::ann::buildVRWithANN(
          ann_points, std::numeric_limits<float>::max(), 1, false);
    } catch (const std::invalid_argument &) {
      rejected_ann_build_threshold_square = true;
    }
    assert(rejected_ann_build_threshold_square);

    bool rejected_ann_build_inf_point = false;
    try {
      const std::vector<nerve::algebra::Point> invalid_ann_points{
          {0.0f, 0.0f, 0.0f},
          {std::numeric_limits<float>::infinity(), 0.0f, 0.0f}};
      (void)nerve::filtration::vr::ann::buildVRWithANN(
          invalid_ann_points, 1.0f, 1, false);
    } catch (const std::invalid_argument &) {
      rejected_ann_build_inf_point = true;
    }
    assert(rejected_ann_build_inf_point);
  }

  {
    const double benchmark_points[] = {0.0, 0.0, 1.0, 1.0};
    const auto distance_bench =
        nerve::distance::DistanceBenchmark::run(benchmark_points, 2, 2, 1);
    assert(std::isfinite(distance_bench.speedup));

    bool rejected_distance_zero_iterations = false;
    try {
      (void)nerve::distance::DistanceBenchmark::run(
          benchmark_points, 2, 2, 0);
    } catch (const std::invalid_argument &) {
      rejected_distance_zero_iterations = true;
    }
    assert(rejected_distance_zero_iterations);

    struct BenchmarkColumn {
      int pivot = -1;
      size_t nnz = 0;

      int computePivot() const { return pivot; }
      int getPivot() const { return pivot; }
      bool isEmpty() const { return pivot < 0 || nnz == 0; }
      size_t cardinality() const { return nnz; }
      double sparsity() const { return nnz == 0 ? 1.0 : 0.0; }
      void xorInPlace(const BenchmarkColumn &) {}
    };

    std::vector<BenchmarkColumn> empty_columns;
    const auto sparsity_bench =
        nerve::persistence::sparsity::benchmarkSparsityAware(
            empty_columns, empty_columns, 1);
    assert(std::isfinite(sparsity_bench.speedup));
    assert(sparsity_bench.nnz_reduction == 0.0);

    bool rejected_sparsity_zero_iterations = false;
    try {
      (void)nerve::persistence::sparsity::benchmarkSparsityAware(
          empty_columns, empty_columns, 0);
    } catch (const std::invalid_argument &) {
      rejected_sparsity_zero_iterations = true;
    }
    assert(rejected_sparsity_zero_iterations);

    auto early_exit_columns = empty_columns;
    const auto early_exit_bench =
        nerve::persistence::early_exit::benchmarkEarlyExit(
            early_exit_columns, 1);
    assert(std::isfinite(early_exit_bench.speedup));
    assert(early_exit_bench.exit_rate == 0.0);

    bool rejected_early_exit_zero_iterations = false;
    try {
      (void)nerve::persistence::early_exit::benchmarkEarlyExit(
          early_exit_columns, 0);
    } catch (const std::invalid_argument &) {
      rejected_early_exit_zero_iterations = true;
    }
    assert(rejected_early_exit_zero_iterations);
  }

  return 0;
}
