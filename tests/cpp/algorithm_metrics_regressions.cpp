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

#include "nerve/algebra/simplex.hpp"
#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/accelerated/accelerated_error_tools.hpp"

int main() {
#include "nerve/persistence/adaptive_acceleration/matrix_multiplication_framework.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/streaming_processor.hpp"
#include "nerve/persistence/approximate/approximate_nearest_neighbor.hpp"
#include "nerve/persistence/approximate/bloom_filter.hpp"
#include "nerve/persistence/approximate/distilled_vr_filtration.hpp"
#include "nerve/persistence/approximate/perfect_hash.hpp"
#include "nerve/persistence/approximate/robin_hood_hash.hpp"
#include "nerve/persistence/approximate/sketching_approximation.hpp"
#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/core/high_dimensional_exact.hpp"
#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/cuda/cuda_multi_gpu.hpp"
#include "nerve/persistence/cuda/cuda_memory_manager.hpp"
#include "nerve/persistence/distilled_vr_filtration.hpp"
#include "nerve/persistence/distributed/mpi_distributed_ph.hpp"
#include "nerve/persistence/kernels/dimension_specialized_kernels.hpp"
#include "nerve/persistence/kernels/kernel_dimension_specialized_ops.hpp"
#include "nerve/persistence/kernels/kernel_h1_ops.hpp"
#include "nerve/persistence/kernels/kernel_h2_alpha_ops.hpp"
#include "nerve/persistence/kernels/kernel_h3_tetrahedra_ops.hpp"
#include "nerve/persistence/kernels/kernel_h4_chunked_ops.hpp"
#include "nerve/persistence/kernels/kernel_h5_prefetch_ops.hpp"
#include "nerve/persistence/kernels/kernel_h6_streaming_ops.hpp"
#include "nerve/persistence/kernels/ph5_high_dim_ops.hpp"
#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "nerve/persistence/memory/memory_pool.hpp"
#include "nerve/persistence/memory/vram_algorithms.hpp"
#include "nerve/persistence/reduction/reduction_clearing_ops.hpp"
#include "nerve/persistence/reduction/reduction_edge_collapse_ops.hpp"
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "nerve/persistence/reduction/reduction_sparsity_aware.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"
#include "nerve/persistence/streaming/tile_streaming_ph.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"
#include "nerve/persistence/utils/early_exit_optimizer.hpp"
#include "nerve/persistence/vr/vr_algorithm_selector_ops.hpp"
#include "nerve/persistence/vr/vr_dispatch_ops.hpp"
#include "nerve/persistence/vr/vr_distance_tiled_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_landmark_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"
#include "nerve/persistence/vr/vr_sparse_rips_ops.hpp"
#include "nerve/serialization/ph5_ph6_schema_serializer.hpp"
#include "nerve/serialization/serialization_manager.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"
#include "nerve/streaming/streaming_tda.hpp"
#include "nerve/streaming/lock_free_streaming.hpp"

namespace nerve::persistence::accelerated::accelerated_error_tools {
nerve::errors::ErrorResult<void> validateMetrics(
    const nerve::common::AcceleratedPerformanceStats&) {
  return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT, "noop");
}
nerve::errors::ErrorResult<void> validatePairs(
    const std::vector<nerve::persistence::Pair>&) {
  return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT, "noop");
}
}
namespace nerve::persistence::accelerated::optimization_recommendations {
std::vector<std::string> suggestActions(
    const nerve::common::PerformanceMetrics& metrics);
}
namespace nerve::persistence::accelerated::performance_impact {
double computeRuntimeChange(const nerve::common::PerformanceMetrics& baseline,
                            const nerve::common::PerformanceMetrics& current);
double computeMemoryChange(const nerve::common::PerformanceMetrics& baseline,
                           const nerve::common::PerformanceMetrics& current);
double computeOverallImpactScore(
    const nerve::common::PerformanceMetrics& baseline,
    const nerve::common::PerformanceMetrics& current);
}

int main() {
  {
    using namespace nerve::algorithms;

    const std::array<long double, 4> long_double_points{
        0.0L,
        0.0L,
        3.0L,
        4.0L,
    };
    EuclideanMetric<long double> metric;
    assert(metric.compute(
               std::span<const long double>(long_double_points.data(), 2),
               std::span<const long double>(long_double_points.data() + 2,
                                            2)) == 5.0L);

    DistanceMatrixComputer<long double> computer;
    const auto distances = computer.compute(long_double_points, 2, 2);
    assert(distances.size() == 4);
    assert(distances[1] == 5.0L);
    assert(distances[2] == 5.0L);

    nerve::distance::DistanceComputer simd_distance;
    const std::vector<double> simd_points{0.0, 0.0, 3.0, 4.0};
    std::vector<double> simd_results;
    simd_distance.computeBatch(simd_points.data(), 2, 2, {{0, 1}},
                               simd_results);
    assert(simd_results.size() == 1);
    assert(simd_results[0] == 5.0);

    bool rejected_simd_bad_index = false;
    try {
      simd_distance.computeBatch(simd_points.data(), 2, 2, {{0, 2}},
                                 simd_results);
    } catch (const std::out_of_range &) {
      rejected_simd_bad_index = true;
    }
    assert(rejected_simd_bad_index);

    bool rejected_simd_nan_point = false;
    try {
      const std::vector<double> invalid_simd_points{
          0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
      simd_distance.computeBatch(invalid_simd_points.data(), 2, 2, {{0, 1}},
                                 simd_results);
    } catch (const std::invalid_argument &) {
      rejected_simd_nan_point = true;
    }
    assert(rejected_simd_nan_point);
  }

  {
    using nerve::instrumentation::CertificateFactory;
    using nerve::instrumentation::StabilityCertificate;

    const auto cert = CertificateFactory::createPh5Ph6Certificate(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), 128, 42, 100,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), 6, 1,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN());
    assert(cert.isValid());
    assert(std::isfinite(cert.bottleneck_upper_bound));
    assert(std::isfinite(cert.wasserstein_upper_bound));
    assert(std::isfinite(cert.numerical_residual));
    assert(std::isfinite(cert.highdim_condition_estimate));
    assert(std::isfinite(cert.effective_rank_estimate));
    assert(std::isfinite(cert.compression_ratio));
    assert(std::isfinite(cert.memory_efficiency_score));

    StabilityCertificate invalid = cert;
    invalid.bottleneck_upper_bound = std::numeric_limits<float>::infinity();
    assert(!invalid.isValid());
  }

  {
    using nerve::instrumentation::MetricEvent;
    using nerve::instrumentation::MetricKind;
    using nerve::instrumentation::TopologicalDelta;

    MetricEvent metric{"finite_metric", MetricKind::GAUGE, 1.0, 1, 0, 0, 0};
    assert(metric.isValid());
    metric.value = std::numeric_limits<double>::infinity();
    assert(!metric.isValid());
    metric.value = std::numeric_limits<double>::quiet_NaN();
    assert(!metric.isValid());

    TopologicalDelta delta{1, 1, 2.0f, 0.5f, 0.25f, 2, 200, 0, 1};
    assert(delta.isValid());
    assert(delta.isSignificant());
    assert(delta.getImpactScore() == 6.0f);

    delta.lifetime_delta = std::numeric_limits<float>::infinity();
    assert(!delta.isValid());
    assert(!delta.isSignificant());
    assert(delta.getImpactScore() == 0.0f);
  }

  {
    nerve::errors::ErrorContext error_context;
    error_context.operation_name = "serialize\"operation";
    error_context.component_name = "errors";
    error_context.durationMs = std::numeric_limits<double>::infinity();
    std::string json = error_context.toJson();
    assert(json.find("\"durationMs\":0.000") != std::string::npos);
    assert(json.find("inf") == std::string::npos);
    assert(json.find("nan") == std::string::npos);
    assert(json.find("serialize\\\"operation") != std::string::npos);

    error_context.durationMs = std::numeric_limits<double>::quiet_NaN();
    json = error_context.toJson();
    assert(json.find("\"durationMs\":0.000") != std::string::npos);
    assert(json.find("nan") == std::string::npos);
  }

  {
    nerve::streaming::StreamDataPoint point{{0.0f, 1.0f}, 1, 7, {2.0f}, 1.0f};
    assert(point.isValid());
    point.coordinates[1] = std::numeric_limits<float>::quiet_NaN();
    assert(!point.isValid());
    point.coordinates[1] = 1.0f;
    point.attributes[0] = std::numeric_limits<float>::infinity();
    assert(!point.isValid());
    point.attributes[0] = 2.0f;
    point.weight = std::numeric_limits<float>::infinity();
    assert(!point.isValid());

    nerve::streaming::StreamingPersistenceDiagram diagram{
        {{0.0f, 1.0f}, {0.5f, std::numeric_limits<float>::infinity()}},
        1,
        2,
        2,
        0.5,
        0.01,
        true};
    assert(diagram.isValid());
    diagram.points[0].second = std::numeric_limits<float>::quiet_NaN();
    assert(!diagram.isValid());
    diagram.points[0].second = 1.0f;
    diagram.computation_time_ms = std::numeric_limits<double>::infinity();
    assert(!diagram.isValid());
    diagram.computation_time_ms = 0.5;
    diagram.approximation_error = -0.01;
    assert(!diagram.isValid());
  }

  {
    nerve::interaction::HigherOrderInteraction interaction{
        {1, 2}, 2, 0.5F, 1, 2, 1.0F, {0.25F, 0.75F}, "pair"};
    assert(interaction.isValid());
    interaction.strength = std::numeric_limits<float>::infinity();
    assert(!interaction.isValid());
    interaction.strength = 0.5F;
    interaction.lifetime = std::numeric_limits<float>::infinity();
    assert(!interaction.isValid());
    interaction.lifetime = 1.0F;
    interaction.attributes[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!interaction.isValid());
  }

  {
    using nerve::persistence::adaptive_acceleration::representative::Cycle;
    using nerve::persistence::adaptive_acceleration::representative::CycleVisualizationData;

    Cycle cycle{{0, 1}, {1.0, -1.0}, 1, 0.0,
                std::numeric_limits<double>::infinity()};
    assert(cycle.isValid());
    cycle.coefficients[0] = std::numeric_limits<double>::quiet_NaN();
    assert(!cycle.isValid());
    cycle.coefficients[0] = 1.0;
    cycle.birth_time = std::numeric_limits<double>::infinity();
    assert(!cycle.isValid());
    cycle.birth_time = 2.0;
    cycle.death_time = 1.0;
    assert(!cycle.isValid());

    CycleVisualizationData visualization{{{0.0, 1.0}}, {{0, 1}}, {1.0}, 1, 2.0, "points"};
    assert(visualization.isValid());
    visualization.vertices[0][0] = std::numeric_limits<double>::quiet_NaN();
    assert(!visualization.isValid());
    visualization.vertices[0][0] = 0.0;
    visualization.edge_weights[0] = std::numeric_limits<double>::infinity();
    assert(!visualization.isValid());
  }

  {
    using nerve::persistence::adaptive_acceleration::streaming::DataChunk;

    DataChunk chunk{{0.0, 0.0, 1.0, 1.0}, 0, 2, 2, 1.0, false};
    assert(chunk.isValid());
    chunk.points[2] = std::numeric_limits<double>::quiet_NaN();
    assert(!chunk.isValid());
    chunk.points[2] = 1.0;
    chunk.max_radius = std::numeric_limits<double>::infinity();
    assert(!chunk.isValid());
    chunk.max_radius = 1.0;
    chunk.num_points = 3;
    assert(!chunk.isValid());
  }

  {
    using MatrixCycle = nerve::persistence::adaptive_acceleration::Cycle;

    MatrixCycle cycle{{0, 1}, {1.0, -1.0}, 1, 0.0,
                      std::numeric_limits<double>::infinity()};
    assert(cycle.isValid());
    cycle.coefficients.pop_back();
    assert(!cycle.isValid());
    cycle.coefficients.push_back(-1.0);
    cycle.coefficients[0] = std::numeric_limits<double>::infinity();
    assert(!cycle.isValid());
    cycle.coefficients[0] = 1.0;
    cycle.birth_time = 2.0;
    cycle.death_time = 1.0;
    assert(!cycle.isValid());
  }

  {
    nerve::approximation::DiagramPoint point{0.0F, 1.0F, 1.0F, 0};
    assert(point.isValid());
    point.birth = std::numeric_limits<float>::quiet_NaN();
    assert(!point.isValid());
    point.birth = 0.0F;
    point.death = std::numeric_limits<float>::infinity();
    assert(!point.isValid());
    point.death = 1.0F;
    point.persistence = std::numeric_limits<float>::infinity();
    assert(!point.isValid());
  }

  {
    nerve::spectral::Eigenpair eigenpair;
    eigenpair.eigenvalue = 0.5;
    eigenpair.eigenvector = {1.0, 0.0};
    eigenpair.error_estimate = 0.0;
    eigenpair.spectral_gap = 0.25;
    eigenpair.participation_ratio = 1.0;
    assert(eigenpair.isValid());

    eigenpair.eigenvector[0] = std::numeric_limits<double>::quiet_NaN();
    assert(!eigenpair.isValid());
    eigenpair.eigenvector[0] = 1.0;
    eigenpair.eigenvalue = std::numeric_limits<double>::infinity();
    assert(!eigenpair.isValid());
    eigenpair.eigenvalue = 0.5;
    eigenpair.complex_eigenvalue =
        std::complex<double>(0.5, std::numeric_limits<double>::quiet_NaN());
    assert(!eigenpair.isValid());
  }

  {
    using namespace nerve::persistence::roaring;

    const auto roaring_config = getOptimalRoaringConfig(2048, 0.99);
    assert(roaring_config.sparsity_threshold == 0.85);
    assert(shouldUseRoaring(0.95));

    bool rejected_roaring_nan_sparsity = false;
    try {
      (void)getOptimalRoaringConfig(
          2048, std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
      rejected_roaring_nan_sparsity = true;
    }
    assert(rejected_roaring_nan_sparsity);

    bool rejected_roaring_infinite_sparsity = false;
    try {
      (void)shouldUseRoaring(std::numeric_limits<double>::infinity());
    } catch (const std::invalid_argument &) {
      rejected_roaring_infinite_sparsity = true;
    }
    assert(rejected_roaring_infinite_sparsity);
  }

  {
    using namespace nerve::algorithms;

    const std::vector<double> points{
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    };

    PCAFilter<double> pca(2);
    const auto pca_values = pca.apply(points, 3, 2);
    assert(pca_values.size() == 6);

    DensityFilter<double> density(1);
    const auto density_values = density.apply(points, 3, 2);
    assert(density_values.size() == 3);

    Cover<double>::Config cover_config;
    cover_config.resolution = 2;
    cover_config.overlap = 0.25;
    Cover<double> cover(cover_config);
    const auto cover_sets = cover.build(pca_values, 3, 2);
    assert(!cover_sets.empty());

    bool rejected_nan_cover_filter = false;
    try {
      const std::vector<double> bad_filter{
          0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
      (void)cover.build(bad_filter, 3, 1);
    } catch (const std::invalid_argument &) {
      rejected_nan_cover_filter = true;
    }
    assert(rejected_nan_cover_filter);

    bool rejected_inf_cover_filter = false;
    try {
      const std::vector<double> bad_filter{
          0.0, 0.0, 1.0, std::numeric_limits<double>::infinity(), 2.0, 2.0};
      (void)cover.build(bad_filter, 3, 2);
    } catch (const std::invalid_argument &) {
      rejected_inf_cover_filter = true;
    }
    assert(rejected_inf_cover_filter);

    bool rejected_inf_cover_overlap = false;
    try {
      Cover<double>::Config bad_cover_config;
      bad_cover_config.overlap = std::numeric_limits<double>::infinity();
      Cover<double> bad_cover(bad_cover_config);
      const std::vector<double> filter{0.0, 1.0};
      (void)bad_cover.build(filter, 2, 1);
    } catch (const std::invalid_argument &) {
      rejected_inf_cover_overlap = true;
    }
    assert(rejected_inf_cover_overlap);

    bool rejected_mapper_nan_points = false;
    try {
      const std::vector<double> bad_points{
          0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
      EccentricityFilter<double> eccentricity;
      (void)eccentricity.apply(bad_points, 2, 2);
    } catch (const std::invalid_argument &) {
      rejected_mapper_nan_points = true;
    }
    assert(rejected_mapper_nan_points);

    bool rejected_mapper_compute_nan_points = false;
    try {
      MapperAlgorithm<double>::Config mapper_config;
      MapperAlgorithm<double> mapper(mapper_config);
      const std::vector<double> bad_points{
          0.0, 0.0, 1.0, std::numeric_limits<double>::quiet_NaN()};
      (void)mapper.compute(bad_points, 2, 2);
    } catch (const std::invalid_argument &) {
      rejected_mapper_compute_nan_points = true;
    }
    assert(rejected_mapper_compute_nan_points);

    const double dummy = 0.0;
    const std::span<const double> tiny_span(&dummy, 1);
    assert(pca.apply(tiny_span, std::numeric_limits<std::size_t>::max(), 2)
               .empty());

    DBSCANClustering<double>::Config dbscan_config;
    DBSCANClustering<double> dbscan(dbscan_config);
    assert(dbscan.cluster(tiny_span, std::numeric_limits<std::size_t>::max(), 1)
               .empty());

    bool rejected_dbscan_nan_points = false;
    try {
      const std::vector<double> bad_points{
          0.0, 0.0, 1.0, std::numeric_limits<double>::quiet_NaN()};
      (void)dbscan.cluster(bad_points, 2, 2);
    } catch (const std::invalid_argument &) {
      rejected_dbscan_nan_points = true;
    }
    assert(rejected_dbscan_nan_points);

    bool rejected_single_linkage_inf_config = false;
    try {
      SingleLinkageClustering<double>::Config linkage_config;
      linkage_config.linkage_distance = std::numeric_limits<double>::infinity();
      SingleLinkageClustering<double> linkage(linkage_config);
      (void)linkage.cluster(points, 3, 2);
    } catch (const std::invalid_argument &) {
      rejected_single_linkage_inf_config = true;
    }
    assert(rejected_single_linkage_inf_config);

    ConnectedComponentsClustering<double> connected;
    assert(
        connected
            .cluster(tiny_span,
                     static_cast<std::size_t>(std::numeric_limits<int>::max()) +
                         1,
                     1)
            .empty());

    MapperGraph<double> graph;
    graph.nodes = {
        MapperNode<double>{0, {0}, {0.0}},
        MapperNode<double>{1, {1}, {1.0}},
        MapperNode<double>{2, {2}, {2.0}},
    };
    graph.edges = {MapperEdge<double>{0, 1, 1.0, 1}};
    const auto components = connected_components(graph);
    assert(components.size() == 2);
    assert(graph_diameter(graph) == 1);
  }


  return 0;
}
