#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_problem_analysis.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/adaptive_acceleration/lockfree_multicore.hpp"
#include "nerve/persistence/adaptive_acceleration/matrix_multiplication_framework.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/adaptive_acceleration/sparsification_engine.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/approximate_processor.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/streaming_processor.hpp"

#include <cassert>
#include <limits>
#include <vector>

int main()
{
    using namespace nerve::persistence::adaptive_acceleration;
    namespace approx = nerve::persistence::adaptive_acceleration::approximation;
    namespace stream = nerve::persistence::adaptive_acceleration::streaming;

    auto matrix_result = SparseMatrix::fromDenseMatrix({1.0}, 1, 1);
    assert(matrix_result.isSuccess());
    const SparseMatrix matrix = matrix_result.value();

    auto engine_result = MatrixMultiplicationEngine::create(MatrixMultiplicationConfig{});
    assert(engine_result.isSuccess());
    auto &engine = engine_result.value();

    ProblemCharacteristics problem;
    auto valid = engine->compute(matrix, problem);
    assert(valid.isSuccess());
    assert(valid.value().size() == 1);
    assert(valid.value()[0].death == 1.0);

    problem.estimated_columns = std::numeric_limits<std::size_t>::max() / 4;
    auto oversized_estimate = engine->compute(matrix, problem);
    assert(oversized_estimate.isError());
    assert(oversized_estimate.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);

    auto product = engine->fastMatrixMultiply(matrix, matrix);
    assert(product.isSuccess());
    assert(product.value().isValid());
    assert(product.value()(0, 0) == 1.0);

    auto huge_value = SparseMatrix::fromDenseMatrix({std::numeric_limits<double>::max()}, 1, 1);
    assert(huge_value.isSuccess());
    auto rejected_overflow = engine->fastMatrixMultiply(huge_value.value(), huge_value.value());
    assert(rejected_overflow.isError());
    assert(rejected_overflow.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    auto sparsifier_result = SparsificationEngine::create(SparsificationConfig{});
    assert(sparsifier_result.isSuccess());
    auto &sparsifier = sparsifier_result.value();

    auto rectangular_result = SparseMatrix::fromDenseMatrix({1.0, 2.0}, 1, 2);
    assert(rectangular_result.isSuccess());
    auto rectangular = sparsifier->sparsify(rectangular_result.value(),
                                            SparsificationStrategy::SWAP_REDUCTION, 1.0);
    assert(rectangular.isSuccess());
    assert(rectangular.value().numRows() == 1);
    assert(rectangular.value().numCols() == 2);
    assert(rectangular.value()(0, 0) == 1.0);
    assert(rectangular.value()(0, 1) == 2.0);

    auto invalid_target = sparsifier->sparsify(matrix, SparsificationStrategy::SWAP_REDUCTION,
                                               std::numeric_limits<double>::quiet_NaN());
    assert(invalid_target.isError());
    assert(invalid_target.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    SparsificationConfig invalid_config;
    invalid_config.max_sparsity_loss = std::numeric_limits<double>::quiet_NaN();
    auto invalid_sparsifier = SparsificationEngine::create(invalid_config);
    assert(invalid_sparsifier.isError());
    assert(invalid_sparsifier.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    auto huge_shape = SparseMatrix::fromBoundaryMatrix({}, 46341, 46341);
    assert(huge_shape.isSuccess());
    auto rejected_dense =
        sparsifier->sparsify(huge_shape.value(), SparsificationStrategy::SWAP_REDUCTION, 0.5);
    assert(rejected_dense.isError());
    assert(rejected_dense.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);

    nerve::common::VRConfig vr_config;
    vr_config.max_dim = 1;
    auto vr_engine_result = AdaptiveAccelerationVrEngine::create(vr_config);
    assert(vr_engine_result.isSuccess());
    auto &vr_engine = vr_engine_result.value();

    auto invalid_points = vr_engine->computeVrPersistence(
        {0.0, std::numeric_limits<double>::quiet_NaN()}, 1, vr_config);
    assert(invalid_points.isError());
    assert(invalid_points.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    nerve::common::VRConfig invalid_vr_config = vr_config;
    invalid_vr_config.max_dim = 0;
    auto invalid_vr_run = vr_engine->computeVrPersistence({0.0, 1.0}, 1, invalid_vr_config);
    assert(invalid_vr_run.isError());
    assert(invalid_vr_run.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    std::vector<double> extreme_points = {-std::numeric_limits<double>::max(),
                                          std::numeric_limits<double>::max()};
    nerve::core::BufferView<const double> extreme_view(extreme_points.data(),
                                                       extreme_points.size());
    auto characteristics = ProblemAnalyzer::analyzeProblem(extreme_view, 1);
    assert(std::isfinite(characteristics.density));
    assert(std::isfinite(characteristics.estimated_complexity));
    assert(std::isfinite(characteristics.memory_requirement_mb));

    auto selector_result = AdaptiveAlgorithmSelector::create(AdaptiveConfig{});
    assert(selector_result.isSuccess());
    auto &selector = selector_result.value();
    std::vector<double> invalid_adaptive_points = {0.0, std::numeric_limits<double>::quiet_NaN()};
    nerve::core::BufferView<const double> invalid_adaptive_view(invalid_adaptive_points.data(),
                                                                invalid_adaptive_points.size());
    auto invalid_adaptive = selector->executeAdaptive(invalid_adaptive_view, 1, vr_config);
    assert(invalid_adaptive.isError());
    assert(invalid_adaptive.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    LockfreeMatrixColumn oversized_column(std::numeric_limits<std::size_t>::max());
    assert(oversized_column.empty());
    assert(!oversized_column.push_back(-1));
    assert(oversized_column.push_back(0));

    auto reducer_result = LockfreeReducer::create(LockfreeConfig{});
    assert(reducer_result.isSuccess());
    auto &reducer = reducer_result.value();
    auto valid_reduction = reducer->reduceParallel({oversized_column}, 1);
    assert(valid_reduction.isSuccess());

    LockfreeMatrixColumn invalid_column;
    invalid_column.assignSortedUnique({-1});
    auto invalid_reduction = reducer->reduceParallel({invalid_column}, 1);
    assert(invalid_reduction.isError());
    assert(invalid_reduction.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    approx::ApproximationConfig invalid_approx_config;
    invalid_approx_config.max_error = std::numeric_limits<double>::quiet_NaN();
    auto invalid_approx_processor = approx::ApproximateProcessor::create(invalid_approx_config);
    assert(invalid_approx_processor.isError());
    assert(invalid_approx_processor.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    auto approx_processor = approx::ApproximateProcessor::create(approx::ApproximationConfig{});
    assert(approx_processor.isSuccess());
    std::vector<double> invalid_approx_points = {0.0, std::numeric_limits<double>::quiet_NaN()};
    nerve::core::BufferView<const double> invalid_approx_view(invalid_approx_points.data(),
                                                              invalid_approx_points.size());
    auto invalid_approx_result = approx_processor.value()->computeApproximate(
        invalid_approx_view, 1, approx::ApproximationLevel::LOW_PRECISION, vr_config);
    assert(invalid_approx_result.isError());
    assert(invalid_approx_result.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    std::vector<nerve::persistence::Pair> infinite_pairs = {
        nerve::persistence::Pair{0.0, std::numeric_limits<double>::infinity(), 0}};
    auto bounds = approx::ErrorBoundsCalculator::calculateErrorBounds(
        infinite_pairs, {}, approx::ApproximationLevel::VERY_FAST);
    assert(bounds.isSuccess());
    assert(std::isfinite(bounds.value().bottleneck_distance));
    assert(std::isfinite(bounds.value().wasserstein_distance));
    auto estimated_bounds = approx::ErrorBoundsCalculator::estimateErrorBounds(
        infinite_pairs, approx::ApproximationLevel::VERY_FAST, characteristics);
    assert(estimated_bounds.isSuccess());
    assert(std::isfinite(estimated_bounds.value().wasserstein_distance));

    std::vector<nerve::persistence::Pair> invalid_bound_pairs = {nerve::persistence::Pair{
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0}};
    auto invalid_bounds = approx::ErrorBoundsCalculator::calculateErrorBounds(
        invalid_bound_pairs, infinite_pairs, approx::ApproximationLevel::VERY_FAST);
    assert(invalid_bounds.isError());
    assert(invalid_bounds.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);
    auto invalid_estimated_bounds = approx::ErrorBoundsCalculator::estimateErrorBounds(
        invalid_bound_pairs, approx::ApproximationLevel::VERY_FAST, characteristics);
    assert(invalid_estimated_bounds.isError());
    assert(invalid_estimated_bounds.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    stream::StreamingConfig invalid_stream_config;
    invalid_stream_config.target_efficiency = std::numeric_limits<double>::quiet_NaN();
    auto invalid_streaming = stream::StreamingProcessor::create(invalid_stream_config);
    assert(invalid_streaming.isError());
    assert(invalid_streaming.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    auto oversized_pool = stream::MemoryPool::create(std::numeric_limits<std::size_t>::max());
    assert(oversized_pool.isError());
    assert(oversized_pool.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);

    auto pool = stream::MemoryPool::create(1);
    assert(pool.isSuccess());
    auto oversized_allocation = pool.value()->allocate(std::numeric_limits<std::size_t>::max());
    assert(oversized_allocation.isError());
    assert(oversized_allocation.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);

    auto chunk_processor = stream::ChunkProcessor::create(stream::StreamingConfig{});
    assert(chunk_processor.isSuccess());
    stream::DataChunk invalid_chunk;
    invalid_chunk.points = {std::numeric_limits<double>::quiet_NaN()};
    invalid_chunk.point_dim = 1;
    invalid_chunk.num_points = 1;
    invalid_chunk.max_radius = 1.0;
    auto invalid_parallel =
        chunk_processor.value()->processChunksParallel({invalid_chunk}, vr_config);
    assert(invalid_parallel.isError());
    assert(invalid_parallel.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);

    return 0;
}
