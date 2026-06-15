#include "nerve/summary/compact_summary.hpp"

#include <cassert>
#include <limits>
#include <vector>

namespace {

nerve::summary::CompactSummaryPipeline makePipeline(bool approximate = false) {
    nerve::summary::CompactSummaryPipeline::PipelineConfig config;
    config.enable_approximation = approximate;
    config.max_data_points = 2;
    config.max_persistence_dim = 1;
    config.max_filtration_radius = 2.0;
    return nerve::summary::CompactSummaryPipeline(config);
}

void assertInvalidInput(
    const nerve::errors::ErrorResult<nerve::summary::CompactSummary>& result) {
    assert(result.isError());
    assert(result.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);
}

void assertResourceLimit(
    const nerve::errors::ErrorResult<nerve::summary::CompactSummary>& result) {
    assert(result.isError());
    assert(result.errorCode() == nerve::errors::ErrorCode::E41_RESOURCE_LIMIT);
}

}  // namespace

int main() {
    const auto exact = makePipeline();
    const std::vector<std::vector<float>> points{{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}};
    const auto summary = exact.computeSummary(points, 123, 7);
    assert(summary.isSuccess());
    assert(summary.value().isValid());
    assert(summary.value().timestamp_ns == 123);
    assert(summary.value().symbol_id == 7);
    assert(summary.value().data_points_count == points.size());
    assert(summary.value().eigenvalue_count == 2);

    const std::vector<std::vector<float>> ragged{{0.0F, 0.0F}, {1.0F}};
    assertInvalidInput(exact.computeSummary(ragged, 123, 7));

    const std::vector<std::vector<float>> nonfinite{
        {0.0F, 0.0F}, {std::numeric_limits<float>::quiet_NaN(), 1.0F}};
    assertInvalidInput(exact.computeSummary(nonfinite, 123, 7));

    const std::vector<std::vector<float>> unsafe_magnitude{
        {0.0F, 0.0F}, {std::numeric_limits<float>::max(), 1.0F}};
    assertResourceLimit(exact.computeSummary(unsafe_magnitude, 123, 7));

    const std::vector<std::vector<float>> infinite{
        {0.0F, 0.0F}, {std::numeric_limits<float>::infinity(), 1.0F}};
    const auto approximate = makePipeline(true).computeApproximateSummary(infinite, 123, 7);
    assert(approximate.data_points_count == 0);
    assert(approximate.lifetime_count == 0);

    return 0;
}
