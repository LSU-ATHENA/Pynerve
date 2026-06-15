
#include "nerve/summary/compact_summary.hpp"

#include <algorithm>
#include <memory>

namespace nerve::summary
{

std::unique_ptr<CompactSummaryPipeline>
SummaryFactory::createPipeline(Strategy strategy,
                               const CompactSummaryPipeline::PipelineConfig &config)
{
    CompactSummaryPipeline::PipelineConfig tuned = config;
    if (strategy == Strategy::EXACT)
    {
        tuned.enable_approximation = false;
        tuned.sampling_rate = 1.0F;
    }
    else if (strategy == Strategy::APPROXIMATE)
    {
        tuned.enable_approximation = true;
        tuned.sampling_rate = 0.2F;
    }
    else
    {
        tuned.enable_approximation = true;
        tuned.sampling_rate = 0.1F;
    }
    return std::make_unique<CompactSummaryPipeline>(tuned);
}

errors::ErrorResult<CompactSummary>
SummaryFactory::computeSummary(const std::vector<std::vector<float>> &points, int64_t timestamp_ns,
                               int64_t data_id, float max_time_ms,
                               const core::DeterminismContract &contract)
{
    CompactSummaryPipeline::PipelineConfig exact_config;
    exact_config.enable_approximation = false;
    exact_config.max_computation_time_ms = max_time_ms;

    auto exact = createPipeline(Strategy::EXACT, exact_config);
    return exact->computeSummary(points, timestamp_ns, data_id, contract);
}

bool SummaryValidator::validateSummary(const CompactSummary &summary)
{
    return summary.isValid() && summary.timestamp_ns >= 0 && summary.symbol_id >= 0;
}

bool SummaryValidator::validateSizeConstraints(const CompactSummary &summary)
{
    return summary.isUnderSizeLimit();
}

bool SummaryValidator::validateComputationTime(const CompactSummary &summary, float max_time_ms)
{
    return static_cast<float>(summary.computation_time_us) / 1000.0F <= max_time_ms;
}

float SummaryValidator::estimateAccuracy(const CompactSummary &exact_summary,
                                         const CompactSummary &approximate_summary)
{
    if (exact_summary.lifetime_count == 0)
    {
        return approximate_summary.lifetime_count == 0 ? 1.0F : 0.0F;
    }
    const uint8_t overlap =
        std::min(exact_summary.lifetime_count, approximate_summary.lifetime_count);
    return static_cast<float>(overlap) / static_cast<float>(exact_summary.lifetime_count);
}

std::string SummaryValidator::generateQualityReport(const CompactSummary &summary)
{
    return summary.isValid() ? "compact_summary:valid" : "compact_summary:invalid";
}

} // namespace nerve::summary
