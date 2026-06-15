
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/utils/diagram_statistics.hpp"
#include "nerve/summary/compact_summary.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>

namespace nerve::summary
{

using nerve::persistence::Pair;

namespace
{

errors::ErrorCode flattenPoints(const std::vector<std::vector<float>> &points,
                                std::vector<double> *flat, std::size_t *point_dim)
{
    if (flat == nullptr || point_dim == nullptr)
    {
        return errors::ErrorCode::E54_PH4_INVALID_INPUT;
    }

    flat->clear();
    *point_dim = 0;
    const auto fail = [&](errors::ErrorCode code) {
        flat->clear();
        *point_dim = 0;
        return code;
    };

    if (points.empty())
    {
        return fail(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    const std::size_t dim = points.front().size();
    if (dim == 0)
    {
        return fail(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (points.size() > std::numeric_limits<std::size_t>::max() / dim)
    {
        return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    const std::size_t value_count = points.size() * dim;
    if (value_count > std::vector<double>().max_size())
    {
        return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<float>::max())) / 4.0L;

    for (const auto &row : points)
    {
        if (row.size() != dim)
        {
            return fail(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
        for (const float value : row)
        {
            if (!std::isfinite(value))
            {
                return fail(errors::ErrorCode::E54_PH4_INVALID_INPUT);
            }
            if (std::abs(static_cast<long double>(value)) > safe_abs)
            {
                return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
            }
        }
    }

    try
    {
        flat->reserve(value_count);
    }
    catch (const std::length_error &)
    {
        return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    catch (const std::bad_alloc &)
    {
        return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }

    for (const auto &row : points)
    {
        for (const float v : row)
        {
            flat->push_back(static_cast<double>(v));
        }
    }
    *point_dim = dim;
    return errors::ErrorCode::SUCCESS;
}

double pairSortLifetime(const Pair &pair)
{
    if (pair.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    return pair.death - pair.birth;
}

errors::ErrorCode validateCompactPair(const Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    const bool infinite_death = pair.isInfinite();
    if (!std::isfinite(pair.birth) || (!finite_death && !infinite_death) || pair.dimension < 0 ||
        (finite_death && pair.death < pair.birth))
    {
        return errors::ErrorCode::E54_PH4_INVALID_INPUT;
    }

    const double max_float = static_cast<double>(std::numeric_limits<float>::max());
    if (std::abs(pair.birth) > max_float || (finite_death && std::abs(pair.death) > max_float))
    {
        return errors::ErrorCode::E41_RESOURCE_LIMIT;
    }
    if (finite_death)
    {
        const double persistence = pair.death - pair.birth;
        if (!std::isfinite(persistence) || persistence > max_float)
        {
            return errors::ErrorCode::E41_RESOURCE_LIMIT;
        }
    }
    return errors::ErrorCode::SUCCESS;
}

uint8_t compactDimension(Dimension dimension)
{
    return static_cast<uint8_t>(std::min<Dimension>(
        dimension, static_cast<Dimension>(std::numeric_limits<uint8_t>::max())));
}

} // namespace

CompactSummaryPipeline::CompactSummaryPipeline(const PipelineConfig &config)
    : config_(config)
    , last_metrics_()
{}

errors::ErrorResult<CompactSummary>
CompactSummaryPipeline::computeFromPointCloud(const std::vector<std::vector<float>> &points,
                                              int64_t timestamp_ns, int64_t data_id,
                                              const core::DeterminismContract &contract) const
{
    (void)contract;
    std::vector<double> flat;
    std::size_t point_dim = 0;
    const errors::ErrorCode flatten_status = flattenPoints(points, &flat, &point_dim);
    if (flatten_status != errors::ErrorCode::SUCCESS)
    {
        return errors::ErrorResult<CompactSummary>::error(flatten_status);
    }

    persistence::PersistenceOptions options;
    options.max_dim = config_.max_persistence_dim;
    options.max_radius = config_.max_filtration_radius;
    options.mode = persistence::PersistenceMode::EXACT;
    options.backend = persistence::PersistenceBackend::CPU_EXACT;

    core::BufferView<const double> view(flat.data(), flat.size());
    auto ph_result = persistence::compute(view, static_cast<Size>(point_dim), options);
    if (ph_result.isError())
    {
        return errors::ErrorResult<CompactSummary>::error(ph_result.errorCode());
    }

    const persistence::PersistenceResult &pr = ph_result.value();
    for (const auto &pair : pr.pairs)
    {
        const errors::ErrorCode pair_status = validateCompactPair(pair);
        if (pair_status != errors::ErrorCode::SUCCESS)
        {
            return errors::ErrorResult<CompactSummary>::error(pair_status);
        }
    }
    const std::vector<Size> betti = persistence::bettiNumbersFromPairs(pr.pairs);

    CompactSummary summary;
    std::vector<Pair> sorted_pairs = pr.pairs;
    std::ranges::sort(sorted_pairs, std::greater{},
                      [](const Pair &p) { return pairSortLifetime(p); });

    summary.lifetime_count = static_cast<uint8_t>(
        std::min(sorted_pairs.size(), static_cast<size_t>(CompactSummary::MAX_LIFETIMES)));
    for (uint8_t i = 0; i < summary.lifetime_count; ++i)
    {
        const Pair &p = sorted_pairs[i];
        const float birth = static_cast<float>(p.birth);
        const float death =
            p.isInfinite() ? std::numeric_limits<float>::infinity() : static_cast<float>(p.death);
        const float pers = p.isInfinite() ? std::numeric_limits<float>::infinity()
                                          : static_cast<float>(p.death - p.birth);
        summary.top_lifetimes[i] = CompactSummary::Lifetime{
            birth,
            death,
            compactDimension(p.dimension),
            pers,
        };
    }

    summary.betti_dimension_count = static_cast<uint8_t>(
        std::min(betti.size(), static_cast<size_t>(CompactSummary::MAX_BETTI_DIM)));
    for (uint8_t d = 0; d < summary.betti_dimension_count; ++d)
    {
        summary.betti_counts[d] =
            static_cast<uint16_t>(std::min(betti[d], static_cast<Size>(65535)));
    }
    for (uint8_t d = summary.betti_dimension_count; d < CompactSummary::MAX_BETTI_DIM; ++d)
    {
        summary.betti_counts[d] = 0;
    }

    std::vector<CompactSummary::Lifetime> finite_lifetimes;
    finite_lifetimes.reserve(pr.pairs.size());
    for (const auto &p : pr.pairs)
    {
        if (!p.isInfinite())
        {
            const float pers = static_cast<float>(p.death - p.birth);
            finite_lifetimes.push_back(
                CompactSummary::Lifetime{static_cast<float>(p.birth), static_cast<float>(p.death),
                                         compactDimension(p.dimension), pers});
        }
    }
    summary.persistence_entropy = computePersistenceEntropy(finite_lifetimes);
    summary.betti_entropy =
        bettiDistributionEntropy(summary.betti_counts, summary.betti_dimension_count);

    const auto eigenvalues = computeTopEigenvalues(points);
    summary.eigenvalue_count = static_cast<uint8_t>(
        std::min(eigenvalues.size(), static_cast<size_t>(CompactSummary::MAX_EIGENVALUES)));
    for (uint8_t i = 0; i < summary.eigenvalue_count; ++i)
    {
        summary.top_eigenvalues[i] = eigenvalues[i];
    }
    summary.spectral_entropy = spectralEntropyFromEigenvalues(eigenvalues);

    summary.data_points_count = static_cast<uint16_t>(
        std::min(points.size(), static_cast<size_t>(std::numeric_limits<uint16_t>::max())));
    summary.noise_level =
        points.empty() ? 0.0F
                       : static_cast<float>(1.0 / std::sqrt(static_cast<double>(points.size())));
    summary.timestamp_ns = timestamp_ns;
    summary.symbol_id = data_id;
    summary.computation_time_us = static_cast<uint32_t>(std::min<double>(
        std::numeric_limits<uint32_t>::max(), std::max(0.0, pr.diagnostics.elapsed_ms * 1000.0)));

    return errors::ErrorResult<CompactSummary>::success(CompactSummary(summary));
}

float CompactSummaryPipeline::bettiDistributionEntropy(
    const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &counts,
    uint8_t betti_dimension_count)
{
    std::vector<double> weights;
    const uint8_t n =
        std::min(betti_dimension_count, static_cast<uint8_t>(CompactSummary::MAX_BETTI_DIM));
    weights.reserve(n);
    for (uint8_t i = 0; i < n; ++i)
    {
        weights.push_back(static_cast<double>(counts[i]));
    }
    return static_cast<float>(persistence::shannonEntropyNormalized(weights));
}

float CompactSummaryPipeline::spectralEntropyFromEigenvalues(
    const std::vector<CompactSummary::Eigenvalue> &eigenvalues)
{
    std::vector<double> weights;
    weights.reserve(eigenvalues.size());
    for (const auto &ev : eigenvalues)
    {
        weights.push_back(static_cast<double>(std::max(0.0F, ev.value)));
    }
    return static_cast<float>(persistence::shannonEntropyNormalized(weights));
}

float CompactSummaryPipeline::accuracyVsReferenceBetti(
    const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &ref_betti, uint8_t ref_dim_count,
    const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &approx_betti,
    uint8_t approx_dim_count)
{
    const uint8_t dmax = std::max(ref_dim_count, approx_dim_count);
    double l1 = 0.0;
    for (uint8_t d = 0; d < dmax; ++d)
    {
        const uint16_t a = d < ref_dim_count ? ref_betti[d] : 0;
        const uint16_t b = d < approx_dim_count ? approx_betti[d] : 0;
        l1 += std::abs(static_cast<int>(a) - static_cast<int>(b));
    }
    return static_cast<float>(1.0 / (1.0 + l1));
}

errors::ErrorResult<CompactSummary>
CompactSummaryPipeline::computeSummary(const std::vector<std::vector<float>> &points,
                                       int64_t timestamp_ns, int64_t data_id,
                                       const core::DeterminismContract &contract) const
{
    if (points.empty())
    {
        return errors::ErrorResult<CompactSummary>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<CompactSummary>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    const auto start = std::chrono::steady_clock::now();
    const bool use_approximate =
        config_.enable_approximation && points.size() > config_.max_data_points;

    std::vector<std::vector<float>> working_points =
        use_approximate ? downsamplePoints(points) : points;

    auto summary_result = computeFromPointCloud(working_points, timestamp_ns, data_id, contract);
    if (summary_result.isError())
    {
        return summary_result;
    }
    CompactSummary summary = summary_result.moveValue();

    float accuracy = 1.0F;
    if (use_approximate)
    {
        const size_t ref_n = std::min(points.size(), config_.max_data_points);
        std::vector<std::vector<float>> refPoints(
            points.begin(), points.begin() + static_cast<std::ptrdiff_t>(ref_n));
        auto ref_result = computeFromPointCloud(refPoints, timestamp_ns, data_id, contract);
        if (ref_result.isSuccess())
        {
            const CompactSummary &ref = ref_result.value();
            accuracy =
                accuracyVsReferenceBetti(ref.betti_counts, ref.betti_dimension_count,
                                         summary.betti_counts, summary.betti_dimension_count);
        }
        else
        {
            accuracy = 0.5F;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    summary.computation_time_us = static_cast<uint32_t>(std::max<int64_t>(0, elapsed_us));
    summary.timestamp_ns = timestamp_ns;
    summary.symbol_id = data_id;

    last_metrics_.last_computation_time_ms = static_cast<float>(elapsed_us) / 1000.0F;
    last_metrics_.points_processed = points.size();
    last_metrics_.used_approximation = use_approximate;
    last_metrics_.accuracy_estimate = accuracy;

    return errors::ErrorResult<CompactSummary>::success(CompactSummary(summary));
}

CompactSummary
CompactSummaryPipeline::computeApproximateSummary(const std::vector<std::vector<float>> &points,
                                                  int64_t timestamp_ns, int64_t data_id,
                                                  const core::DeterminismContract &contract) const
{
    if (points.empty())
    {
        return CompactSummary{};
    }
    const std::vector<std::vector<float>> processed =
        config_.enable_approximation ? downsamplePoints(points) : points;
    auto result = computeFromPointCloud(processed, timestamp_ns, data_id, contract);
    if (result.isError())
    {
        return CompactSummary{};
    }
    return result.moveValue();
}

void CompactSummaryPipeline::updateSummary(const std::vector<std::vector<float>> &new_points,
                                           CompactSummary &existing_summary) const
{
    if (new_points.empty())
    {
        return;
    }
    auto result =
        computeSummary(new_points, existing_summary.timestamp_ns, existing_summary.symbol_id);
    if (result.isSuccess())
    {
        existing_summary = result.moveValue();
    }
}

CompactSummaryPipeline::PerformanceMetrics CompactSummaryPipeline::getLastMetrics() const
{
    return last_metrics_;
}

std::vector<CompactSummary::Eigenvalue>
CompactSummaryPipeline::computeTopEigenvalues(const std::vector<std::vector<float>> &points) const
{
    std::vector<CompactSummary::Eigenvalue> values;
    if (points.empty())
    {
        return values;
    }
    const size_t dims = points.front().size();
    values.reserve(std::min(dims, static_cast<size_t>(CompactSummary::MAX_EIGENVALUES)));
    for (size_t d = 0; d < dims && values.size() < CompactSummary::MAX_EIGENVALUES; ++d)
    {
        double mean = 0.0;
        for (const auto &point : points)
        {
            mean += point[d];
        }
        mean /= static_cast<double>(points.size());
        double variance = 0.0;
        for (const auto &point : points)
        {
            const double delta = point[d] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(points.size());
        values.push_back(CompactSummary::Eigenvalue{static_cast<float>(variance), 1});
    }
    std::ranges::sort(values, std::greater{}, &CompactSummary::Eigenvalue::value);
    return values;
}

float CompactSummaryPipeline::computePersistenceEntropy(
    const std::vector<CompactSummary::Lifetime> &lifetimes) const
{
    if (lifetimes.empty())
    {
        return 0.0F;
    }
    std::vector<double> weights;
    weights.reserve(lifetimes.size());
    for (const auto &lifetime : lifetimes)
    {
        weights.push_back(static_cast<double>(std::max(0.0F, lifetime.persistence)));
    }
    return static_cast<float>(persistence::shannonEntropyNormalized(weights));
}

std::vector<std::vector<float>>
CompactSummaryPipeline::downsamplePoints(const std::vector<std::vector<float>> &points) const
{
    if (points.size() <= config_.max_data_points)
    {
        return points;
    }
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(config_.random_seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    indices.resize(config_.max_data_points);
    std::ranges::sort(indices);
    std::vector<std::vector<float>> sampled;
    sampled.reserve(indices.size());
    for (const size_t idx : indices)
    {
        sampled.push_back(points[idx]);
    }
    return sampled;
}

} // namespace nerve::summary
