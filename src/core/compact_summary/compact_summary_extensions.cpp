
#include "nerve/core/detail/compact_summary_extensions.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::core
{

constexpr size_t MAX_VALID_DIMENSION = 100;
constexpr size_t MAX_VALID_BETTI_NUMBER = 1000000;

namespace
{

[[nodiscard]] uint8_t checkedByte(std::size_t value, const char *field)
{
    if (value > std::numeric_limits<uint8_t>::max())
    {
        throw std::out_of_range(field);
    }
    return static_cast<uint8_t>(value);
}

} // namespace

HighDimCompactSummary::HighDimCompactSummary()
    : memory_efficiency_(0.0)
    , computational_efficiency_(0.0)
    , approximation_quality_(0.0)
{}

HighDimCompactSummary::HighDimCompactSummary(const CompactSummary &base_summary)
    : CompactSummary(base_summary)
    , memory_efficiency_(0.0)
    , computational_efficiency_(0.0)
    , approximation_quality_(0.0)
{}

void HighDimCompactSummary::setHighdimBettiNumber(std::size_t dimension, std::size_t value)
{
    highdim_betti_numbers_[dimension] = value;
    updateTop8BettiNumbers();
}

[[nodiscard]] std::size_t HighDimCompactSummary::getHighdimBettiNumber(std::size_t dimension) const
{
    auto it = highdim_betti_numbers_.find(dimension);
    return (it != highdim_betti_numbers_.end()) ? it->second : 0;
}

[[nodiscard]] const std::unordered_map<std::size_t, std::size_t> &
HighDimCompactSummary::getAllHighdimBettiNumbers() const
{
    return highdim_betti_numbers_;
}

void HighDimCompactSummary::setHighdimBettiTop8(const std::vector<std::size_t> &top8)
{
    highdim_betti_top8_.clear();
    const auto count = std::min(top8.size(), size_t(8));
    highdim_betti_top8_.reserve(count);
    std::ranges::copy_n(top8.begin(), static_cast<std::ptrdiff_t>(count),
                        std::back_inserter(highdim_betti_top8_));
}

[[nodiscard]] const std::vector<std::size_t> &HighDimCompactSummary::getHighdimBettiTop8() const
{
    return highdim_betti_top8_;
}

void HighDimCompactSummary::setLifetimeStatistics(const LifetimeStats &stats)
{
    lifetime_stats_ = stats;
}

[[nodiscard]] const LifetimeStats &HighDimCompactSummary::getLifetimeStatistics() const
{
    return lifetime_stats_;
}

void HighDimCompactSummary::setCompressionMetrics(const CompressionMetrics &metrics)
{
    compression_metrics_ = metrics;
}

[[nodiscard]] const CompressionMetrics &HighDimCompactSummary::getCompressionMetrics() const
{
    return compression_metrics_;
}
void HighDimCompactSummary::setHighdimMetrics(const HighDimMetrics &metrics)
{
    highdim_metrics_ = metrics;
}
const HighDimMetrics &HighDimCompactSummary::getHighdimMetrics() const
{
    return highdim_metrics_;
}
void HighDimCompactSummary::setMemoryEfficiency(double efficiency)
{
    memory_efficiency_ = std::max(0.0, std::min(1.0, efficiency));
}
double HighDimCompactSummary::getMemoryEfficiency() const
{
    return memory_efficiency_;
}
void HighDimCompactSummary::setComputationalEfficiency(double efficiency)
{
    computational_efficiency_ = std::max(0.0, std::min(1.0, efficiency));
}
double HighDimCompactSummary::getComputationalEfficiency() const
{
    return computational_efficiency_;
}
void HighDimCompactSummary::setApproximationQuality(double quality)
{
    approximation_quality_ = std::max(0.0, std::min(1.0, quality));
}
double HighDimCompactSummary::getApproximationQuality() const
{
    return approximation_quality_;
}
void HighDimCompactSummary::setDimensionInfo(std::size_t dimension, const DimensionInfo &info)
{
    dimension_info_[dimension] = info;
}
const DimensionInfo &HighDimCompactSummary::getDimensionInfo(std::size_t dimension) const
{
    static const DimensionInfo empty_info{};
    auto it = dimension_info_.find(dimension);
    return (it != dimension_info_.end()) ? it->second : empty_info;
}
void HighDimCompactSummary::serializeHighdimData(std::vector<uint8_t> &buffer) const
{
    buffer.push_back(checkedByte(highdim_betti_numbers_.size(), "Too many Betti entries"));
    for (const auto &[dim, betti] : highdim_betti_numbers_)
    {
        buffer.push_back(checkedByte(dim, "Betti dimension exceeds serialized range"));
        buffer.push_back(checkedByte(betti, "Betti number exceeds serialized range"));
    }
    buffer.push_back(checkedByte(highdim_betti_top8_.size(), "Too many top Betti entries"));
    for (std::size_t betti : highdim_betti_top8_)
    {
        buffer.push_back(checkedByte(betti, "Top Betti number exceeds serialized range"));
    }
    buffer.push_back(static_cast<uint8_t>(memory_efficiency_ * 255));
    buffer.push_back(static_cast<uint8_t>(computational_efficiency_ * 255));
    buffer.push_back(static_cast<uint8_t>(approximation_quality_ * 255));
}
void HighDimCompactSummary::deserializeHighdimData(const std::vector<uint8_t> &buffer)
{
    std::size_t pos = 0;
    if (pos < buffer.size())
    {
        std::size_t num_betti = buffer[pos++];
        const std::size_t max_iterations = std::min(num_betti, (buffer.size() - pos) / 2);
        for (std::size_t i = 0; i < max_iterations; ++i)
        {
            std::size_t dim = buffer[pos++];
            std::size_t betti = buffer[pos++];
            highdim_betti_numbers_[dim] = betti;
        }
    }
    if (pos < buffer.size())
    {
        std::size_t num_top8 = buffer[pos++];
        highdim_betti_top8_.clear();
        for (std::size_t i = 0; i < num_top8 && pos < buffer.size(); ++i)
        {
            highdim_betti_top8_.push_back(buffer[pos++]);
        }
    }
    if (pos + 2 < buffer.size())
    {
        memory_efficiency_ = buffer[pos++] / 255.0;
        computational_efficiency_ = buffer[pos++] / 255.0;
        approximation_quality_ = buffer[pos++] / 255.0;
    }
}
bool HighDimCompactSummary::validateHighdimData() const
{
    for (const auto &[dim, betti] : highdim_betti_numbers_)
    {
        if (dim > MAX_VALID_DIMENSION || betti > MAX_VALID_BETTI_NUMBER)
        {
            return false;
        }
    }
    if (memory_efficiency_ < 0.0 || memory_efficiency_ > 1.0 || computational_efficiency_ < 0.0 ||
        computational_efficiency_ > 1.0 || approximation_quality_ < 0.0 ||
        approximation_quality_ > 1.0)
    {
        return false;
    }
    return true;
}
std::size_t HighDimCompactSummary::getTotalHighdimBettiNumber() const
{
    std::size_t total = 0;
    for (const auto &[dim, betti] : highdim_betti_numbers_)
    {
        if (betti > std::numeric_limits<std::size_t>::max() - total)
        {
            throw std::overflow_error("Total Betti number overflow");
        }
        total += betti;
    }
    return total;
}
double HighDimCompactSummary::getAverageLifetime() const
{
    return lifetime_stats_.mean_lifetime;
}
std::size_t HighDimCompactSummary::getDominantDimension() const
{
    std::size_t dominant_dim = 0;
    std::size_t max_betti = 0;
    for (const auto &[dim, betti] : highdim_betti_numbers_)
    {
        if (betti > max_betti)
        {
            max_betti = betti;
            dominant_dim = dim;
        }
    }
    return dominant_dim;
}
void HighDimCompactSummary::updateTop8BettiNumbers()
{
    std::vector<std::pair<std::size_t, std::size_t>> sortedBetti(highdim_betti_numbers_.begin(),
                                                                 highdim_betti_numbers_.end());
    std::ranges::sort(sortedBetti, std::greater{}, &std::pair<std::size_t, std::size_t>::second);
    highdim_betti_top8_.clear();
    highdim_betti_top8_.reserve(std::min(sortedBetti.size(), std::size_t(8)));
    const std::size_t top_count = std::min(sortedBetti.size(), std::size_t(8));
    for (std::size_t i = 0; i < top_count; ++i)
    {
        highdim_betti_top8_.push_back(sortedBetti[i].second);
    }
}
PH5CompactSummary::PH5CompactSummary()
    : HighDimCompactSummary()
{}
void PH5CompactSummary::setCohomologyMetrics(const CohomologyMetrics &metrics)
{
    cohomology_metrics_ = metrics;
}
const CohomologyMetrics &PH5CompactSummary::getCohomologyMetrics() const
{
    return cohomology_metrics_;
}
void PH5CompactSummary::setReductionStats(const ReductionStats &stats)
{
    reduction_stats_ = stats;
}
[[nodiscard]] const ReductionStats &PH5CompactSummary::getReductionStats() const
{
    return reduction_stats_;
}
void PH5CompactSummary::setOrderingInfo(const OrderingInfo &info)
{
    ordering_info_ = info;
}
[[nodiscard]] const OrderingInfo &PH5CompactSummary::getOrderingInfo() const
{
    return ordering_info_;
}
void PH5CompactSummary::setBudgetInfo(const BudgetInfo &info)
{
    budget_info_ = info;
}
[[nodiscard]] const BudgetInfo &PH5CompactSummary::getBudgetInfo() const
{
    return budget_info_;
}
PH6CompactSummary::PH6CompactSummary()
    : HighDimCompactSummary()
{}
void PH6CompactSummary::setWitnessMetrics(const WitnessMetrics &metrics)
{
    witness_metrics_ = metrics;
}
[[nodiscard]] const WitnessMetrics &PH6CompactSummary::getWitnessMetrics() const
{
    return witness_metrics_;
}
void PH6CompactSummary::setSamplingInfo(const SamplingInfo &info)
{
    sampling_info_ = info;
}
[[nodiscard]] const SamplingInfo &PH6CompactSummary::getSamplingInfo() const
{
    return sampling_info_;
}
void PH6CompactSummary::setLandmarkInfo(const LandmarkInfo &info)
{
    landmark_info_ = info;
}
[[nodiscard]] const LandmarkInfo &PH6CompactSummary::getLandmarkInfo() const
{
    return landmark_info_;
}
void PH6CompactSummary::setTruncationInfo(const TruncationInfo &info)
{
    truncation_info_ = info;
}
[[nodiscard]] const TruncationInfo &PH6CompactSummary::getTruncationInfo() const
{
    return truncation_info_;
}
} // namespace nerve::core
