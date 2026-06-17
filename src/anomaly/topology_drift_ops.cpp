#include "nerve/anomaly/topology_drift.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace nerve::anomaly
{

PersistenceProfile computePersistenceProfile(const std::vector<nerve::Pair> &pairs, Size num_bins)
{
    PersistenceProfile profile;
    profile.num_bins = num_bins;
    profile.total_pairs = pairs.size();
    profile.birth_histogram.resize(num_bins, 0);
    profile.death_histogram.resize(num_bins, 0);
    profile.persistence_histogram.resize(num_bins, 0);

    if (pairs.empty())
        return profile;

    double max_birth = 0.0, max_death = 0.0;
    for (const auto &p : pairs)
    {
        max_birth = std::max(max_birth, p.birth);
        if (std::isfinite(p.death))
            max_death = std::max(max_death, p.death);
    }
    double max_val = std::max(max_birth, max_death);
    if (max_val <= 0.0)
        max_val = 1.0;

    profile.birth_mean = 0.0;
    profile.death_mean = 0.0;
    profile.persistence_mean = 0.0;
    Size finite_count = 0;

    for (const auto &p : pairs)
    {
        Size bin = std::min(static_cast<Size>(p.birth / max_val * num_bins), num_bins - 1);
        ++profile.birth_histogram[bin];
        profile.birth_mean += p.birth;

        if (std::isfinite(p.death))
        {
            Size dbin = std::min(static_cast<Size>(p.death / max_val * num_bins), num_bins - 1);
            ++profile.death_histogram[dbin];
            profile.death_mean += p.death;
            double pers = p.death - p.birth;
            Size pbin = std::min(static_cast<Size>(pers / max_val * num_bins), num_bins - 1);
            ++profile.persistence_histogram[pbin];
            profile.persistence_mean += pers;
            ++finite_count;
        }
    }

    Size n = pairs.size();
    profile.birth_mean /= n;
    profile.death_mean /= finite_count > 0 ? finite_count : 1;
    profile.persistence_mean /= finite_count > 0 ? finite_count : 1;

    profile.birth_variance = 0.0;
    profile.death_variance = 0.0;
    profile.persistence_variance = 0.0;
    for (const auto &p : pairs)
    {
        double d = p.birth - profile.birth_mean;
        profile.birth_variance += d * d;
        if (std::isfinite(p.death))
        {
            double dd = p.death - profile.death_mean;
            profile.death_variance += dd * dd;
            double pd = (p.death - p.birth) - profile.persistence_mean;
            profile.persistence_variance += pd * pd;
        }
    }
    profile.birth_variance /= n;
    return profile;
}

double computeDriftScore(const PersistenceProfile &baseline, const PersistenceProfile &current)
{
    if (baseline.total_pairs == 0)
        return 0.0;

    double birth_dist = std::fabs(baseline.birth_mean - current.birth_mean) /
                        (std::sqrt(baseline.birth_variance) + 1e-10);
    double death_dist = std::fabs(baseline.death_mean - current.death_mean) /
                        (std::sqrt(baseline.death_variance) + 1e-10);
    double pers_dist = std::fabs(baseline.persistence_mean - current.persistence_mean) /
                       (std::sqrt(baseline.persistence_variance) + 1e-10);
    double pair_ratio =
        static_cast<double>(current.total_pairs) / static_cast<double>(baseline.total_pairs);

    double hist_dist = 0.0;
    for (Size i = 0; i < std::min(baseline.birth_histogram.size(), current.birth_histogram.size());
         ++i)
    {
        double b = static_cast<double>(baseline.birth_histogram[i]);
        double c = static_cast<double>(current.birth_histogram[i]);
        double sum = b + c;
        if (sum > 0.0)
            hist_dist += (b - c) * (b - c) / sum;
    }

    double score =
        birth_dist + death_dist + pers_dist + std::fabs(1.0 - pair_ratio) * 2.0 + hist_dist * 0.5;
    return score / 5.0;
}

BettiChangeDetector::BettiChangeDetector(const ChangeConfig &config)
    : config_(config)
    , current_window_size_(config.min_window_size)
{}

std::vector<BettiChangeDetector::ChangePoint>
BettiChangeDetector::detectChanges(const std::vector<std::vector<double>> &betti_sequences,
                                   const std::vector<int64_t> &timestamps)
{
    std::vector<ChangePoint> changes;
    if (betti_sequences.size() < config_.min_window_size * 2 || timestamps.size() < 2)
    {
        return changes;
    }

    for (size_t i = config_.min_window_size; i + config_.min_window_size <= betti_sequences.size();
         ++i)
    {
        std::vector<double> before(betti_sequences[i - config_.min_window_size].size());
        std::vector<double> after(betti_sequences[i].size());

        for (size_t j = 0; j < config_.min_window_size; ++j)
        {
            const auto &bseq = betti_sequences[i - config_.min_window_size + j];
            for (size_t k = 0; k < bseq.size() && k < before.size(); ++k)
            {
                before[k] += bseq[k];
            }
        }
        for (size_t j = 0; j < config_.min_window_size && i + j < betti_sequences.size(); ++j)
        {
            const auto &aseq = betti_sequences[i + j];
            for (size_t k = 0; k < aseq.size() && k < after.size(); ++k)
            {
                after[k] += aseq[k];
            }
        }

        for (auto &v : before)
            v /= static_cast<double>(config_.min_window_size);
        for (auto &v : after)
            v /= static_cast<double>(config_.min_window_size);

        auto cp = detectSingleChange(
            before, after, i < timestamps.size() ? timestamps[i] : static_cast<int64_t>(i));
        if (cp.change_magnitude >= config_.min_change_threshold)
        {
            changes.push_back(std::move(cp));
        }
    }
    return changes;
}

BettiChangeDetector::ChangePoint
BettiChangeDetector::detectSingleChange(const std::vector<double> &betti_before,
                                        const std::vector<double> &betti_after,
                                        int64_t timestamp_ns)
{
    ChangePoint cp;
    cp.timestamp_ns = timestamp_ns;
    cp.window_index = 0;
    cp.betti_before = betti_before;
    cp.betti_after = betti_after;
    cp.change_type = "statistical";

    double stat = computeStatisticalDistance(betti_before, betti_after);
    cp.change_magnitude = stat;
    cp.p_value = computePValue(betti_before, betti_after);
    return cp;
}

bool BettiChangeDetector::updateAndDetect(const std::vector<double> &new_betti,
                                          int64_t timestamp_ns, ChangePoint &detected_change)
{
    historical_betti_.push_back(new_betti);
    historical_timestamps_.push_back(timestamp_ns);
    updateWindowSize();

    if (historical_betti_.size() < current_window_size_ * 2)
    {
        return false;
    }

    size_t split = historical_betti_.size() - current_window_size_;
    std::vector<double> before, after;
    for (size_t i = 0; i < current_window_size_ && i < split; ++i)
    {
        const auto &b = historical_betti_[split - current_window_size_ + i];
        if (before.empty())
            before.resize(b.size());
        for (size_t j = 0; j < b.size() && j < before.size(); ++j)
            before[j] += b[j];
    }
    for (size_t i = 0; i < current_window_size_ && split + i < historical_betti_.size(); ++i)
    {
        const auto &b = historical_betti_[split + i];
        if (after.empty())
            after.resize(b.size());
        for (size_t j = 0; j < b.size() && j < after.size(); ++j)
            after[j] += b[j];
    }

    for (auto &v : before)
        v /= static_cast<double>(current_window_size_);
    for (auto &v : after)
        v /= static_cast<double>(current_window_size_);

    detected_change = detectSingleChange(before, after, timestamp_ns);
    return detected_change.change_magnitude >= config_.min_change_threshold;
}

double BettiChangeDetector::computeStatisticalDistance(const std::vector<double> &seq1,
                                                       const std::vector<double> &seq2)
{
    double ks = computeKolmogorovSmirnovStatistic(seq1, seq2);
    double cs = computeChiSquaredStatistic(seq1, seq2);
    return std::max(ks, cs);
}

double BettiChangeDetector::computePValue(const std::vector<double> &seq1,
                                          const std::vector<double> &seq2)
{
    double ks_stat = computeKolmogorovSmirnovStatistic(seq1, seq2);
    size_t n1 = seq1.size();
    size_t n2 = seq2.size();
    if (n1 == 0 || n2 == 0)
    {
        return 1.0;
    }
    double lambda =
        ks_stat * std::sqrt(static_cast<double>(n1 * n2) / static_cast<double>(n1 + n2));
    return 2.0 * std::exp(-2.0 * lambda * lambda);
}

BettiChangeDetector::TrendInfo
BettiChangeDetector::analyzeTrend(const std::vector<double> &betti_sequence) const
{
    TrendInfo info;
    info.is_trending = false;
    info.trend_direction = "stable";

    size_t n = betti_sequence.size();
    if (n < 2)
    {
        info.trend_slope = 0.0;
        info.trend_strength = 0.0;
        return info;
    }

    double mean_x = static_cast<double>(n - 1) / 2.0;
    double mean_y = 0.0;
    for (const auto &v : betti_sequence)
        mean_y += v;
    mean_y /= static_cast<double>(n);

    double num = 0.0, den_x = 0.0, den_y = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double dx = static_cast<double>(i) - mean_x;
        double dy = betti_sequence[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if (den_x <= 0.0 || den_y <= 0.0)
    {
        info.trend_slope = 0.0;
        info.trend_strength = 0.0;
        return info;
    }

    info.trend_slope = num / den_x;
    double r = num / std::sqrt(den_x * den_y);
    info.trend_strength = std::abs(r);

    if (std::abs(r) > 0.5)
    {
        info.is_trending = true;
        info.trend_direction = (info.trend_slope > 0.0) ? "increasing" : "decreasing";
    }

    return info;
}

double BettiChangeDetector::computeKolmogorovSmirnovStatistic(const std::vector<double> &seq1,
                                                              const std::vector<double> &seq2) const
{
    std::vector<double> sorted1 = seq1;
    std::vector<double> sorted2 = seq2;
    std::sort(sorted1.begin(), sorted1.end());
    std::sort(sorted2.begin(), sorted2.end());

    double max_diff = 0.0;
    size_t i = 0, j = 0;
    double n1 = static_cast<double>(sorted1.size());
    double n2 = static_cast<double>(sorted2.size());
    if (n1 <= 0.0 || n2 <= 0.0)
        return 0.0;

    double cdf1 = 0.0, cdf2 = 0.0;
    while (i < sorted1.size() || j < sorted2.size())
    {
        double val;
        if (j >= sorted2.size() || (i < sorted1.size() && sorted1[i] <= sorted2[j]))
        {
            val = sorted1[i];
            while (i < sorted1.size() && sorted1[i] == val)
            {
                cdf1 += 1.0 / n1;
                ++i;
            }
        }
        else
        {
            val = sorted2[j];
            while (j < sorted2.size() && sorted2[j] == val)
            {
                cdf2 += 1.0 / n2;
                ++j;
            }
        }
        double diff = std::abs(cdf1 - cdf2);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

double BettiChangeDetector::computeChiSquaredStatistic(const std::vector<double> &seq1,
                                                       const std::vector<double> &seq2) const
{
    size_t n = std::min(seq1.size(), seq2.size());
    if (n == 0)
        return 0.0;

    double stat = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        if (seq2[i] > 0.0)
        {
            double diff = seq1[i] - seq2[i];
            stat += (diff * diff) / seq2[i];
        }
    }
    return stat;
}

void BettiChangeDetector::updateWindowSize()
{
    size_t n = historical_betti_.size();
    current_window_size_ = std::clamp(n / 2, config_.min_window_size, config_.max_window_size);
}

LifetimeDriftDetector::LifetimeDriftDetector(const DriftConfig &config)
    : config_(config)
    , adaptive_threshold_(config.drift_threshold)
{}

std::vector<LifetimeDriftDetector::DriftPoint>
LifetimeDriftDetector::detectDrift(const std::vector<std::vector<float>> &lifetime_sequences,
                                   const std::vector<int64_t> &timestamps)
{
    std::vector<DriftPoint> drifts;
    if (lifetime_sequences.size() < config_.reference_window_size)
        return drifts;

    std::vector<float> reference;
    for (size_t i = 0; i < config_.reference_window_size && i < lifetime_sequences.size(); ++i)
    {
        for (auto v : lifetime_sequences[i])
            reference.push_back(v);
    }

    for (size_t i = config_.reference_window_size;
         i + config_.detection_window_size <= lifetime_sequences.size(); ++i)
    {
        std::vector<float> current;
        for (size_t j = 0; j < config_.detection_window_size && i + j < lifetime_sequences.size();
             ++j)
        {
            for (auto v : lifetime_sequences[i + j])
                current.push_back(v);
        }

        int64_t ts = i < timestamps.size() ? timestamps[i] : static_cast<int64_t>(i);
        auto dp = detectSingleDrift(reference, current, ts);
        if (dp.is_drift_detected)
        {
            drifts.push_back(std::move(dp));
            updateAdaptiveThreshold(dp.drift_score);
        }
    }
    return drifts;
}

LifetimeDriftDetector::DriftPoint
LifetimeDriftDetector::detectSingleDrift(const std::vector<float> &reference_lifetimes,
                                         const std::vector<float> &current_lifetimes,
                                         int64_t timestamp_ns)
{
    DriftPoint dp;
    dp.timestamp_ns = timestamp_ns;
    dp.reference_lifetimes = reference_lifetimes;
    dp.current_lifetimes = current_lifetimes;

    double wass = computeSlicedWassersteinDistance(reference_lifetimes, current_lifetimes);
    double emd = computeEmdDistance(reference_lifetimes, current_lifetimes);

    dp.distance_value = std::max(wass, emd);
    dp.drift_score = dp.distance_value;

    double kl = computeKlDivergence(reference_lifetimes, current_lifetimes);
    dp.p_value = std::exp(-kl) * 2.0;
    dp.is_drift_detected = dp.drift_score > adaptive_threshold_;
    return dp;
}

bool LifetimeDriftDetector::updateAndDetect(const std::vector<float> &new_lifetimes,
                                            int64_t timestamp_ns, DriftPoint &detected_drift)
{
    historical_lifetimes_.push_back(new_lifetimes);
    historical_timestamps_.push_back(timestamp_ns);

    if (historical_lifetimes_.size() <
        config_.reference_window_size + config_.detection_window_size)
    {
        return false;
    }

    if (reference_lifetimes_.empty())
    {
        for (size_t i = 0; i < config_.reference_window_size && i < historical_lifetimes_.size();
             ++i)
        {
            for (auto v : historical_lifetimes_[i])
                reference_lifetimes_.push_back(v);
        }
    }

    std::vector<float> current;
    size_t start = historical_lifetimes_.size() - config_.detection_window_size;
    for (size_t i = start; i < historical_lifetimes_.size(); ++i)
    {
        for (auto v : historical_lifetimes_[i])
            current.push_back(v);
    }

    detected_drift = detectSingleDrift(reference_lifetimes_, current, timestamp_ns);
    return detected_drift.is_drift_detected;
}

double LifetimeDriftDetector::computeSlicedWassersteinDistance(const std::vector<float> &dist1,
                                                               const std::vector<float> &dist2,
                                                               size_t num_projections)
{
    if (dist1.empty() || dist2.empty())
        return 0.0;

    std::vector<float> s1 = dist1;
    std::vector<float> s2 = dist2;
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());

    size_t n = std::min(s1.size(), s2.size());
    double total = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        total += std::abs(static_cast<double>(s1[i]) - static_cast<double>(s2[i]));
    }
    return total / static_cast<double>(n);
}

double LifetimeDriftDetector::computeEmdDistance(const std::vector<float> &dist1,
                                                 const std::vector<float> &dist2)
{
    if (dist1.empty() && dist2.empty())
        return 0.0;

    std::vector<float> s1 = dist1;
    std::vector<float> s2 = dist2;
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());

    double emd = 0.0;
    double cum1 = 0.0, cum2 = 0.0;
    size_t i = 0, j = 0;
    while (i < s1.size() || j < s2.size())
    {
        double val1 =
            (i < s1.size()) ? static_cast<double>(s1[i]) : std::numeric_limits<double>::max();
        double val2 =
            (j < s2.size()) ? static_cast<double>(s2[j]) : std::numeric_limits<double>::max();

        if (val1 <= val2)
        {
            cum1 += 1.0 / static_cast<double>(s1.size());
            ++i;
        }
        if (val2 <= val1)
        {
            cum2 += 1.0 / static_cast<double>(s2.size());
            ++j;
        }
        emd += std::abs(cum1 - cum2);
    }
    return emd;
}

double LifetimeDriftDetector::computeKlDivergence(const std::vector<float> &dist1,
                                                  const std::vector<float> &dist2)
{
    auto hist1 = computeHistogram(dist1);
    auto hist2 = computeHistogram(dist2);

    if (hist1.empty() || hist2.empty() || hist1.size() != hist2.size())
        return 0.0;

    double kl = 0.0;
    double total1 = 0.0, total2 = 0.0;
    for (const auto &v : hist1)
        total1 += static_cast<double>(v);
    for (const auto &v : hist2)
        total2 += static_cast<double>(v);

    if (total1 <= 0.0 || total2 <= 0.0)
        return 0.0;

    for (size_t i = 0; i < hist1.size(); ++i)
    {
        double p = static_cast<double>(hist1[i]) / total1 + 1e-10;
        double q = static_cast<double>(hist2[i]) / total2 + 1e-10;
        kl += p * std::log(p / q);
    }
    return kl;
}

void LifetimeDriftDetector::updateReferenceDistribution(const std::vector<float> &new_reference)
{
    reference_lifetimes_ = new_reference;
}

void LifetimeDriftDetector::resetReference()
{
    reference_lifetimes_.clear();
}

std::vector<float> LifetimeDriftDetector::computeHistogram(const std::vector<float> &lifetimes,
                                                           size_t num_bins) const
{
    std::vector<float> hist(num_bins, 0.0f);
    if (lifetimes.empty())
        return hist;

    float min_val = *std::min_element(lifetimes.begin(), lifetimes.end());
    float max_val = *std::max_element(lifetimes.begin(), lifetimes.end());
    float range = max_val - min_val;
    if (range <= 0.0f)
        range = 1.0f;

    for (auto v : lifetimes)
    {
        size_t bin =
            std::min(static_cast<size_t>((v - min_val) / range * static_cast<float>(num_bins)),
                     num_bins - 1);
        hist[bin] += 1.0f;
    }
    return hist;
}

void LifetimeDriftDetector::updateAdaptiveThreshold(double current_drift_score)
{
    if (config_.enable_adaptive_threshold)
    {
        adaptive_threshold_ =
            0.5 * adaptive_threshold_ + 0.5 * current_drift_score * config_.drift_threshold;
    }
}

} // namespace nerve::anomaly
