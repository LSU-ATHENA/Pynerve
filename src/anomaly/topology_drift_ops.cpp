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
        Size bin = std::min(static_cast<Size>(p.birth / max_val * static_cast<double>(num_bins)),
                            num_bins - 1);
        ++profile.birth_histogram[bin];
        profile.birth_mean += p.birth;

        if (std::isfinite(p.death))
        {
            Size dbin = std::min(
                static_cast<Size>(p.death / max_val * static_cast<double>(num_bins)), num_bins - 1);
            ++profile.death_histogram[dbin];
            profile.death_mean += p.death;
            double pers = p.death - p.birth;
            Size pbin = std::min(static_cast<Size>(pers / max_val * static_cast<double>(num_bins)),
                                 num_bins - 1);
            ++profile.persistence_histogram[pbin];
            profile.persistence_mean += pers;
            ++finite_count;
        }
    }

    Size n = pairs.size();
    profile.birth_mean /= static_cast<double>(n);
    profile.death_mean /= finite_count > 0 ? static_cast<double>(finite_count) : 1.0;
    profile.persistence_mean /= finite_count > 0 ? static_cast<double>(finite_count) : 1.0;

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
    profile.birth_variance /= static_cast<double>(n);
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

// ──────────────────────────────────────────────────────────────────────
// MarketAnomalyDetector
// ──────────────────────────────────────────────────────────────────────

MarketAnomalyDetector::MarketAnomalyDetector(const MarketConfig &config) : config_(config) {}

std::vector<MarketAnomalyDetector::AnomalyEvent>
MarketAnomalyDetector::detectAnomalies(const std::vector<int64_t> &timestamps,
                                       const std::vector<double> &prices,
                                       const std::vector<float> &volumes,
                                       const std::vector<std::vector<float>> &topological_features)
{
    // Seed the normal-behaviour models with the first lookback_window points.
    size_t seed_count = std::min(config_.lookback_window, timestamps.size());
    price_history_.assign(prices.begin(), prices.begin() + static_cast<ptrdiff_t>(seed_count));
    volume_history_.assign(volumes.begin(), volumes.begin() + static_cast<ptrdiff_t>(seed_count));
    topology_history_.assign(topological_features.begin(),
                             topological_features.begin() + static_cast<ptrdiff_t>(seed_count));

    std::vector<AnomalyEvent> events;
    for (size_t i = seed_count; i < timestamps.size(); ++i)
    {
        AnomalyEvent ae;
        if (updateAndDetect(timestamps[i], prices[i], volumes[i], topological_features[i], ae))
            events.push_back(std::move(ae));
    }
    return events;
}

MarketAnomalyDetector::AnomalyEvent
MarketAnomalyDetector::detectSingleAnomaly(int64_t timestamp_ns, double price, float volume,
                                           const std::vector<float> &topological_features)
{
    AnomalyEvent ae;
    ae.timestamp_ns = timestamp_ns;
    ae.is_critical = false;

    std::vector<AnomalyEvent> sub_events;
    if (config_.enable_price_analysis && !price_history_.empty())
    {
        auto pe = detectPriceAnomaly(price, price_history_);
        if (pe.anomaly_score > 0.0)
            sub_events.push_back(std::move(pe));
    }
    if (config_.enable_volume_analysis && !volume_history_.empty())
    {
        auto ve = detectVolumeAnomaly(volume, volume_history_);
        if (ve.anomaly_score > 0.0)
            sub_events.push_back(std::move(ve));
    }
    if (config_.enable_topology_analysis && !topology_history_.empty())
    {
        auto te = detectTopologyAnomaly(topological_features, topology_history_);
        if (te.anomaly_score > 0.0)
            sub_events.push_back(std::move(te));
    }

    if (sub_events.empty())
    {
        ae.anomaly_score = 0.0;
        ae.p_value = 1.0;
        ae.anomaly_type = "none";
        ae.description = "No anomaly detected";
        return ae;
    }

    // Combine sub-events.
    double total_score = 0.0;
    double max_score = 0.0;
    std::vector<std::string> factor_names;
    std::vector<double> factor_scores;
    for (const auto &se : sub_events)
    {
        total_score += se.anomaly_score;
        max_score = std::max(max_score, se.anomaly_score);
        factor_names.push_back(se.anomaly_type);
        factor_scores.push_back(se.anomaly_score);
        ae.contributing_factors.insert(ae.contributing_factors.end(),
                                       se.contributing_factors.begin(),
                                       se.contributing_factors.end());
    }
    auto combined = detectCombinedAnomaly(factor_scores, factor_names);

    ae.anomaly_score = combined.anomaly_score;
    ae.p_value = combined.p_value;
    ae.anomaly_type = combined.anomaly_type;
    ae.description = combined.description;
    ae.is_critical = combined.is_critical;
    return ae;
}

bool MarketAnomalyDetector::updateAndDetect(int64_t timestamp_ns, double price, float volume,
                                             const std::vector<float> &topological_features,
                                             AnomalyEvent &detected_anomaly)
{
    detected_anomaly =
        detectSingleAnomaly(timestamp_ns, price, volume, topological_features);
    price_history_.push_back(price);
    volume_history_.push_back(volume);
    topology_history_.push_back(topological_features);

    // Trim to lookback window.
    while (price_history_.size() > config_.lookback_window)
        price_history_.erase(price_history_.begin());
    while (volume_history_.size() > config_.lookback_window)
        volume_history_.erase(volume_history_.begin());
    while (topology_history_.size() > config_.lookback_window)
        topology_history_.erase(topology_history_.begin());

    return detected_anomaly.anomaly_score > 0.0;
}

MarketAnomalyDetector::AnomalyEvent
MarketAnomalyDetector::detectPriceAnomaly(double current_price,
                                           const std::vector<double> &price_history)
{
    AnomalyEvent ae;
    ae.anomaly_type = "price";
    // Compute z-score from the explicit price_history parameter.
    double z = 0.0;
    if (!price_history.empty())
    {
        double mean = 0.0;
        for (auto p : price_history)
            mean += p;
        mean /= static_cast<double>(price_history.size());
        double var = 0.0;
        for (auto p : price_history)
        {
            double d = p - mean;
            var += d * d;
        }
        var /= static_cast<double>(price_history.size());
        if (var > 0.0)
            z = (current_price - mean) / std::sqrt(var);
    }
    double abs_z = std::abs(z);
    ae.anomaly_score = std::max(0.0, (abs_z - 2.0) / 4.0);
    ae.p_value = 2.0 * (1.0 - 0.5 * (1.0 + std::erf(abs_z / std::sqrt(2.0))));
    ae.is_critical = abs_z > 3.0;
    if (ae.anomaly_score > 0.0)
    {
        ae.description =
            abs_z > 3.0 ? "Critical price spike (z=" + std::to_string(z) + ")"
                        : "Price anomaly detected (z=" + std::to_string(z) + ")";
        ae.contributing_factors = {std::abs(z)};
    }
    else
    {
        ae.description = "No price anomaly";
    }
    return ae;
}

MarketAnomalyDetector::AnomalyEvent
MarketAnomalyDetector::detectVolumeAnomaly(float current_volume,
                                            const std::vector<float> &volume_history)
{
    AnomalyEvent ae;
    ae.anomaly_type = "volume";
    // Compute z-score from the explicit volume_history parameter.
    double z = 0.0;
    if (!volume_history.empty())
    {
        double mean = 0.0;
        for (auto v : volume_history)
            mean += static_cast<double>(v);
        mean /= static_cast<double>(volume_history.size());
        double var = 0.0;
        for (auto v : volume_history)
        {
            double d = static_cast<double>(v) - mean;
            var += d * d;
        }
        var /= static_cast<double>(volume_history.size());
        if (var > 0.0)
            z = (static_cast<double>(current_volume) - mean) / std::sqrt(var);
    }
    double abs_z = std::abs(z);
    ae.anomaly_score = std::max(0.0, (abs_z - 2.0) / 4.0);
    ae.p_value = 2.0 * (1.0 - 0.5 * (1.0 + std::erf(abs_z / std::sqrt(2.0))));
    ae.is_critical = abs_z > 3.0;
    if (ae.anomaly_score > 0.0)
    {
        ae.description =
            abs_z > 3.0 ? "Critical volume spike (z=" + std::to_string(z) + ")"
                        : "Volume anomaly detected (z=" + std::to_string(z) + ")";
        ae.contributing_factors = {static_cast<double>(std::abs(z))};
    }
    else
    {
        ae.description = "No volume anomaly";
    }
    return ae;
}

MarketAnomalyDetector::AnomalyEvent
MarketAnomalyDetector::detectTopologyAnomaly(
    const std::vector<float> &current_features,
    const std::vector<std::vector<float>> &feature_history)
{
    AnomalyEvent ae;
    ae.anomaly_type = "topology";
    // Compute anomaly score from the explicit feature_history parameter.
    double score = 0.0;
    if (!feature_history.empty())
    {
        size_t dim = current_features.size();
        std::vector<double> mean(dim, 0.0);
        for (const auto &fv : feature_history)
        {
            for (size_t j = 0; j < dim && j < fv.size(); ++j)
                mean[j] += static_cast<double>(fv[j]);
        }
        for (auto &m : mean)
            m /= static_cast<double>(feature_history.size());
        double dist = 0.0;
        for (size_t j = 0; j < dim && j < current_features.size(); ++j)
        {
            double d = static_cast<double>(current_features[j]) - mean[j];
            dist += d * d;
        }
        score = std::sqrt(dist) / std::sqrt(static_cast<double>(dim) + 1e-10);
    }
    ae.anomaly_score = std::max(0.0, (score - config_.topology_anomaly_threshold) /
                                         (config_.topology_anomaly_threshold + 1e-10));
    ae.p_value = std::exp(-score);
    ae.is_critical = score > config_.topology_anomaly_threshold * 1.5;
    if (ae.anomaly_score > 0.0)
    {
        ae.description = ae.is_critical ? "Critical topological anomaly"
                                        : "Topological anomaly detected";
        ae.contributing_factors = {score};
    }
    else
    {
        ae.description = "No topology anomaly";
    }
    return ae;
}

MarketAnomalyDetector::AnomalyEvent MarketAnomalyDetector::detectCombinedAnomaly(
    const std::vector<double> &factor_scores, const std::vector<std::string> &factor_names)
{
    AnomalyEvent ae;
    ae.anomaly_type = "combined";
    if (factor_scores.empty())
    {
        ae.anomaly_score = 0.0;
        ae.p_value = 1.0;
        ae.description = "No anomaly factors";
        return ae;
    }
    double sum = 0.0, max_s = 0.0;
    for (auto s : factor_scores)
    {
        sum += s;
        max_s = std::max(max_s, s);
    }
    ae.anomaly_score = (sum + max_s) / 2.0;
    ae.p_value = std::exp(-ae.anomaly_score);
    ae.is_critical = max_s > 0.8;
    ae.description =
        "Combined anomaly (factors: " + std::to_string(factor_scores.size()) + ")";
    ae.contributing_factors = factor_scores;
    return ae;
}

void MarketAnomalyDetector::updateNormalBehavior(
    const std::vector<double> &prices, const std::vector<float> &volumes,
    const std::vector<std::vector<float>> &topological_features)
{
    price_history_ = prices;
    volume_history_ = volumes;
    topology_history_ = topological_features;
    while (price_history_.size() > config_.lookback_window)
        price_history_.erase(price_history_.begin());
    while (volume_history_.size() > config_.lookback_window)
        volume_history_.erase(volume_history_.begin());
    while (topology_history_.size() > config_.lookback_window)
        topology_history_.erase(topology_history_.begin());
}

void MarketAnomalyDetector::resetNormalBehavior()
{
    price_history_.clear();
    volume_history_.clear();
    topology_history_.clear();
}

double MarketAnomalyDetector::computePriceZscore(double current_price) const
{
    if (price_history_.empty())
        return 0.0;
    double mean = 0.0;
    for (auto p : price_history_)
        mean += p;
    mean /= static_cast<double>(price_history_.size());
    double var = 0.0;
    for (auto p : price_history_)
    {
        double d = p - mean;
        var += d * d;
    }
    var /= static_cast<double>(price_history_.size());
    if (var <= 0.0)
        return 0.0;
    return (current_price - mean) / std::sqrt(var);
}

double MarketAnomalyDetector::computeVolumeZscore(float current_volume) const
{
    if (volume_history_.empty())
        return 0.0;
    double mean = 0.0;
    for (auto v : volume_history_)
        mean += static_cast<double>(v);
    mean /= static_cast<double>(volume_history_.size());
    double var = 0.0;
    for (auto v : volume_history_)
    {
        double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= static_cast<double>(volume_history_.size());
    if (var <= 0.0)
        return 0.0;
    return (static_cast<double>(current_volume) - mean) / std::sqrt(var);
}

double MarketAnomalyDetector::computeTopologyAnomalyScore(
    const std::vector<float> &current_features) const
{
    if (topology_history_.empty())
        return 0.0;
    // Compute mean feature vector.
    size_t dim = current_features.size();
    std::vector<double> mean(dim, 0.0);
    for (const auto &fv : topology_history_)
    {
        for (size_t j = 0; j < dim && j < fv.size(); ++j)
            mean[j] += static_cast<double>(fv[j]);
    }
    for (auto &m : mean)
        m /= static_cast<double>(topology_history_.size());

    // Euclidean distance from mean, normalized by dimension.
    double dist = 0.0;
    for (size_t j = 0; j < dim && j < current_features.size(); ++j)
    {
        double d = static_cast<double>(current_features[j]) - mean[j];
        dist += d * d;
    }
    return std::sqrt(dist) / std::sqrt(static_cast<double>(dim) + 1e-10);
}

std::vector<double>
MarketAnomalyDetector::normalizeFeatures(const std::vector<double> &features) const
{
    if (features.empty())
        return {};
    double sum = 0.0;
    for (auto f : features)
        sum += f;
    if (sum <= 0.0)
        return features;
    std::vector<double> result;
    result.reserve(features.size());
    for (auto f : features)
        result.push_back(f / sum);
    return result;
}

// ──────────────────────────────────────────────────────────────────────
// OnlinePValueCalculator
// ──────────────────────────────────────────────────────────────────────

OnlinePValueCalculator::OnlinePValueCalculator(const PValueConfig &config) : config_(config) {}

double OnlinePValueCalculator::computePValue(double test_statistic,
                                             const std::vector<double> &null_distribution)
{
    return computeTwoSidedPValue(test_statistic, null_distribution);
}

double OnlinePValueCalculator::computeEmpiricalPValue(
    double test_statistic, const std::vector<double> &sample_distribution)
{
    if (sample_distribution.empty())
        return 1.0;
    size_t count = 0;
    for (auto v : sample_distribution)
    {
        if (std::abs(v) >= std::abs(test_statistic))
            ++count;
    }
    return static_cast<double>(count) / static_cast<double>(sample_distribution.size());
}

std::vector<double>
OnlinePValueCalculator::bonferroniCorrection(const std::vector<double> &p_values)
{
    size_t m = p_values.size();
    std::vector<double> corrected;
    corrected.reserve(m);
    for (auto p : p_values)
        corrected.push_back(std::min(1.0, p * static_cast<double>(m)));
    return corrected;
}

std::vector<double>
OnlinePValueCalculator::benjaminiHochbergFdr(const std::vector<double> &p_values)
{
    size_t m = p_values.size();
    if (m == 0)
        return {};
    // Store (p_value, index) pairs for sorting.
    std::vector<std::pair<double, size_t>> sorted;
    sorted.reserve(m);
    for (size_t i = 0; i < m; ++i)
        sorted.emplace_back(p_values[i], i);
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> adjusted(m, 0.0);
    double prev = 0.0;
    for (size_t k = 0; k < m; ++k)
    {
        double bh = sorted[k].first * static_cast<double>(m) / static_cast<double>(k + 1);
        double val = std::max(prev, std::min(1.0, bh));
        adjusted[sorted[k].second] = val;
        prev = val;
    }
    return adjusted;
}

void OnlinePValueCalculator::updateNullDistribution(double new_sample)
{
    std::unique_lock lock(mutex_);
    null_distribution_.push_back(new_sample);
    if (null_distribution_.size() > 10000)
        null_distribution_.erase(null_distribution_.begin());
}

void OnlinePValueCalculator::updateSampleDistribution(double new_sample)
{
    std::unique_lock lock(mutex_);
    sample_distribution_.push_back(new_sample);
    if (sample_distribution_.size() > 10000)
        sample_distribution_.erase(sample_distribution_.begin());
}

bool OnlinePValueCalculator::isSignificant(double p_value) const
{
    return p_value < config_.significance_level;
}

std::vector<bool>
OnlinePValueCalculator::multipleTestingSignificance(const std::vector<double> &p_values)
{
    std::vector<double> corrected;
    if (config_.enable_fdr_control)
        corrected = benjaminiHochbergFdr(p_values);
    else
        corrected = bonferroniCorrection(p_values);
    std::vector<bool> result;
    result.reserve(corrected.size());
    for (auto p : corrected)
        result.push_back(p < config_.significance_level);
    return result;
}

double OnlinePValueCalculator::computeEffectSize(double sample_mean, double null_mean,
                                                 double sample_std, double null_std)
{
    double pooled_std = std::sqrt((sample_std * sample_std + null_std * null_std) / 2.0);
    if (pooled_std <= 0.0)
        return 0.0;
    return (sample_mean - null_mean) / pooled_std;
}

double OnlinePValueCalculator::computeConfidenceInterval(const std::vector<double> &samples,
                                                         double confidence_level)
{
    if (samples.empty())
        return 0.0;
    double mean = 0.0;
    for (auto s : samples)
        mean += s;
    mean /= static_cast<double>(samples.size());
    double var = 0.0;
    for (auto s : samples)
    {
        double d = s - mean;
        var += d * d;
    }
    var /= static_cast<double>(samples.size());
    double se = std::sqrt(var / static_cast<double>(samples.size()));
    // Approximate 1.96 for 95%, 2.576 for 99%, etc.
    double z = 1.96;
    if (confidence_level > 0.97)
        z = 2.326;
    if (confidence_level > 0.98)
        z = 2.576;
    return z * se;
}

double OnlinePValueCalculator::computeTwoSidedPValue(
    double test_statistic, const std::vector<double> &distribution) const
{
    if (distribution.empty())
        return 1.0;
    double abs_stat = std::abs(test_statistic);
    size_t count = 0;
    for (auto v : distribution)
    {
        if (std::abs(v) >= abs_stat)
            ++count;
    }
    return static_cast<double>(count) / static_cast<double>(distribution.size());
}

// ──────────────────────────────────────────────────────────────────────
// RegimeChangeDetector
// ──────────────────────────────────────────────────────────────────────

RegimeChangeDetector::RegimeChangeDetector(const RegimeConfig &config) : config_(config) {}

std::vector<RegimeChangeDetector::Regime>
RegimeChangeDetector::detectRegimes(const std::vector<std::vector<float>> &topological_features,
                                    const std::vector<int64_t> &timestamps)
{
    std::vector<Regime> regimes;
    if (topological_features.size() < config_.min_regime_duration || timestamps.empty())
        return regimes;

    // Use k-means-like clustering to partition features into regimes.
    auto clusters = clusterFeatures(topological_features, config_.num_regimes);
    if (clusters.size() < 2)
        return regimes;

    // Find contiguous runs of the same cluster assignment.
    size_t n = topological_features.size();
    size_t cur_cluster = 0;
    int64_t start_ts = timestamps[0];
    size_t run_len = 1;

    for (size_t i = 1; i < n; ++i)
    {
        size_t cluster_id = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (size_t c = 0; c < clusters.size(); ++c)
        {
            double dist = 0.0;
            for (size_t j = 0; j < topological_features[i].size() && j < clusters[c].size(); ++j)
            {
                double d = static_cast<double>(topological_features[i][j]) - clusters[c][j];
                dist += d * d;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                cluster_id = c;
            }
        }
        if (cluster_id != cur_cluster)
        {
            if (run_len >= config_.min_regime_duration)
            {
                Regime r;
                r.regime_id = static_cast<int>(cur_cluster);
                r.start_timestamp_ns = start_ts;
                r.end_timestamp_ns = timestamps[i - 1];
                r.duration_points = run_len;
                auto features = extractRegimeFeatures(std::vector<std::vector<float>>(
                    topological_features.begin() + static_cast<ptrdiff_t>(i - run_len),
                    topological_features.begin() + static_cast<ptrdiff_t>(i)));
                r.characteristic_features = features;
                r.stability_score =
                    1.0 - std::min(1.0, min_dist / (static_cast<double>(topological_features[i].size()) + 1e-10));
                r.description = characterizeRegime(features);
                regimes.push_back(std::move(r));
            }
            cur_cluster = cluster_id;
            start_ts = timestamps[i];
            run_len = 1;
        }
        else
        {
            ++run_len;
        }
    }
    // Emit final regime.
    if (run_len >= config_.min_regime_duration)
    {
        Regime r;
        r.regime_id = static_cast<int>(cur_cluster);
        r.start_timestamp_ns = start_ts;
        r.end_timestamp_ns = timestamps.back();
        r.duration_points = run_len;
        auto features = extractRegimeFeatures(std::vector<std::vector<float>>(
            topological_features.end() - static_cast<ptrdiff_t>(run_len),
            topological_features.end()));
        r.characteristic_features = features;
        r.stability_score = 1.0;
        r.description = characterizeRegime(features);
        regimes.push_back(std::move(r));
    }

    current_regimes_ = regimes;
    return regimes;
}

std::vector<RegimeChangeDetector::RegimeChange>
RegimeChangeDetector::detectRegimeChanges(
    const std::vector<std::vector<float>> &topological_features,
    const std::vector<int64_t> &timestamps)
{
    auto regimes = detectRegimes(topological_features, timestamps);
    std::vector<RegimeChange> changes;
    if (regimes.size() < 2)
        return changes;

    for (size_t i = 1; i < regimes.size(); ++i)
    {
        RegimeChange rc;
        rc.timestamp_ns = regimes[i].start_timestamp_ns;
        rc.from_regime_id = regimes[i - 1].regime_id;
        rc.to_regime_id = regimes[i].regime_id;
        rc.change_confidence =
            computeRegimeSimilarity(regimes[i - 1].characteristic_features,
                                    regimes[i].characteristic_features);
        rc.change_confidence = 1.0 - rc.change_confidence;
        rc.change_description =
            "Regime " + std::to_string(regimes[i - 1].regime_id) + " -> " +
            std::to_string(regimes[i].regime_id);
        changes.push_back(std::move(rc));
    }
    return changes;
}

bool RegimeChangeDetector::updateAndDetect(const std::vector<float> &new_features,
                                            int64_t timestamp_ns, RegimeChange &detected_change)
{
    feature_history_.push_back(new_features);
    timestamp_history_.push_back(timestamp_ns);
    if (feature_history_.size() < config_.min_regime_duration * 2)
        return false;

    auto changes = detectRegimeChanges(feature_history_, timestamp_history_);
    if (!changes.empty())
    {
        detected_change = changes.back();
        return detected_change.change_confidence >= config_.regime_change_threshold;
    }
    return false;
}

std::vector<double>
RegimeChangeDetector::extractRegimeFeatures(const std::vector<std::vector<float>> &feature_window)
{
    if (feature_window.empty())
        return {};
    size_t dim = feature_window[0].size();
    std::vector<double> features(dim, 0.0);
    for (const auto &fv : feature_window)
    {
        for (size_t j = 0; j < dim && j < fv.size(); ++j)
            features[j] += static_cast<double>(fv[j]);
    }
    for (auto &f : features)
        f /= static_cast<double>(feature_window.size());
    return features;
}

std::string RegimeChangeDetector::characterizeRegime(const std::vector<double> &features)
{
    if (features.empty())
        return "unknown";
    double mean = 0.0;
    for (auto f : features)
        mean += f;
    mean /= static_cast<double>(features.size());
    if (mean < 0.2)
        return "low-activity";
    if (mean < 0.5)
        return "moderate";
    if (mean < 0.8)
        return "high-activity";
    return "extreme";
}

RegimeChangeDetector::HMMModel
RegimeChangeDetector::trainHmm(const std::vector<std::vector<float>> &features,
                               const std::vector<int> &regime_labels)
{
    HMMModel model;
    if (features.empty() || regime_labels.empty() || features.size() != regime_labels.size())
        return model;

    size_t n_states = config_.num_regimes;
    size_t dim = features[0].size();
    std::vector<size_t> counts(n_states, 0);

    // Emission means: average features per regime.
    model.emission_means.resize(n_states, std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < features.size(); ++i)
    {
        size_t r = static_cast<size_t>(regime_labels[i]) % n_states;
        ++counts[r];
        for (size_t j = 0; j < dim && j < features[i].size(); ++j)
            model.emission_means[r][j] += static_cast<double>(features[i][j]);
    }
    for (size_t r = 0; r < n_states; ++r)
    {
        if (counts[r] > 0)
        {
            for (auto &v : model.emission_means[r])
                v /= static_cast<double>(counts[r]);
        }
    }

    // Transition matrix: count transitions between regimes.
    model.transition_matrix.resize(n_states, std::vector<double>(n_states, 0.0));
    for (size_t i = 1; i < regime_labels.size(); ++i)
    {
        size_t from = static_cast<size_t>(regime_labels[i - 1]) % n_states;
        size_t to = static_cast<size_t>(regime_labels[i]) % n_states;
        model.transition_matrix[from][to] += 1.0;
    }
    for (size_t r = 0; r < n_states; ++r)
    {
        double total = 0.0;
        for (size_t c = 0; c < n_states; ++c)
            total += model.transition_matrix[r][c];
        if (total > 0.0)
        {
            for (size_t c = 0; c < n_states; ++c)
                model.transition_matrix[r][c] /= total;
        }
        else
        {
            for (size_t c = 0; c < n_states; ++c)
                model.transition_matrix[r][c] = 1.0 / static_cast<double>(n_states);
        }
    }

    // Initial probabilities: proportional to regime counts.
    model.initial_probabilities.resize(n_states, 0.0);
    size_t total_count = 0;
    for (auto c : counts)
        total_count += c;
    for (size_t r = 0; r < n_states; ++r)
        model.initial_probabilities[r] =
            total_count > 0 ? static_cast<double>(counts[r]) / static_cast<double>(total_count) : 1.0 / static_cast<double>(n_states);

    return model;
}

std::vector<int> RegimeChangeDetector::predictRegimesHmm(
    const HMMModel &model, const std::vector<std::vector<float>> &features)
{
    if (features.empty() || model.emission_means.empty())
        return {};

    size_t n_states = model.emission_means.size();
    std::vector<int> predictions;
    predictions.reserve(features.size());

    for (const auto &fv : features)
    {
        double min_dist = std::numeric_limits<double>::max();
        size_t best = 0;
        for (size_t s = 0; s < n_states; ++s)
        {
            double dist = 0.0;
            for (size_t j = 0; j < fv.size() && j < model.emission_means[s].size(); ++j)
            {
                double d = static_cast<double>(fv[j]) - model.emission_means[s][j];
                dist += d * d;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                best = s;
            }
        }
        predictions.push_back(static_cast<int>(best));
    }
    return predictions;
}

double RegimeChangeDetector::computeRegimeSimilarity(const std::vector<double> &features1,
                                                      const std::vector<double> &features2)
{
    if (features1.empty() || features2.empty())
        return 0.0;
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < features1.size() && i < features2.size(); ++i)
    {
        dot += features1[i] * features2[i];
        norm1 += features1[i] * features1[i];
        norm2 += features2[i] * features2[i];
    }
    double denom = std::sqrt(norm1) * std::sqrt(norm2);
    if (denom <= 0.0)
        return 0.0;
    return std::max(0.0, dot / denom);
}

std::vector<std::vector<float>>
RegimeChangeDetector::clusterFeatures(const std::vector<std::vector<float>> &features,
                                       size_t num_clusters)
{
    if (features.empty() || num_clusters == 0)
        return {};

    size_t dim = features[0].size();
    size_t n = features.size();
    num_clusters = std::min(num_clusters, n);

    // Initialize centroids with evenly-spaced samples.
    std::vector<std::vector<float>> centroids(num_clusters);
    for (size_t c = 0; c < num_clusters; ++c)
    {
        size_t idx = c * n / num_clusters;
        centroids[c] = features[idx];
    }

    // Simple k-means: 10 iterations.
    for (int iter = 0; iter < 10; ++iter)
    {
        std::vector<std::vector<std::vector<float>>> assignments(num_clusters);
        for (const auto &fv : features)
        {
            double min_dist = std::numeric_limits<double>::max();
            size_t best = 0;
            for (size_t c = 0; c < num_clusters; ++c)
            {
                double dist = 0.0;
                for (size_t j = 0; j < dim && j < fv.size() && j < centroids[c].size(); ++j)
                {
                    double d = static_cast<double>(fv[j]) - static_cast<double>(centroids[c][j]);
                    dist += d * d;
                }
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best = c;
                }
            }
            assignments[best].push_back(fv);
        }
        for (size_t c = 0; c < num_clusters; ++c)
        {
            if (assignments[c].empty())
                continue;
            centroids[c].assign(dim, 0.0f);
            for (const auto &fv : assignments[c])
            {
                for (size_t j = 0; j < dim && j < fv.size(); ++j)
                    centroids[c][j] += fv[j];
            }
            for (auto &v : centroids[c])
                v /= static_cast<float>(assignments[c].size());
        }
    }
    return centroids;
}

// ──────────────────────────────────────────────────────────────────────
// AnomalyDetectionManager (singleton)
// ──────────────────────────────────────────────────────────────────────

AnomalyDetectionManager &AnomalyDetectionManager::instance()
{
    static AnomalyDetectionManager mgr;
    return mgr;
}

void AnomalyDetectionManager::setBettiDetectorConfig(
    const BettiChangeDetector::ChangeConfig &config)
{
    std::unique_lock lock(mutex_);
    betti_detector_ = std::make_shared<BettiChangeDetector>(config);
}

void AnomalyDetectionManager::setDriftDetectorConfig(
    const LifetimeDriftDetector::DriftConfig &config)
{
    std::unique_lock lock(mutex_);
    drift_detector_ = std::make_shared<LifetimeDriftDetector>(config);
}

void AnomalyDetectionManager::setMarketDetectorConfig(
    const MarketAnomalyDetector::MarketConfig &config)
{
    std::unique_lock lock(mutex_);
    market_detector_ = std::make_shared<MarketAnomalyDetector>(config);
}

void AnomalyDetectionManager::setPvalueConfig(
    const OnlinePValueCalculator::PValueConfig &config)
{
    std::unique_lock lock(mutex_);
    pvalue_calculator_ = std::make_shared<OnlinePValueCalculator>(config);
}

void AnomalyDetectionManager::setRegimeConfig(
    const RegimeChangeDetector::RegimeConfig &config)
{
    std::unique_lock lock(mutex_);
    regime_detector_ = std::make_shared<RegimeChangeDetector>(config);
}

std::shared_ptr<BettiChangeDetector> AnomalyDetectionManager::getBettiDetector()
{
    std::unique_lock lock(mutex_);
    if (!betti_detector_)
        betti_detector_ =
            std::make_shared<BettiChangeDetector>(BettiChangeDetector::ChangeConfig{});
    return betti_detector_;
}

std::shared_ptr<LifetimeDriftDetector> AnomalyDetectionManager::getDriftDetector()
{
    std::unique_lock lock(mutex_);
    if (!drift_detector_)
        drift_detector_ =
            std::make_shared<LifetimeDriftDetector>(LifetimeDriftDetector::DriftConfig{});
    return drift_detector_;
}

std::shared_ptr<MarketAnomalyDetector> AnomalyDetectionManager::getMarketDetector()
{
    std::unique_lock lock(mutex_);
    if (!market_detector_)
        market_detector_ =
            std::make_shared<MarketAnomalyDetector>(MarketAnomalyDetector::MarketConfig{});
    return market_detector_;
}

std::shared_ptr<OnlinePValueCalculator> AnomalyDetectionManager::getPvalueCalculator()
{
    std::unique_lock lock(mutex_);
    if (!pvalue_calculator_)
        pvalue_calculator_ =
            std::make_shared<OnlinePValueCalculator>(OnlinePValueCalculator::PValueConfig{});
    return pvalue_calculator_;
}

std::shared_ptr<RegimeChangeDetector> AnomalyDetectionManager::getRegimeDetector()
{
    std::unique_lock lock(mutex_);
    if (!regime_detector_)
        regime_detector_ =
            std::make_shared<RegimeChangeDetector>(RegimeChangeDetector::RegimeConfig{});
    return regime_detector_;
}

AnomalyDetectionManager::AnomalyReport AnomalyDetectionManager::detectAllAnomalies(
    const std::vector<int64_t> &timestamps, const std::vector<double> &prices,
    const std::vector<float> &volumes,
    const std::vector<std::vector<double>> &betti_sequences,
    const std::vector<std::vector<float>> &lifetime_sequences,
    const std::vector<std::vector<float>> &topological_features)
{
    AnomalyReport report;
    auto betti = getBettiDetector();
    auto drift = getDriftDetector();
    auto market = getMarketDetector();
    auto regime = getRegimeDetector();

    report.betti_changes = betti->detectChanges(betti_sequences, timestamps);
    report.drift_points = drift->detectDrift(lifetime_sequences, timestamps);
    report.market_anomalies = market->detectAnomalies(timestamps, prices, volumes, topological_features);
    report.regime_changes = regime->detectRegimeChanges(topological_features, timestamps);

    // Overall anomaly score: weighted average of per-detector event scores.
    double score = 0.0;
    int count = 0;
    for (const auto &cp : report.betti_changes)
    {
        score += cp.change_magnitude;
        ++count;
    }
    for (const auto &dp : report.drift_points)
    {
        score += dp.drift_score;
        ++count;
    }
    for (const auto &ae : report.market_anomalies)
    {
        score += ae.anomaly_score;
        ++count;
    }
    for (const auto &rc : report.regime_changes)
    {
        score += rc.change_confidence;
        ++count;
    }
    report.overall_anomaly_score = count > 0 ? score / static_cast<double>(count) : 0.0;
    report.summary_report = generateSummaryReport(report);
    return report;
}

std::vector<std::string>
AnomalyDetectionManager::generateAlerts(const AnomalyReport &report)
{
    std::vector<std::string> alerts;
    if (report.overall_anomaly_score > 0.5)
        alerts.push_back("HIGH: Overall anomaly score " +
                         std::to_string(report.overall_anomaly_score));
    else if (report.overall_anomaly_score > 0.2)
        alerts.push_back("MEDIUM: Overall anomaly score " +
                         std::to_string(report.overall_anomaly_score));
    if (!report.betti_changes.empty())
        alerts.push_back("Betti changes detected: " +
                         std::to_string(report.betti_changes.size()));
    if (!report.drift_points.empty())
        alerts.push_back("Lifetime drift detected: " +
                         std::to_string(report.drift_points.size()));
    if (!report.market_anomalies.empty())
        alerts.push_back("Market anomalies detected: " +
                         std::to_string(report.market_anomalies.size()));
    if (!report.regime_changes.empty())
        alerts.push_back("Regime changes detected: " +
                         std::to_string(report.regime_changes.size()));
    return alerts;
}

bool AnomalyDetectionManager::sendAlerts(const std::vector<std::string> &alerts)
{
    // Stub: alerts are consumed by the caller via generateAlerts().
    // Returns true if there were alerts to send.
    return !alerts.empty();
}

std::string
AnomalyDetectionManager::generateSummaryReport(const AnomalyReport &report)
{
    std::string summary = "Anomaly Report\n";
    summary += "  Betti changes: " + std::to_string(report.betti_changes.size()) + "\n";
    summary += "  Drift points: " + std::to_string(report.drift_points.size()) + "\n";
    summary += "  Market anomalies: " + std::to_string(report.market_anomalies.size()) + "\n";
    summary += "  Regime changes: " + std::to_string(report.regime_changes.size()) + "\n";
    summary +=
        "  Overall score: " + std::to_string(report.overall_anomaly_score);
    return summary;
}

} // namespace nerve::anomaly
