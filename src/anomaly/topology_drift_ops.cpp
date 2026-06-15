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

} // namespace nerve::anomaly
