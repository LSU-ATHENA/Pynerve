#include "nerve/anomaly/topology_drift.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_betti_change_detector_basic()
{
    nerve::anomaly::BettiChangeDetector::ChangeConfig cfg;
    cfg.significance_level = 0.05;
    cfg.min_window_size = 10;

    nerve::anomaly::BettiChangeDetector detector(cfg);

    std::vector<std::vector<double>> betti_seq = {
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    std::vector<int64_t> timestamps = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto changes = detector.detectChanges(betti_seq, timestamps);

    for (const auto &cp : changes)
    {
        if (cp.change_magnitude < 0.0)
        {
            std::cerr << "negative change magnitude\n";
            return false;
        }
        if (cp.p_value < 0.0 || cp.p_value > 1.0)
        {
            std::cerr << "p_value out of range\n";
            return false;
        }
    }

    return true;
}

bool check_lifetime_drift_detector_basic()
{
    nerve::anomaly::LifetimeDriftDetector::DriftConfig cfg;
    cfg.drift_threshold = 0.1;
    cfg.reference_window_size = 5;
    cfg.detection_window_size = 3;
    cfg.distance_metric = "wasserstein";

    nerve::anomaly::LifetimeDriftDetector detector(cfg);

    std::vector<float> ref = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    std::vector<float> curr = {0.6f, 1.1f, 1.6f};

    auto dp = detector.detectSingleDrift(ref, curr, 100);

    if (dp.drift_score < 0.0)
    {
        std::cerr << "drift score negative\n";
        return false;
    }
    if (dp.p_value < 0.0 || dp.p_value > 1.0)
    {
        std::cerr << "p_value out of range\n";
        return false;
    }
    if (dp.distance_value < 0.0)
    {
        std::cerr << "distance value negative\n";
        return false;
    }

    return true;
}

bool check_drift_score_range()
{
    nerve::anomaly::LifetimeDriftDetector::DriftConfig cfg;
    cfg.drift_threshold = 0.5;
    cfg.reference_window_size = 10;
    cfg.detection_window_size = 5;
    cfg.distance_metric = "wasserstein";

    nerve::anomaly::LifetimeDriftDetector detector(cfg);

    std::vector<float> base_lifetimes;
    for (int i = 0; i < 10; ++i)
        base_lifetimes.push_back(static_cast<float>(i) * 0.1f);

    std::vector<float> drift_lifetimes;
    for (int i = 0; i < 5; ++i)
        drift_lifetimes.push_back(static_cast<float>(i) * 0.5f);

    auto dp = detector.detectSingleDrift(base_lifetimes, drift_lifetimes, 1);

    if (dp.drift_score < 0.0f)
    {
        std::cerr << "drift score negative: " << dp.drift_score << "\n";
        return false;
    }

    return true;
}

bool check_update_and_detect()
{
    nerve::anomaly::LifetimeDriftDetector::DriftConfig cfg;
    cfg.drift_threshold = 1.0;
    cfg.reference_window_size = 3;
    cfg.detection_window_size = 2;

    nerve::anomaly::LifetimeDriftDetector detector(cfg);

    bool detected = false;
    nerve::anomaly::LifetimeDriftDetector::DriftPoint dp;
    bool result = detector.updateAndDetect({0.1f, 0.2f}, 1, dp);

    if (dp.drift_score < 0.0)
    {
        std::cerr << "drift score should be >= 0\n";
        return false;
    }

    static_cast<void>(detected);
    static_cast<void>(result);
    return true;
}

bool check_persistence_profile_stats()
{
    nerve::anomaly::PersistenceProfile pp;
    pp.num_bins = 10;
    pp.total_pairs = 100;
    pp.birth_mean = 0.5;
    pp.death_mean = 1.5;
    pp.persistence_mean = 1.0;
    pp.birth_variance = 0.1;
    pp.death_variance = 0.2;
    pp.persistence_variance = 0.15;

    pp.birth_histogram.resize(10, 10);
    pp.death_histogram.resize(10, 10);
    pp.persistence_histogram.resize(10, 10);

    if (pp.total_pairs == 0)
    {
        std::cerr << "profile should have pairs\n";
        return false;
    }
    if (pp.birth_mean > pp.death_mean + kTol)
    {
        std::cerr << "birth mean > death mean\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_betti_change_detector_basic())
    {
        std::cerr << "FAIL: betti change detector basic\n";
        return 1;
    }
    if (!check_lifetime_drift_detector_basic())
    {
        std::cerr << "FAIL: lifetime drift detector basic\n";
        return 1;
    }
    if (!check_drift_score_range())
    {
        std::cerr << "FAIL: drift score range\n";
        return 1;
    }
    if (!check_update_and_detect())
    {
        std::cerr << "FAIL: update and detect\n";
        return 1;
    }
    if (!check_persistence_profile_stats())
    {
        std::cerr << "FAIL: persistence profile stats\n";
        return 1;
    }
    return 0;
}
