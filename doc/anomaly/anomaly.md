# Anomaly

## Quick start

```python
import pynerve.anomaly as anom

# Detect drift in persistence lifetimes across time windows
detector = anom.LifetimeDriftDetector(
    drift_threshold=0.1,
    reference_window_size=500,
    distance_metric="wasserstein",
)
drift_points = detector.detectDrift(lifetime_sequences, timestamps)
# -> drift_points[0].drift_score = 0.23, drift_points[0].is_drift_detected = True

# Betti number change detection across sliding windows
betti_detector = anom.BettiChangeDetector(
    significance_level=0.05,
    min_window_size=100,
)
changes = betti_detector.detectChanges(betti_sequences, timestamps)
# -> changes[0].change_magnitude = 0.42, changes[0].p_value = 0.003
```

Compute persistence profiles (birth/death/persistence histograms) and compare
them across time windows to detect topological drift in data streams. Supports
Betti-number tracking, lifetime-distribution drift, and regime-change detection.


## API

```cpp
#include <nerve/anomaly/topology_drift.hpp>

namespace nerve::anomaly {

// Betti-number change detection with statistical testing
class BettiChangeDetector {
    explicit BettiChangeDetector(const ChangeConfig& config);
    vector<ChangePoint> detectChanges(
        const vector<vector<double>>& betti_sequences,
        const vector<int64_t>& timestamps);
    double computeStatisticalDistance(
        const vector<double>& seq1, const vector<double>& seq2);
    TrendInfo analyzeTrend(const vector<double>& betti_sequence) const;
};

// Lifetime distribution drift detection
class LifetimeDriftDetector {
    explicit LifetimeDriftDetector(const DriftConfig& config);
    vector<DriftPoint> detectDrift(
        const vector<vector<float>>& lifetime_sequences,
        const vector<int64_t>& timestamps);
    double computeSlicedWassersteinDistance(
        const vector<float>& dist1, const vector<float>& dist2,
        size_t num_projections = 100);
    void updateReferenceDistribution(const vector<float>& new_reference);
};

// Regime change detection with HMM
class RegimeChangeDetector {
    explicit RegimeChangeDetector(const RegimeConfig& config);
    vector<Regime> detectRegimes(
        const vector<vector<float>>& topological_features,
        const vector<int64_t>& timestamps);
    HMMModel trainHmm(const vector<vector<float>>& features,
                      const vector<int>& regime_labels);
    vector<int> predictRegimesHmm(const HMMModel& model,
                                  const vector<vector<float>>& features);
};

// Combined manager wrapping all detectors
class AnomalyDetectionManager {
    static AnomalyDetectionManager& instance();
    ComprehensiveAnomalyReport detectAllAnomalies(
        const vector<int64_t>& timestamps,
        const vector<double>& prices,
        const vector<float>& volumes,
        const vector<vector<double>>& betti_sequences,
        const vector<vector<float>>& lifetime_sequences,
        const vector<vector<float>>& topological_features);
};

}
```

`compute_profile` extracts birth/death/persistence histograms from diagram
pairs. `drift_score` compares two profile distributions and returns a scalar
score (higher = more drift). Both are composable within the detector classes.
