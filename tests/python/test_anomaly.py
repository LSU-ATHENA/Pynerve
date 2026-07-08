"""Unit tests for the C++ backed anomaly detection classes in nerve_extras.anomaly.

Covers MarketAnomalyDetector, OnlinePValueCalculator, RegimeChangeDetector,
and AnomalyDetectionManager.
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

# Add the build directory so nerve_extras can be imported
_build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.isdir(_build_path) and _build_path not in sys.path:
    sys.path.insert(0, _build_path)

try:
    import nerve_extras
except ImportError:
    pytest.skip("nerve_extras C++ module not available", allow_module_level=True)


# MarketAnomalyDetector


def _make_market_config(
    price_thresh: float = 2.0,
    volume_thresh: float = 2.0,
    topo_thresh: float = 2.0,
    lookback: int = 10,
) -> nerve_extras.anomaly.MarketConfig:
    cfg = nerve_extras.anomaly.MarketConfig()
    cfg.price_change_threshold = price_thresh
    cfg.volume_spike_threshold = volume_thresh
    cfg.topology_anomaly_threshold = topo_thresh
    cfg.lookback_window = lookback
    cfg.enable_price_analysis = True
    cfg.enable_volume_analysis = True
    cfg.enable_topology_analysis = True
    return cfg


@pytest.mark.nerve_extras
class TestMarketAnomalyDetector:
    """Tests for MarketAnomalyDetector."""

    def test_constructor(self) -> None:
        cfg = _make_market_config()
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        assert detector is not None

    @pytest.mark.parametrize("field", [
        "price_change_threshold",
        "volume_spike_threshold",
        "topology_anomaly_threshold",
        "lookback_window",
    ])
    def test_config_fields(self, field: str) -> None:
        cfg = _make_market_config()
        assert hasattr(cfg, field)

    def test_price_anomaly_detection(self) -> None:
        """A large price deviation should be flagged."""
        rng = random.Random(42)
        history = [10.0 + rng.gauss(0, 0.1) for _ in range(20)]
        current_price = 15.0  # 5 standard deviations from mean

        cfg = _make_market_config(price_thresh=2.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_price_anomaly(current_price, history)

        assert event.anomaly_score > 2.0
        assert event.anomaly_type == "price"
        assert event.p_value < 0.05

    def test_price_normal_no_anomaly(self) -> None:
        """A small price deviation should not be flagged."""
        rng = random.Random(43)
        history = [10.0 + rng.gauss(0, 0.1) for _ in range(20)]
        current_price = 10.05  # within noise

        cfg = _make_market_config(price_thresh=3.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_price_anomaly(current_price, history)

        assert event.anomaly_score < 3.0
        assert not event.is_critical

    def test_volume_anomaly_detection(self) -> None:
        """A large volume spike should be flagged."""
        rng = random.Random(44)
        history = [float(rng.gauss(100, 10)) for _ in range(20)]
        current_volume = 500.0  # 40 standard deviations

        cfg = _make_market_config(volume_thresh=2.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_volume_anomaly(current_volume, history)

        assert event.anomaly_type == "volume"
        assert event.anomaly_score > 2.0
        assert event.p_value < 0.05

    def test_topology_anomaly_detection(self) -> None:
        """A large deviation in topological features should be flagged.

        Note: detectTopologyAnomaly uses the detector's internal history,
        so we prime it via update_and_detect first.
        """
        cfg = _make_market_config(topo_thresh=1.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)

        # Prime internal history with normal data
        rng = random.Random(45)
        for _ in range(10):
            feats = [float(rng.gauss(0, 0.1)) for _ in range(3)]
            detector.update_and_detect(0, 10.0, 100.0, feats)

        history = [[float(rng.gauss(0, 0.1)) for _ in range(3)] for __ in range(10)]
        current = [5.0, 5.0, 5.0]  # extreme
        event = detector.detect_topology_anomaly(current, history)

        assert event.anomaly_type == "topology"
        assert event.anomaly_score > 1.0

    def test_topology_anomaly_insufficient_data(self) -> None:
        """Empty history should return score 0."""
        cfg = _make_market_config()
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_topology_anomaly([1.0, 2.0], [])

        assert event.anomaly_score == 0.0
        assert event.description == "insufficient_data"

    def test_combined_anomaly(self) -> None:
        """Combined anomaly should aggregate factor scores."""
        cfg = _make_market_config()
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_combined_anomaly(
            [1.0, 2.0, 3.0], ["price", "volume", "topology"]
        )

        assert event.anomaly_type == "combined"
        assert event.anomaly_score > 0.0
        assert "price=" in event.description
        assert "volume=" in event.description

    def test_detect_single_anomaly_integration(self) -> None:
        """detectSingleAnomaly with all features enabled."""
        cfg = _make_market_config(price_thresh=1.0, topo_thresh=1.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)

        # Prime the history first
        rng = random.Random(46)
        for _ in range(5):
            detector.update_and_detect(
                0, 10.0 + rng.gauss(0, 0.1), 100.0 + rng.gauss(0, 5), [0.0, 0.0]
            )

        event = detector.detect_single_anomaly(
            999, 20.0, 500.0, [5.0, 5.0]
        )
        assert event.timestamp_ns == 999
        assert event.anomaly_score > 0.0
        assert isinstance(event.contributing_factors, list)

    def test_update_and_detect(self) -> None:
        """updateAndDetect should maintain state and detect anomalies."""
        cfg = _make_market_config(price_thresh=1.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)

        # Normal updates should not flag
        rng = random.Random(47)
        for i in range(10):
            detected, event = detector.update_and_detect(
                i, 10.0 + rng.gauss(0, 0.1), 100.0, [0.0, 0.0]
            )
            if i < 5:
                assert not detected

    def test_reset_normal_behavior(self) -> None:
        """resetNormalBehavior clears all history."""
        cfg = _make_market_config()
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)

        detector.update_normal_behavior([10.0], [100.0], [[0.0]])
        detector.reset_normal_behavior()

        # After reset, price history is empty -> score 0
        event = detector.detect_price_anomaly(10.0, [])
        assert event.anomaly_score == 0.0

    def test_market_config_defaults(self) -> None:
        """MarketConfig should have sensible defaults."""
        cfg = nerve_extras.anomaly.MarketConfig()
        assert cfg.price_change_threshold > 0
        assert cfg.lookback_window > 0
        assert cfg.enable_price_analysis
        assert cfg.enable_volume_analysis
        assert cfg.enable_topology_analysis

    def test_anomaly_event_fields(self) -> None:
        """AnomalyEvent struct fields should be accessible."""
        cfg = _make_market_config(price_thresh=1.0)
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_price_anomaly(50.0, [1.0, 2.0, 1.5, 2.5, 1.8])

        assert hasattr(event, "timestamp_ns")
        assert hasattr(event, "anomaly_type")
        assert hasattr(event, "anomaly_score")
        assert hasattr(event, "p_value")
        assert hasattr(event, "description")
        assert hasattr(event, "contributing_factors")
        assert hasattr(event, "is_critical")

    def test_empty_price_history_returns_zero_score(self) -> None:
        """With no price history, price anomaly score should be 0."""
        cfg = _make_market_config()
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)
        event = detector.detect_price_anomaly(10.0, [])
        assert event.anomaly_score == 0.0
        assert event.description == "insufficient_history"

    def test_detect_anomalies_batch(self) -> None:
        """Batch detectAnomalies should process multiple timesteps."""
        cfg = _make_market_config(topo_thresh=10.0)  # very high, no anomalies expected
        detector = nerve_extras.anomaly.MarketAnomalyDetector(cfg)

        timestamps = list(range(10))
        prices = [10.0] * 10
        volumes = [100.0] * 10
        features = [[0.0] * 2 for _ in range(10)]

        events = detector.detect_anomalies(timestamps, prices, volumes, features)
        assert isinstance(events, list)
        assert len(events) == 0  # no anomalies with high threshold


# OnlinePValueCalculator


@pytest.mark.nerve_extras
class TestOnlinePValueCalculator:
    """Tests for OnlinePValueCalculator."""

    def test_constructor(self) -> None:
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        assert calc is not None

    def test_is_significant(self) -> None:
        cfg = nerve_extras.anomaly.PValueConfig()
        cfg.significance_level = 0.05
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        assert calc.is_significant(0.01) is True
        assert calc.is_significant(0.10) is False
        assert calc.is_significant(0.05) is False  # exactly at threshold

    def test_compute_p_value(self) -> None:
        """Extreme test statistic should yield a small p-value."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        null_dist = [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05]
        p = calc.compute_p_value(10.0, null_dist)
        assert p < 0.5  # should be somewhat small

    def test_compute_p_value_zero_dist(self) -> None:
        """Empty distribution returns p=1.0."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        p = calc.compute_p_value(1.0, [])
        assert p == 1.0

    def test_empirical_p_value(self) -> None:
        """Empirical p-value should match computePValue for same input."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        dist = [0.0, 0.5, -0.5, 1.0, -1.0]
        p1 = calc.compute_p_value(0.8, dist)
        p2 = calc.compute_empirical_p_value(0.8, dist)
        assert p1 == p2

    def test_bonferroni_correction(self) -> None:
        """Bonferroni should cap at 1.0 and multiply by n."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        pvals = [0.01, 0.05, 0.001]
        corrected = calc.bonferroni_correction(pvals)
        assert len(corrected) == 3
        # 0.01 * 3 = 0.03
        assert math.isclose(corrected[0], 0.03, rel_tol=1e-6)
        # 0.05 * 3 = 0.15
        assert math.isclose(corrected[1], 0.15, rel_tol=1e-6)
        # 0.001 * 3 = 0.003
        assert math.isclose(corrected[2], 0.003, rel_tol=1e-6)
        # All values <= 1.0
        for c in corrected:
            assert c <= 1.0

    def test_bonferroni_capping(self) -> None:
        """Bonferroni should cap at 1.0 for large p-values."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        pvals = [0.5, 0.9]
        corrected = calc.bonferroni_correction(pvals)
        assert corrected[0] == 1.0
        assert corrected[1] == 1.0

    def test_benjamini_hochberg_fdr(self) -> None:
        """BH FDR should produce sorted significant thresholds."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        pvals = [0.001, 0.01, 0.5, 0.8]
        corrected = calc.benjamini_hochberg_fdr(pvals)
        assert len(corrected) == 4
        assert corrected[0] <= 1.0
        assert corrected[1] <= 1.0
        # Monotonicity: after sorting, corrected should be non-decreasing
        for i in range(1, len(corrected)):
            assert corrected[i] >= corrected[i - 1] - 1e-10

    def test_fdr_empty_input(self) -> None:
        """Empty input to BH FDR returns empty list."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        assert calc.benjamini_hochberg_fdr([]) == []

    def test_update_null_distribution(self) -> None:
        """updateNullDistribution should add samples."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        for x in [0.1, 0.2, 0.3]:
            calc.update_null_distribution(x)
        p = calc.compute_p_value(10.0, [0.1, 0.2, 0.3])
        assert p < 1.0

    def test_update_sample_distribution(self) -> None:
        """updateSampleDistribution should add samples."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        for x in [1.0, 2.0, 3.0]:
            calc.update_sample_distribution(x)
        p = calc.compute_empirical_p_value(10.0, [1.0, 2.0, 3.0])
        assert p < 1.0

    def test_multiple_testing_significance(self) -> None:
        """multipleTestingSignificance should return bool list."""
        cfg = nerve_extras.anomaly.PValueConfig()
        cfg.significance_level = 0.05
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        result = calc.multiple_testing_significance([0.001, 0.5, 0.02])
        assert len(result) == 3
        assert all(isinstance(r, bool) for r in result)

    def test_compute_effect_size(self) -> None:
        """Cohen's d should be computed correctly."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        d = calc.compute_effect_size(5.0, 3.0, 1.0, 1.0)
        # d = (5-3) / sqrt((1+1)/2) = 2 / 1 = 2.0
        assert math.isclose(d, 2.0, rel_tol=1e-6)

    def test_compute_effect_size_zero_variance(self) -> None:
        """Effect size with zero variance returns 0."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        d = calc.compute_effect_size(1.0, 0.0, 0.0, 0.0)
        assert d == 0.0

    def test_confidence_interval(self) -> None:
        """Confidence interval should be positive for varied samples."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)

        ci = calc.compute_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ci > 0.0

    def test_confidence_interval_empty(self) -> None:
        """Empty samples returns 0."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        ci = calc.compute_confidence_interval([])
        assert ci == 0.0

    def test_multiple_testing_empty(self) -> None:
        """Empty input to multipleTestingSignificance returns empty list."""
        cfg = nerve_extras.anomaly.PValueConfig()
        calc = nerve_extras.anomaly.OnlinePValueCalculator(cfg)
        assert calc.multiple_testing_significance([]) == []

    def test_pvalue_config_defaults(self) -> None:
        """PValueConfig should have sensible defaults."""
        cfg = nerve_extras.anomaly.PValueConfig()
        assert cfg.significance_level > 0
        assert cfg.min_samples > 0
        assert cfg.fdr_rate > 0


# RegimeChangeDetector


def _make_regime_config(num_regimes: int = 3, min_duration: int = 2) -> nerve_extras.anomaly.RegimeConfig:
    cfg = nerve_extras.anomaly.RegimeConfig()
    cfg.num_regimes = num_regimes
    cfg.min_regime_duration = min_duration
    cfg.enable_hmm_detection = True
    cfg.transition_probability = 0.1
    return cfg


@pytest.mark.nerve_extras
class TestRegimeChangeDetector:
    """Tests for RegimeChangeDetector."""

    def test_constructor(self) -> None:
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        assert detector is not None

    def test_detect_regimes_two_clusters(self) -> None:
        """Two distinct feature clusters should be detected as separate regimes."""
        rng = random.Random(50)
        feats_low = [[0.1 + rng.gauss(0, 0.02), 0.2 + rng.gauss(0, 0.02)] for _ in range(10)]
        feats_high = [[10.0 + rng.gauss(0, 0.02), 20.0 + rng.gauss(0, 0.02)] for _ in range(10)]
        features = feats_low + feats_high
        timestamps = list(range(20))

        cfg = _make_regime_config(num_regimes=2, min_duration=2)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        regimes = detector.detect_regimes(features, timestamps)

        assert len(regimes) >= 1
        for r in regimes:
            assert hasattr(r, "regime_id")
            assert hasattr(r, "stability_score")
            assert hasattr(r, "description")
            assert r.duration_points >= 2

    def test_detect_regime_changes(self) -> None:
        """Regime changes should be detected between clusters."""
        rng = random.Random(51)
        feats_a = [[0.1, 0.2] for _ in range(5)]
        feats_b = [[10.0, 20.0] for _ in range(5)]
        features = feats_a + feats_b
        timestamps = list(range(10))

        cfg = _make_regime_config(num_regimes=2, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        changes = detector.detect_regime_changes(features, timestamps)

        # Should detect at least one regime change
        assert len(changes) >= 1
        for rc in changes:
            assert hasattr(rc, "from_regime_id")
            assert hasattr(rc, "to_regime_id")
            assert hasattr(rc, "change_confidence")
            assert hasattr(rc, "change_description")

    def test_detect_regime_changes_single_regime(self) -> None:
        """No changes when all features belong to one cluster."""
        features = [[0.1, 0.2] for _ in range(5)]
        timestamps = list(range(5))

        cfg = _make_regime_config(num_regimes=1, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        changes = detector.detect_regime_changes(features, timestamps)
        assert len(changes) == 0

    def test_update_and_detect(self) -> None:
        """updateAndDetect accumulates history and detects change points."""
        cfg = _make_regime_config(num_regimes=2, min_duration=3)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)

        # Feed regime 0 data, then regime 1 data
        detected_any = False
        for i in range(10):
            feat = [0.1, 0.2] if i < 5 else [10.0, 20.0]
            detected, change = detector.update_and_detect(feat, i)
            if detected:
                detected_any = True
        assert detected_any

    def test_extract_regime_features(self) -> None:
        """extractRegimeFeatures returns mean + variance for a feature window."""
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)

        window = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        features = detector.extract_regime_features(window)

        # 2 dimensions -> 4 features (2 means + 2 stds)
        assert len(features) == 4
        # Means: (1+3+5)/3 = 3.0, (2+4+6)/3 = 4.0
        assert math.isclose(features[0], 3.0, rel_tol=1e-6)
        assert math.isclose(features[1], 4.0, rel_tol=1e-6)

    def test_extract_regime_features_empty(self) -> None:
        """Empty window returns empty list."""
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        assert detector.extract_regime_features([]) == []

    def test_characterize_regime(self) -> None:
        """characterizeRegime returns a descriptive string."""
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)

        desc = detector.characterize_regime([3.0, 4.0, 0.2, 0.3])
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_characterize_regime_empty(self) -> None:
        """Empty features returns 'unknown'."""
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        assert detector.characterize_regime([]) == "unknown"

    def test_train_hmm(self) -> None:
        """trainHmm returns a model with emission/tran/initial params."""
        rng = random.Random(52)
        feats_0 = [[0.1 + rng.gauss(0, 0.05) for _ in range(2)] for __ in range(10)]
        feats_1 = [[10.0 + rng.gauss(0, 0.05) for _ in range(2)] for __ in range(10)]
        features = feats_0 + feats_1
        labels = [0] * 10 + [1] * 10

        cfg = _make_regime_config(num_regimes=2, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        model = detector.train_hmm(features, labels)

        assert len(model.emission_means) == 2
        assert len(model.emission_covariances) == 2
        assert len(model.transition_matrix) == 2
        assert len(model.initial_probabilities) == 2

        # First regime's mean should be ~0.1
        assert abs(model.emission_means[0][0] - 0.1) < 0.2
        # Second regime's mean should be ~10.0
        assert abs(model.emission_means[1][0] - 10.0) < 0.2
        # Initial probability should be 1.0 for regime 0 (first label)
        assert model.initial_probabilities[0] == 1.0
        assert model.initial_probabilities[1] == 0.0

    def test_predict_regimes_hmm(self) -> None:
        """predictRegimesHmm should assign each point to the nearest regime."""
        features = [[0.15, 0.25], [9.0, 18.0], [0.2, 0.3], [11.0, 22.0]]
        labels = [0, 1, 0, 1]

        cfg = _make_regime_config(num_regimes=2, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        model = detector.train_hmm(features, labels)

        predictions = detector.predict_regimes_hmm(model, [[0.1, 0.2], [10.0, 20.0]])
        assert len(predictions) == 2
        # Point near 0.1,0.2 should be regime 0; point near 10,20 should be regime 1
        assert predictions[0] == 0
        assert predictions[1] == 1

    def test_empty_inputs(self) -> None:
        """Empty features should return empty results."""
        cfg = _make_regime_config()
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)

        assert detector.detect_regimes([], []) == []
        assert detector.detect_regime_changes([], []) == []

        model = detector.train_hmm([], [])
        assert len(model.emission_means) == 0

    def test_regime_struct_fields(self) -> None:
        """Regime struct fields should be accessible."""
        cfg = _make_regime_config(num_regimes=1, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        regimes = detector.detect_regimes([[0.1, 0.2]], [0])

        if regimes:
            r = regimes[0]
            assert hasattr(r, "regime_id")
            assert hasattr(r, "characteristic_features")
            assert hasattr(r, "stability_score")
            assert hasattr(r, "description")

    def test_regime_change_struct_fields(self) -> None:
        """RegimeChange struct fields should be accessible."""
        cfg = _make_regime_config(num_regimes=2, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        changes = detector.detect_regime_changes(
            [[0.1, 0.2], [10.0, 20.0]], [0, 1]
        )

        if changes:
            rc = changes[0]
            assert hasattr(rc, "timestamp_ns")
            assert hasattr(rc, "from_regime_id")
            assert hasattr(rc, "to_regime_id")
            assert hasattr(rc, "change_confidence")
            assert hasattr(rc, "change_description")

    def test_hmm_model_fields(self) -> None:
        """HMMModel struct fields should be accessible."""
        cfg = _make_regime_config(num_regimes=1, min_duration=1)
        detector = nerve_extras.anomaly.RegimeChangeDetector(cfg)
        model = detector.train_hmm([[0.1, 0.2], [0.3, 0.4]], [0, 0])

        assert hasattr(model, "emission_means")
        assert hasattr(model, "emission_covariances")
        assert hasattr(model, "transition_matrix")
        assert hasattr(model, "initial_probabilities")

    def test_regime_config_defaults(self) -> None:
        """RegimeConfig should have sensible defaults."""
        cfg = nerve_extras.anomaly.RegimeConfig()
        assert cfg.num_regimes > 0
        assert cfg.min_regime_duration > 0
        assert cfg.regime_change_threshold > 0


# AnomalyDetectionManager


@pytest.mark.nerve_extras
class TestAnomalyDetectionManager:
    """Tests for AnomalyDetectionManager."""

    def test_singleton(self) -> None:
        """instance() returns the same object."""
        mgr1 = nerve_extras.anomaly.AnomalyDetectionManager.instance()
        mgr2 = nerve_extras.anomaly.AnomalyDetectionManager.instance()
        assert mgr1 is mgr2

    def test_configure_betti_detector(self) -> None:
        """setBettiDetectorConfig should create a usable BettiChangeDetector."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        cfg = nerve_extras.anomaly.ChangeConfig()
        cfg.min_window_size = 5
        cfg.max_window_size = 10
        mgr.set_betti_detector_config(cfg)

        betti = mgr.get_betti_detector()
        assert betti is not None

        # Test it works
        result = betti.detect_single_change([1.0, 2.0], [1.5, 2.5], 1000)
        assert result.change_magnitude >= 0

    def test_configure_drift_detector(self) -> None:
        """setDriftDetectorConfig should create a usable LifetimeDriftDetector."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        cfg = nerve_extras.anomaly.DriftConfig()
        cfg.drift_threshold = 0.5
        mgr.set_drift_detector_config(cfg)

        drift = mgr.get_drift_detector()
        assert drift is not None

    def test_configure_market_detector(self) -> None:
        """setMarketDetectorConfig should create a usable MarketAnomalyDetector."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        cfg = nerve_extras.anomaly.MarketConfig()
        cfg.price_change_threshold = 2.0
        mgr.set_market_detector_config(cfg)

        market = mgr.get_market_detector()
        assert market is not None

    def test_configure_pvalue_calculator(self) -> None:
        """setPvalueConfig should create a usable OnlinePValueCalculator."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        cfg = nerve_extras.anomaly.PValueConfig()
        cfg.significance_level = 0.01
        mgr.set_pvalue_config(cfg)

        pval = mgr.get_pvalue_calculator()
        assert pval is not None

    def test_configure_regime_detector(self) -> None:
        """setRegimeConfig should create a usable RegimeChangeDetector."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        cfg = nerve_extras.anomaly.RegimeConfig()
        cfg.num_regimes = 2
        mgr.set_regime_config(cfg)

        regime = mgr.get_regime_detector()
        assert regime is not None

    def test_detect_all_anomalies(self) -> None:
        """detectAllAnomalies should return a report with all detector outputs."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        # Configure all detectors
        betti_cfg = nerve_extras.anomaly.ChangeConfig()
        betti_cfg.min_window_size = 3
        betti_cfg.max_window_size = 10
        mgr.set_betti_detector_config(betti_cfg)

        market_cfg = nerve_extras.anomaly.MarketConfig()
        market_cfg.price_change_threshold = 10.0  # high, don't trigger
        mgr.set_market_detector_config(market_cfg)

        regime_cfg = nerve_extras.anomaly.RegimeConfig()
        regime_cfg.num_regimes = 2
        regime_cfg.min_regime_duration = 2
        mgr.set_regime_config(regime_cfg)

        # Generate test data
        rng = random.Random(55)
        n = 20
        timestamps = list(range(n))
        prices = [10.0 + rng.gauss(0, 0.1) for _ in range(n)]
        volumes = [1.0 + rng.gauss(0, 0.05) for _ in range(n)]
        betti_seqs = [[rng.gauss(0, 0.1) for _ in range(2)] for __ in range(n)]
        lifetime_seqs = [[rng.gauss(1.0, 0.1) for _ in range(3)] for __ in range(n)]
        topo_features = [[rng.gauss(0, 0.1) for _ in range(2)] for __ in range(n)]

        report = mgr.detect_all_anomalies(
            timestamps, prices, volumes,
            betti_seqs, lifetime_seqs, topo_features
        )

        assert hasattr(report, "betti_changes")
        assert hasattr(report, "drift_points")
        assert hasattr(report, "market_anomalies")
        assert hasattr(report, "regime_changes")
        assert hasattr(report, "overall_anomaly_score")
        assert hasattr(report, "summary_report")
        assert report.overall_anomaly_score >= 0.0
        assert isinstance(report.summary_report, str)

    def test_generate_alerts(self) -> None:
        """generateAlerts should convert a report to alert strings."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        # Create a minimal report
        report = nerve_extras.anomaly.AnomalyReport()
        report.overall_anomaly_score = 0.6

        alerts = mgr.generate_alerts(report)
        assert isinstance(alerts, list)
        # Should include a critical alert for score > 0.5
        critical_alerts = [a for a in alerts if "[CRITICAL]" in a]
        assert len(critical_alerts) >= 1

    def test_send_alerts(self) -> None:
        """sendAlerts returns True (stub)."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()
        assert mgr.send_alerts(["test alert"]) is True

    def test_unconfigured_detectors(self) -> None:
        """detectAllAnomalies works with unconfigured detectors."""
        mgr = nerve_extras.anomaly.AnomalyDetectionManager.instance()

        # Only configure market detector
        market_cfg = nerve_extras.anomaly.MarketConfig()
        market_cfg.price_change_threshold = 10.0
        mgr.set_market_detector_config(market_cfg)

        report = mgr.detect_all_anomalies(
            [0], [10.0], [1.0], [], [], [[0.0]]
        )
        assert report.overall_anomaly_score >= 0
        assert isinstance(report.summary_report, str)
