"""Tests for fallback classes (_fallback_classes.py)."""

from __future__ import annotations

import dataclasses
import math

import pytest
from pynerve._fallback_classes import (
    EventType,
    PH5PH6Config,
    PH5PH6Engine,
    PH5PH6Metrics,
    PersistenceBackend,
    PersistenceEngine,
    PersistenceMode,
    PersistenceOptions,
)
from pynerve.exceptions import ValidationError


class TestPersistenceMode:
    def test_values(self):
        assert PersistenceMode.EXACT.value == "EXACT"
        assert PersistenceMode.APPROX.value == "APPROX"

    def test_membership(self):
        assert PersistenceMode("EXACT") == PersistenceMode.EXACT
        assert PersistenceMode("APPROX") == PersistenceMode.APPROX

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            PersistenceMode("INVALID")


class TestPersistenceBackend:
    def test_values(self):
        assert PersistenceBackend.CPU_EXACT.value == "CPU_EXACT"
        assert PersistenceBackend.CPU_ADAPTIVE_ACCELERATION.value == "CPU_ADAPTIVE_ACCELERATION"
        assert PersistenceBackend.CUDA_HYBRID.value == "CUDA_HYBRID"

    def test_membership(self):
        assert PersistenceBackend("CPU_EXACT") == PersistenceBackend.CPU_EXACT
        assert PersistenceBackend("CUDA_HYBRID") == PersistenceBackend.CUDA_HYBRID


class TestPersistenceEngine:
    def test_values(self):
        assert PersistenceEngine.AUTO.value == "auto"
        assert PersistenceEngine.PH4.value == "ph4"
        assert PersistenceEngine.PH6.value == "ph6"

    def test_membership(self):
        assert PersistenceEngine("auto") == PersistenceEngine.AUTO
        assert PersistenceEngine("ph4") == PersistenceEngine.PH4
        assert PersistenceEngine("ph5") == PersistenceEngine.PH5
        assert PersistenceEngine("ph6") == PersistenceEngine.PH6


class TestEventType:
    def test_values(self):
        assert EventType.ADD.value == "add"
        assert EventType.REMOVE.value == "remove"

    def test_membership(self):
        assert EventType("add") == EventType.ADD
        assert EventType("remove") == EventType.REMOVE


class TestPersistenceOptionsDefaults:
    def test_default_construction(self):
        opts = PersistenceOptions()
        assert opts.mode == PersistenceMode.EXACT
        assert opts.backend == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION
        assert opts.max_dim == 2
        assert opts.max_radius is None
        assert opts.threads == 0
        assert opts.error_tolerance == 0.0

    def test_is_frozen(self):
        opts = PersistenceOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.max_dim = 3  # type: ignore[misc]

    def test_explicit_construction(self):
        opts = PersistenceOptions(
            mode=PersistenceMode.APPROX,
            backend=PersistenceBackend.CUDA_HYBRID,
            max_dim=3,
            max_radius=1.5,
            threads=4,
            error_tolerance=1e-6,
        )
        assert opts.mode == PersistenceMode.APPROX
        assert opts.backend == PersistenceBackend.CUDA_HYBRID
        assert opts.max_dim == 3
        assert opts.max_radius == 1.5
        assert opts.threads == 4
        assert opts.error_tolerance == 1e-6


class TestPersistenceOptionsReplace:
    def test_replace_max_dim(self):
        opts = PersistenceOptions(max_dim=2)
        new_opts = opts.replace(max_dim=4)
        assert new_opts.max_dim == 4
        assert opts.max_dim == 2
        assert new_opts is not opts

    def test_replace_max_radius(self):
        opts = PersistenceOptions(max_radius=None)
        new_opts = opts.replace(max_radius=3.0)
        assert new_opts.max_radius == 3.0
        assert opts.max_radius is None

    def test_replace_mode(self):
        opts = PersistenceOptions(mode=PersistenceMode.EXACT)
        new_opts = opts.replace(mode=PersistenceMode.APPROX)
        assert new_opts.mode == PersistenceMode.APPROX

    def test_replace_backend(self):
        opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
        new_opts = opts.replace(backend=PersistenceBackend.CUDA_HYBRID)
        assert new_opts.backend == PersistenceBackend.CUDA_HYBRID

    def test_replace_threads(self):
        opts = PersistenceOptions(threads=0)
        new_opts = opts.replace(threads=8)
        assert new_opts.threads == 8

    def test_replace_error_tolerance(self):
        opts = PersistenceOptions(error_tolerance=0.0)
        new_opts = opts.replace(error_tolerance=0.001)
        assert new_opts.error_tolerance == 0.001

    def test_replace_multiple_fields(self):
        opts = PersistenceOptions(max_dim=2, max_radius=None)
        new_opts = opts.replace(max_dim=5, max_radius=10.0)
        assert new_opts.max_dim == 5
        assert new_opts.max_radius == 10.0

    def test_replace_unknown_kwarg_raises(self):
        opts = PersistenceOptions()
        with pytest.raises(TypeError):
            opts.replace(nonexistent=42)  # type: ignore[call-arg]


class TestPersistenceOptionsValidation:
    def test_max_dim_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            PersistenceOptions(max_dim=-1)

    def test_max_dim_bool_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(max_dim=True)  # type: ignore[arg-type]

    def test_max_dim_float_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(max_dim=3.14)  # type: ignore[arg-type]

    def test_max_dim_none_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(max_dim=None)  # type: ignore[arg-type]

    def test_max_radius_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            PersistenceOptions(max_radius=float("nan"))

    def test_max_radius_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            PersistenceOptions(max_radius=-1.0)

    def test_max_radius_inf_allowed(self):
        opts = PersistenceOptions(max_radius=float("inf"))
        assert math.isinf(opts.max_radius)  # type: ignore[arg-type]

    def test_max_radius_string_raises(self):
        with pytest.raises(ValidationError, match="number"):
            PersistenceOptions(max_radius="large")  # type: ignore[arg-type]

    def test_max_radius_zero_allowed(self):
        opts = PersistenceOptions(max_radius=0.0)
        assert opts.max_radius == 0.0

    def test_threads_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            PersistenceOptions(threads=-1)

    def test_threads_bool_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(threads=False)  # type: ignore[arg-type]

    def test_threads_float_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(threads=2.5)  # type: ignore[arg-type]

    def test_threads_none_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            PersistenceOptions(threads=None)  # type: ignore[arg-type]

    def test_error_tolerance_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            PersistenceOptions(error_tolerance=float("nan"))

    def test_error_tolerance_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            PersistenceOptions(error_tolerance=-0.1)

    def test_error_tolerance_string_raises(self):
        with pytest.raises(ValidationError, match="number"):
            PersistenceOptions(error_tolerance="1e-6")  # type: ignore[arg-type]

    def test_max_dim_coerced_to_int(self):
        opts = PersistenceOptions(max_dim=3)
        assert isinstance(opts.max_dim, int)
        assert opts.max_dim == 3

    def test_threads_coerced_to_int(self):
        opts = PersistenceOptions(threads=4)
        assert isinstance(opts.threads, int)
        assert opts.threads == 4


class TestPH5PH6ConfigDefaults:
    def test_default_construction(self):
        config = PH5PH6Config()
        assert config.numerical_tolerance > 0
        assert config.max_iterations == 1000
        assert config.enable_stability_checks is True
        assert config.validate_results is True
        assert config.require_bitwise_reproducibility is False
        assert config.enable_checksum_validation is True
        assert config.computation_id == ""

    def test_explicit_construction(self):
        config = PH5PH6Config(
            numerical_tolerance=1e-8,
            max_iterations=500,
            enable_stability_checks=False,
            validate_results=False,
            require_bitwise_reproducibility=True,
            enable_checksum_validation=False,
            computation_id="test-123",
        )
        assert config.numerical_tolerance == 1e-8
        assert config.max_iterations == 500
        assert config.enable_stability_checks is False
        assert config.validate_results is False
        assert config.require_bitwise_reproducibility is True
        assert config.enable_checksum_validation is False
        assert config.computation_id == "test-123"


class TestPH5PH6ConfigRepr:
    def test_repr_default(self):
        config = PH5PH6Config()
        r = repr(config)
        assert "PH5PH6Config" in r
        assert "numerical_tolerance" in r

    def test_repr_empty_computation_id_omitted(self):
        config = PH5PH6Config(computation_id="")
        r = repr(config)
        assert "computation_id" not in r

    def test_repr_with_computation_id(self):
        config = PH5PH6Config(computation_id="abc")
        r = repr(config)
        assert "computation_id='abc'" in r

    def test_repr_falsy_bools_omitted(self):
        config = PH5PH6Config(
            enable_stability_checks=False,
            require_bitwise_reproducibility=False,
            enable_checksum_validation=False,
            validate_results=False,
        )
        r = repr(config)
        assert "enable_stability_checks" not in r
        assert "require_bitwise_reproducibility" not in r


class TestPH5PH6MetricsDefaults:
    def test_default_construction(self):
        metrics = PH5PH6Metrics()
        assert metrics.computation_time_ms == 0.0
        assert metrics.peak_memory_bytes == 0
        assert metrics.original_simplices == 0
        assert metrics.final_simplices == 0
        assert metrics.compression_ratio == 1.0
        assert metrics.quality_score == 0.0
        assert metrics.passed_stability_checks is False
        assert metrics.numerical_errors == 0
        assert metrics.checksum_validation_passed is False

    def test_explicit_construction(self):
        metrics = PH5PH6Metrics(
            computation_time_ms=123.4,
            peak_memory_bytes=8192,
            original_simplices=1000,
            final_simplices=500,
            compression_ratio=2.0,
            quality_score=0.95,
            passed_stability_checks=True,
            numerical_errors=3,
            checksum_validation_passed=True,
        )
        assert metrics.computation_time_ms == 123.4
        assert metrics.peak_memory_bytes == 8192
        assert metrics.original_simplices == 1000
        assert metrics.final_simplices == 500
        assert metrics.compression_ratio == 2.0
        assert metrics.quality_score == 0.95
        assert metrics.passed_stability_checks is True
        assert metrics.numerical_errors == 3
        assert metrics.checksum_validation_passed is True


class TestPH5PH6MetricsRepr:
    def test_repr_default(self):
        metrics = PH5PH6Metrics()
        r = repr(metrics)
        assert "PH5PH6Metrics" in r

    def test_repr_default_zero_fields_omitted(self):
        metrics = PH5PH6Metrics(computation_time_ms=0.0)
        r = repr(metrics)
        assert "computation_time_ms" not in r

    def test_repr_with_data(self):
        metrics = PH5PH6Metrics(computation_time_ms=100.0, original_simplices=500)
        r = repr(metrics)
        assert "computation_time_ms=100.0" in r
        assert "original_simplices=500" in r

    def test_repr_falsy_bools_omitted(self):
        metrics = PH5PH6Metrics(passed_stability_checks=False)
        r = repr(metrics)
        assert "passed_stability_checks" not in r


class TestPH5PH6Engine:
    def test_default_construction(self):
        engine = PH5PH6Engine()
        assert isinstance(engine.config, PH5PH6Config)
        assert engine.config.max_iterations == 1000

    def test_explicit_config(self):
        config = PH5PH6Config(max_iterations=500)
        engine = PH5PH6Engine(config)
        assert engine.config.max_iterations == 500

    def test_none_config_uses_default(self):
        engine = PH5PH6Engine(None)
        assert isinstance(engine.config, PH5PH6Config)

    def test_repr(self):
        engine = PH5PH6Engine()
        r = repr(engine)
        assert "PH5PH6Engine" in r
        assert "config=" in r

    def test_repr_with_custom_config(self):
        config = PH5PH6Config(max_iterations=500)
        engine = PH5PH6Engine(config)
        r = repr(engine)
        assert "max_iterations=500" in r
