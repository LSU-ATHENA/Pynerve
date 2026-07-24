"""Tests for diagnostics.py — DiagnosticsCollector, DiagnosticInfo, verbose."""

from __future__ import annotations

import logging
import threading
from unittest.mock import patch

import pytest
from pynerve.diagnostics import (
    DiagnosticInfo,
    DiagnosticsCollector,
    verbose,
)


class TestDiagnosticInfo:
    def test_default_construction(self):
        info = DiagnosticInfo(operation="test_op", duration=0.5)
        assert info.operation == "test_op"
        assert info.duration == 0.5
        assert info.memory_delta is None
        assert info.gpu_memory_delta is None
        assert info.n_points is None
        assert info.n_simplices is None
        assert info.backend is None
        assert info.error is None

    def test_full_construction(self):
        info = DiagnosticInfo(
            operation="compute",
            duration=2.5,
            memory_delta=10.0,
            gpu_memory_delta=100.0,
            n_points=500,
            n_simplices=1000,
            backend="cpu",
            error=None,
        )
        assert info.operation == "compute"
        assert info.duration == 2.5
        assert info.memory_delta == 10.0
        assert info.gpu_memory_delta == 100.0
        assert info.n_points == 500
        assert info.n_simplices == 1000
        assert info.backend == "cpu"
        assert info.error is None

    def test_repr_ok(self):
        info = DiagnosticInfo(operation="fast_op", duration=0.001)
        r = repr(info)
        assert "[OK]" in r
        assert "fast_op" in r

    def test_repr_error(self):
        info = DiagnosticInfo(operation="failed_op", duration=1.0, error="something broke")
        r = repr(info)
        assert "[ERROR]" in r
        assert "failed_op" in r

    def test_empty_operation_raises(self):
        with pytest.raises(Exception):
            DiagnosticInfo(operation="", duration=0.0)

    def test_negative_duration_raises(self):
        with pytest.raises(Exception):
            DiagnosticInfo(operation="x", duration=-1.0)


class TestDiagnosticsCollectorInit:
    def test_empty_on_creation(self):
        dc = DiagnosticsCollector()
        assert len(dc.diagnostics) == 0

    def test_repr_empty(self):
        dc = DiagnosticsCollector()
        r = repr(dc)
        assert "operations=0" in r
        assert "errors=0" in r


class TestDiagnosticsCollectorTrack:
    def test_track_successful_operation(self):
        dc = DiagnosticsCollector()
        with dc.track("my_op"):
            pass
        assert len(dc.diagnostics) == 1
        assert dc.diagnostics[0].operation == "my_op"
        assert dc.diagnostics[0].error is None
        assert dc.diagnostics[0].duration >= 0

    def test_track_failed_operation(self):
        dc = DiagnosticsCollector()
        with pytest.raises(ValueError, match="test error"):
            with dc.track("bad_op"):
                raise ValueError("test error")
        assert len(dc.diagnostics) == 1
        assert dc.diagnostics[0].error == "test error"

    def test_track_with_kwargs(self):
        dc = DiagnosticsCollector()
        with dc.track("big_op", n_points=1000, n_simplices=5000, backend="cuda"):
            pass
        info = dc.diagnostics[0]
        assert info.n_points == 1000
        assert info.n_simplices == 5000
        assert info.backend == "cuda"

    def test_track_updates_info_in_block(self):
        dc = DiagnosticsCollector()
        with dc.track("update_op") as info:
            info.n_points = 42
            info.backend = "gpu"
        assert dc.diagnostics[0].n_points == 42
        assert dc.diagnostics[0].backend == "gpu"

    def test_track_unexpected_kwargs_warns(self):
        dc = DiagnosticsCollector()
        with pytest.warns(UserWarning, match="Unexpected"):
            with dc.track("op", unknown_kwarg=123):
                pass

    def test_track_multiple_operations(self):
        dc = DiagnosticsCollector()
        for i in range(5):
            with dc.track(f"op_{i}"):
                pass
        assert len(dc.diagnostics) == 5

    def test_diagnostics_property_returns_copy(self):
        dc = DiagnosticsCollector()
        with dc.track("op1"):
            pass
        diags = dc.diagnostics
        diags.clear()
        assert len(dc.diagnostics) == 1


class TestDiagnosticsCollectorReport:
    def test_report_single_operation(self):
        dc = DiagnosticsCollector()
        with dc.track("single_op"):
            pass
        report = dc.report()
        assert "Diagnostic Report" in report
        assert "single_op" in report
        assert "OK" in report
        assert "Total time" in report

    def test_report_with_error(self):
        dc = DiagnosticsCollector()
        with pytest.raises(ValueError):
            with dc.track("error_op"):
                raise ValueError("bad")
        report = dc.report()
        assert "ERROR" in report
        assert "bad" in report

    def test_report_multiple_operations(self):
        dc = DiagnosticsCollector()
        with dc.track("op1"):
            pass
        with dc.track("op2"):
            pass
        report = dc.report()
        assert "op1" in report
        assert "op2" in report


class TestDiagnosticsCollectorSummary:
    def test_summary_empty(self):
        dc = DiagnosticsCollector()
        assert dc.summary() == {}

    def test_summary_single(self):
        dc = DiagnosticsCollector()
        with dc.track("op"):
            pass
        s = dc.summary()
        assert s["n_operations"] == 1
        assert s["n_errors"] == 0
        assert s["total_time"] >= 0
        assert s["error_rate"] == 0

    def test_summary_with_errors(self):
        dc = DiagnosticsCollector()
        with dc.track("good"):
            pass
        with pytest.raises(ValueError):
            with dc.track("bad"):
                raise ValueError("x")
        s = dc.summary()
        assert s["n_operations"] == 2
        assert s["n_errors"] == 1
        assert s["error_rate"] == 0.5

    def test_summary_mean_max(self):
        dc = DiagnosticsCollector()
        with dc.track("op1"):
            pass
        with dc.track("op2"):
            pass
        s = dc.summary()
        assert "mean_time" in s
        assert "max_time" in s
        assert s["mean_time"] >= 0
        assert s["max_time"] >= 0


class TestVerbose:
    def test_enabled_default(self):
        with verbose():
            pass

    def test_enabled_false(self):
        with verbose(enabled=False):
            pass

    def test_level_info(self):
        with verbose(level="info"):
            pass

    def test_level_debug(self):
        with verbose(level="debug"):
            pass

    def test_level_trace(self):
        with verbose(level="trace"):
            pass

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            with verbose(level="invalid"):
                pass

    def test_non_bool_enabled_raises(self):
        with pytest.raises(TypeError, match="boolean"):
            with verbose(enabled="yes"):
                pass

    def test_nested_verbose(self):
        with verbose(level="info"):
            with verbose(level="debug"):
                pass

    def test_logger_level_changes(self):
        logger = logging.getLogger("pynerve")
        original = logger.level
        with verbose(level="debug"):
            assert logger.level == logging.DEBUG
        assert logger.level == original
