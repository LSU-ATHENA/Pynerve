"""Tests for pynerve/__init__.py — package init, __getattr__, __version__, cublas config."""

from __future__ import annotations

import pytest

import pynerve


class TestPackageInit:
    def test_version_string(self):
        assert pynerve.__version__
        assert isinstance(pynerve.__version__, str)

    def test_public_modules_accessible(self):
        modules = ["cache", "exceptions", "formats", "torch", "merge"]
        for mod in modules:
            assert hasattr(pynerve, mod)

    def test_error_classes_exported(self):
        errors = [
            "NerveError", "ValidationError", "ShapeError", "DtypeError",
            "DeviceError", "BackendRequiredError", "PersistenceError",
        ]
        for err in errors:
            assert hasattr(pynerve, err)

    def test_compute_api_exported(self):
        funcs = [
            "compute_persistence", "compute_persistence_ph0",
            "update_persistence", "persistence_image",
        ]
        for func in funcs:
            assert hasattr(pynerve, func)

    def test_error_codes_exported(self):
        assert hasattr(pynerve, "SUCCESS")
        assert hasattr(pynerve, "UNKNOWN")
        assert hasattr(pynerve, "ErrorCategory")
        assert hasattr(pynerve, "ErrorSeverity")

    def test_types_exported(self):
        types_ = [
            "PersistenceDiagramLike", "PersistenceComputer",
            "FilterFunction", "ClusteringAlgorithm",
            "DistanceMetric", "VectorizationMethod",
            "PointCloud", "DistanceMatrix", "PersistencePair",
        ]
        for t in types_:
            assert hasattr(pynerve, t)

    def test_fallback_classes_exported(self):
        classes = [
            "PersistenceMode", "PersistenceBackend",
            "PersistenceEngine", "PersistenceOptions",
            "EventType",
        ]
        for c in classes:
            assert hasattr(pynerve, c)

    def test_diagram_types_exported(self):
        assert hasattr(pynerve, "Diagram")
        assert hasattr(pynerve, "DiagramLike")

    def test_translate_cpp_exception_exported(self):
        assert hasattr(pynerve, "translate_cpp_exception")


class TestGetAttr:
    def test_valid_module(self):
        mod = pynerve.cache
        assert mod is not None

    def test_nonexistent_module_raises(self):
        with pytest.raises(AttributeError):
            _ = pynerve.nonexistent_module_xyz


class TestCublasConfig:
    def test_cublas_config_callable(self):
        from pynerve import _ensure_cublas_config
        _ensure_cublas_config()  # should not raise


class TestDir:
    def test_dir_contains_public(self):
        d = dir(pynerve)
        assert "compute_persistence" in d
        assert "NerveError" in d
        assert "__version__" in d
