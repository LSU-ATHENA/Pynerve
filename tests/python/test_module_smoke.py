"""Smoke tests for modules that lack direct coverage.

Verifies that each module can be imported and its public API elements
are accessible. These tests do not require ``nerve_internal`` (C++ extension).
"""

from __future__ import annotations


class TestInternalModuleImports:
    def test_fast_modules_importable(self):
        from pynerve import (
            _fast_boundary,
        )

        assert _fast_boundary is not None

    def test_numba_modules_importable(self):
        from pynerve import (
            _numba_compat,
        )

        assert _numba_compat is not None

    def test_formats_modules_importable(self):
        from pynerve._formats_auto import auto_load

        assert auto_load is not None

    def test_streaming_module_importable(self):
        from pynerve._streaming_persistence import StreamingPersistence

        assert StreamingPersistence is not None

    def test_utils_modules_importable(self):
        from pynerve import typing

        assert typing.PointCloud is not None

    def test_cupy_modules_importable(self):
        from pynerve import _cupy_compat

        assert _cupy_compat is not None


class TestPublicModuleImports:
    def test_cupy_ops_importable(self):
        import pynerve

        assert pynerve.cupy_ops is not None

    def test_numba_kernels_importable(self):
        import pynerve

        assert pynerve.numba_kernels is not None

    def test_benchmark_api_accessible(self):
        import pynerve

        assert pynerve.benchmark.Timer is not None
        assert pynerve.benchmark.TimerResult is not None
        assert pynerve.benchmark.ScalabilityResult is not None


class TestSubpackageInitImports:
    def test_ssl_init_importable(self):
        import pynerve.ssl

        assert pynerve.ssl.BYOLTopology is not None

    def test_regularization_init_importable(self):
        import pynerve.regularization

        assert pynerve.regularization.AdaptivePersistentDropout is not None

    def test_nn_init_importable(self):
        import pynerve.nn

        assert pynerve.nn.PersistentHomology is not None
        assert pynerve.nn.SparsePH is not None

    def test_training_init_importable(self):
        import pynerve.training

        assert pynerve.training.BettiCurriculum is not None

    def test_mapper_init_importable(self):
        import pynerve.mapper

        assert pynerve.mapper.AdaptiveCover is not None

    def test_diff_init_importable(self):
        import pynerve.diff

        assert pynerve.diff.DifferentiableVietorisRips is not None


class TestFallbackClassesSmoke:
    def test_persistence_options_defaults(self):
        from pynerve._fallback_classes import PersistenceOptions

        opts = PersistenceOptions()
        assert opts.max_dim == 2
        assert opts.max_radius is None
        assert opts.threads == 0
        assert opts.error_tolerance == 0.0

    def test_ph5ph6_config_defaults(self):
        from pynerve._fallback_classes import PH5PH6Config

        cfg = PH5PH6Config()
        assert cfg.numerical_tolerance == 1e-9
        assert cfg.max_iterations == 1000

    def test_ph5ph6_metrics_defaults(self):
        from pynerve._fallback_classes import PH5PH6Metrics

        m = PH5PH6Metrics()
        assert m.computation_time_ms == 0.0
        assert m.passed_stability_checks is False


class TypeAliasSmoke:
    def test_type_aliases_accessible(self):
        import pynerve

        assert pynerve.PointCloud is not None
        assert pynerve.PersistenceDiagramLike is not None
        assert pynerve.PersistencePair is not None
        assert pynerve.DistanceMatrix is not None
        assert pynerve.DistanceMetric is not None
        assert pynerve.VectorizationMethod is not None
        assert pynerve.FilterFunction is not None
        assert pynerve.ClusteringAlgorithm is not None

    def test_diagram_types_accessible(self):
        import pynerve

        assert pynerve.Diagram is not None
        assert pynerve.DiagramLike is not None
