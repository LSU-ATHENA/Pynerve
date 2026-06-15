"""Tests for the async facade module."""

from __future__ import annotations

import pytest


class TestAsyncApiModule:
    def test_async_api_module_importable(self):
        import pynerve.async_api

        assert pynerve.async_api is not None

    def test_async_api_exports_functions(self):
        import pynerve.async_api

        assert hasattr(pynerve.async_api, "compute_persistence_async")
        assert hasattr(pynerve.async_api, "load_diagrams_async")
        assert hasattr(pynerve.async_api, "stream_persistence")

    def test_async_api_has_gpu_and_loader(self):
        import pynerve.async_api

        assert hasattr(pynerve.async_api, "AsyncGPUTransfer")
        assert hasattr(pynerve.async_api, "AsyncDiagramLoader")

    def test_async_api_all_coverage(self):
        import pynerve.async_api

        expected = {
            "AsyncDiagramLoader",
            "AsyncGPUTransfer",
            "compute_persistence_async",
            "load_diagrams_async",
            "stream_persistence",
        }
        assert set(pynerve.async_api.__all__) == expected


class TestAsyncFacadeValidation:
    def test_compute_persistence_async_rejects_bad_points_type(self):
        import asyncio

        from pynerve._async_facade import compute_persistence_async

        with pytest.raises((TypeError, ValueError)):
            asyncio.run(compute_persistence_async("not a point cloud"))

    def test_stream_persistence_rejects_non_bool_gpu(self):
        from pynerve._async_facade import stream_persistence

        gen = stream_persistence("data", use_gpu="yes")
        with pytest.raises(TypeError, match="boolean"):
            import asyncio

            asyncio.run(gen.__anext__())
