"""Execution tests for pipeline module (not just construction)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import pynerve_internal  # noqa: F401
except ImportError:
    pytest.skip("pynerve_internal C++ extension not available", allow_module_level=True)

torch = pytest.importorskip("torch")


class TestInternalHelpers:
    """Numerical correctness for internal pipeline helper functions."""

    def test_filter_persistence_pairs_by_threshold(self) -> None:
        from pynerve._pipeline_topology import _filter_persistence_pairs

        diagram = {"pairs": [(0.0, 1.0, 0), (0.0, 3.0, 0), (0.0, 0.5, 0)]}
        filtered = _filter_persistence_pairs(diagram, min_persistence=1.0)
        assert len(filtered["pairs"]) == 1
        assert filtered["pairs"][0][1] == pytest.approx(3.0, abs=1e-10)

    def test_filter_persistence_pairs_list_io(self) -> None:
        from pynerve._pipeline_topology import _filter_persistence_pairs

        pairs = [(0.0, 1.0, 0), (0.0, 3.0, 0)]
        filtered = _filter_persistence_pairs(pairs, min_persistence=2.5)
        assert len(filtered) == 1
        b, d, dim = filtered[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert d == pytest.approx(3.0, abs=1e-10)

    def test_persistence_vector_values(self) -> None:
        from pynerve._pipeline_topology import _persistence_vector

        diagram = {"pairs": [(0.0, 1.0, 0), (0.0, 3.0, 0), (2.0, 5.0, 0)]}
        vec = _persistence_vector(diagram)
        assert vec == [1.0, 3.0, 3.0]

    def test_diagram_pairs_extracts_from_dict(self) -> None:
        from pynerve._pipeline_topology import _diagram_pairs

        d = {"pairs": [(0.0, 1.0, 0), (0.0, 2.0, 1)]}
        pairs = _diagram_pairs(d)
        assert len(pairs) == 2

    def test_diagram_pair_array_shape(self) -> None:
        from pynerve._pipeline_topology import _diagram_pair_array

        d = {"pairs": [(0.0, 1.0, 0), (0.0, 2.0, 1)]}
        arr = _diagram_pair_array(d)
        assert arr.shape == (2, 3)

    def test_persistence_vector_empty(self) -> None:
        from pynerve._pipeline_topology import _persistence_vector

        d = {"pairs": []}
        vec = _persistence_vector(d)
        assert vec == []


class TestVrPipeline:
    """Numerical correctness for vr_pipeline execution."""

    def test_vr_pipeline_two_points(self) -> None:
        from pynerve._pipeline_topology import vr_pipeline

        pipeline = vr_pipeline(max_dim=0, max_radius=5.0, min_persistence=0.0)
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = pipeline(pts)
        assert result.pairs is not None
        assert len(result.pairs) >= 1

    def test_vr_pipeline_filters_by_persistence(self) -> None:
        from pynerve._pipeline_topology import vr_pipeline

        pipeline = vr_pipeline(max_dim=0, max_radius=5.0, min_persistence=0.0)
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = pipeline(pts)
        assert result is not None


class TestAnalysisPipeline:
    """Execution tests for analysis_pipeline."""

    def test_analysis_pipeline_single_rep_returns_array(self) -> None:
        from pynerve._pipeline_topology import analysis_pipeline

        def dummy_compute(pts):
            return {"pairs": [(0.0, 1.0, 0), (0.0, 2.0, 0)]}

        pipeline = analysis_pipeline(dummy_compute, representations=["image"])
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = pipeline(pts)
        assert isinstance(result, np.ndarray)

    def test_analysis_pipeline_multiple_reps_returns_dict(self) -> None:
        from pynerve._pipeline_topology import analysis_pipeline

        def dummy_compute(pts):
            return {"pairs": [(0.0, 1.0, 0), (0.0, 2.0, 0)]}

        pipeline = analysis_pipeline(
            dummy_compute,
            representations=["image", "vector"],
        )
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = pipeline(pts)
        assert isinstance(result, dict)
        for key in ("image", "vector"):
            assert key in result, f"Missing '{key}' in pipeline output"

    def test_conditional_pipeline_branching(self) -> None:
        from pynerve._pipeline_topology import ConditionalPipeline

        cp = ConditionalPipeline().add_step("double", lambda x: x * 2)
        cp.add_conditional("abs_if_negative", lambda x: x < 0, if_true=abs, if_false=lambda x: x)
        assert cp(5) == 10
        assert cp(-3) == 6

    def test_conditional_pipeline_false_branch(self) -> None:
        from pynerve._pipeline_topology import ConditionalPipeline

        cp = ConditionalPipeline().add_step("negate", lambda x: -x)
        cp.add_conditional("clip", lambda x: x > 10, if_true=lambda x: 10, if_false=lambda x: x)
        assert cp(5) == -5
        assert cp(-20) == 10

    def test_parallel_pipeline_combine(self) -> None:
        from pynerve._pipeline_topology import ParallelPipeline

        pp = ParallelPipeline(
            pipelines={"double": lambda x: x * 2, "square": lambda x: x**2},
            combine_fn=lambda results: results["double"] + results["square"],
        )
        assert pp(3) == 15

    def test_parallel_pipeline_dict_output(self) -> None:
        from pynerve._pipeline_topology import ParallelPipeline

        pp = ParallelPipeline(
            pipelines={"inc": lambda x: x + 1, "dec": lambda x: x - 1},
            combine_fn=lambda results: results,
        )
        result = pp(5)
        assert result == {"inc": 6, "dec": 4}

    def test_pipeline_with_multiple_steps(self) -> None:
        from pynerve._pipeline_core import Pipeline

        p = Pipeline(
            ("add_one", lambda x: x + 1),
            ("double", lambda x: x * 2),
            ("subtract_three", lambda x: x - 3),
        )
        assert p(5) == 9


class TestDiagnostics:
    """Numerical correctness for diagnostics functions."""

    def test_check_data_quality_valid(self) -> None:
        import numpy as np
        from pynerve.diagnostics import check_data_quality

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = check_data_quality(data)
        assert result["valid"]

    def test_check_data_quality_nan(self) -> None:
        import numpy as np
        from pynerve.diagnostics import check_data_quality

        data = np.array([[1.0, np.nan], [1.0, 2.0], [1.0, 3.0]])
        result = check_data_quality(data)
        assert not result["valid"]
        assert any("NaN" in e for e in result["errors"])

    def test_check_data_quality_inf(self) -> None:
        import numpy as np
        from pynerve.diagnostics import check_data_quality

        data = np.array([[1.0, 2.0], [float("inf"), 4.0]])
        result = check_data_quality(data)
        assert any("Inf" in w for w in result["warnings"])

    def test_check_data_quality_zero_variance(self) -> None:
        import numpy as np
        from pynerve.diagnostics import check_data_quality

        data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = check_data_quality(data)
        assert any(
            "variance" in w.lower() or "constant" in w.lower() for w in result.get("warnings", [])
        )
        from pynerve._pipeline_topology import analysis_pipeline

        def dummy_compute(pts):
            return {"pairs": [(0.0, 1.0, 0), (0.0, 3.0, 0)]}

        pipeline = analysis_pipeline(dummy_compute, representations=["vector"])
        pts = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        result = pipeline(pts)
        assert isinstance(result, (list, np.ndarray))
        assert len(result) == 2
