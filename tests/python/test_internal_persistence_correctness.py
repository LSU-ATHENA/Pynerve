"""Direct numerical tests for internal persistence modules.

Covers Python fallback persistence, backend dispatch, and helpers
that are only exercised indirectly through the public API.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestDiagramFromDistanceMatrixPython:
    """Pure-Python persistence via union-find: _diagram_from_distance_matrix."""

    def test_two_points_h0(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_distance_matrix_python

        D = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
        pd = _diagram_from_distance_matrix_python(D, max_dim=0, single_input=True)
        deaths = pd.deaths()
        births = pd.births()
        finite = deaths[torch.isfinite(deaths)]
        assert finite[0].item() == pytest.approx(1.0, abs=1e-10)
        assert births[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_three_points_mst(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_distance_matrix_python

        D = torch.tensor(
            [[[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]],
            dtype=torch.float64,
        )
        pd = _diagram_from_distance_matrix_python(D, max_dim=0, single_input=True)
        deaths = pd.deaths()
        finite = deaths[torch.isfinite(deaths)].sort().values
        # MST edges: (0,1) at 1.0, (1,2) at 2.0
        assert finite[0].item() == pytest.approx(1.0, abs=1e-10)
        assert finite[1].item() == pytest.approx(2.0, abs=1e-10)

    def test_single_point(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_distance_matrix_python

        D = torch.tensor([[[0.0]]], dtype=torch.float64)
        pd = _diagram_from_distance_matrix_python(D, max_dim=0, single_input=True)
        deaths = pd.deaths()
        births = pd.births()
        assert not torch.isfinite(deaths[0])
        assert births[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_batch_input(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_distance_matrix_python

        D = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 2.0], [2.0, 0.0]]],
            dtype=torch.float64,
        )
        pd = _diagram_from_distance_matrix_python(D, max_dim=0, single_input=False)
        assert pd.batch_size == 2
        d0 = pd.diagrams[0, :, 1]
        d0_finite = d0[torch.isfinite(d0)]
        assert d0_finite[0].item() == pytest.approx(1.0, abs=1e-10)
        d1 = pd.diagrams[1, :, 1]
        d1_finite = d1[torch.isfinite(d1)]
        assert d1_finite[0].item() == pytest.approx(2.0, abs=1e-10)


class TestBuildSortedEdges:
    """Edge sorting from distance matrices."""

    def test_sorted_order(self) -> None:
        from pynerve.torch._persistence_helpers import _build_sorted_edges

        D = torch.tensor([[0.0, 3.0, 1.0], [3.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
        edges = _build_sorted_edges(D, 3)
        assert edges[0] == (1.0, 0, 2)
        assert edges[1] == (2.0, 1, 2)
        assert edges[2] == (3.0, 0, 1)

    def test_inf_edge_skipped(self) -> None:
        from pynerve.torch._persistence_helpers import _build_sorted_edges

        D = torch.tensor([[0.0, float("inf")], [float("inf"), 0.0]])
        edges = _build_sorted_edges(D, 2)
        assert len(edges) == 0


class TestDiagramFromBackendParts:
    """Padding and stacking of backend result parts."""

    def test_single_batch_passthrough(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_backend_parts

        d = torch.zeros((5, 3))
        m = torch.zeros((5,), dtype=torch.bool)
        n = torch.zeros((2,), dtype=torch.long)
        pd = _diagram_from_backend_parts([d], [m], [n], batched=False)
        assert pd.batch_size == 1

    def test_multi_batch_padding(self) -> None:
        from pynerve.torch._persistence_helpers import _diagram_from_backend_parts

        d1 = torch.zeros((3, 3))
        d2 = torch.zeros((5, 3))
        m1 = torch.zeros((3,), dtype=torch.bool)
        m2 = torch.zeros((5,), dtype=torch.bool)
        n1 = torch.zeros((2,), dtype=torch.long)
        n2 = torch.zeros((2,), dtype=torch.long)
        pd = _diagram_from_backend_parts([d1, d2], [m1, m2], [n1, n2], batched=True)
        assert pd.diagrams.shape[0] == 2
        assert pd.diagrams.shape[1] == 5


class TestBackendContext:
    """BackendContext forces Python fallback for testing."""

    def test_python_context_blocks_torch_backend(self) -> None:
        from pynerve.torch._backend import BackendContext, backend

        backend._initialize()
        with BackendContext("python"):
            assert backend._torch_c is None
            assert backend._core_c is None
        assert backend._torch_c is not None or backend._core_c is not None

    def test_torch_c_context_blocks_core(self) -> None:
        from pynerve.torch._backend import BackendContext, backend

        with BackendContext("torch_c"):
            assert backend._core_c is None

    def test_get_backend_info_returns_dict(self) -> None:
        from pynerve.torch._backend import get_backend_info

        info = get_backend_info()
        assert "torch_c_available" in info
        assert "core_c_available" in info
        assert "python_impl" in info


class TestPythonBackend:
    """PythonBackend produces correct persistence values."""

    def test_two_points_distance_matrix(self) -> None:
        from pynerve.torch._persistence_core_impl import PythonBackend

        backend = PythonBackend()
        D = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
        diagrams, mask, num_pairs = backend.compute_distance_matrix(D, max_dim=0)
        deaths = diagrams[0, :, 1]
        finite = deaths[torch.isfinite(deaths)]
        assert finite[0].item() == pytest.approx(1.0, abs=1e-10)

    def test_batched_distance_matrix(self) -> None:
        from pynerve.torch._persistence_core_impl import PythonBackend

        backend = PythonBackend()
        D = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 3.0], [3.0, 0.0]]],
            dtype=torch.float64,
        )
        diagrams, mask, num_pairs = backend.compute_distance_matrix(D, max_dim=0)
        d0 = diagrams[0, :, 1]
        d0_finite = d0[torch.isfinite(d0)]
        assert d0_finite[0].item() == pytest.approx(1.0, abs=1e-10)
        d1 = diagrams[1, :, 1]
        d1_finite = d1[torch.isfinite(d1)]
        assert d1_finite[0].item() == pytest.approx(3.0, abs=1e-10)

    def test_single_point_input(self) -> None:
        from pynerve.torch._persistence_core_impl import PythonBackend

        backend = PythonBackend()
        pts = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
        diagrams, mask, num_pairs = backend.compute_vr(
            pts, max_dim=0, max_radius=5.0, metric="euclidean"
        )
        deaths = diagrams[0, :, 1]
        assert not torch.isfinite(deaths[0])
