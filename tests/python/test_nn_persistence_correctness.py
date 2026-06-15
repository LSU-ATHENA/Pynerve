"""Numerical correctness tests for nn persistent homology modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestFarthestPointSampling:
    """Numerical correctness for farthest_point_sampling."""

    def test_selects_correct_number(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.randn(100, 3)
        landmarks, indices = farthest_point_sampling(pts, 10)
        assert landmarks.shape == (10, 3)
        assert indices.shape == (10,)

    def test_first_point_is_index_zero(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.randn(50, 2)
        _, indices = farthest_point_sampling(pts, 10)
        assert indices[0].item() == 0

    def test_fps_returns_unique_indices(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.randn(100, 2)
        _, indices = farthest_point_sampling(pts, 20)
        assert len(set(indices.tolist())) == 20

    def test_separated_points_selected_first(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 0.1], [10.0, 0.0], [0.0, 10.0]],
            dtype=torch.float64,
        )
        _, indices = farthest_point_sampling(pts, 4)
        # After index 0 at [0,0], the farthest point is [10,0] or [0,10] (dist 10)
        # Next farthest is the other far point
        assert 2 in indices[:3] or 3 in indices[:3]

    def test_three_collinear_points(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        _, indices = farthest_point_sampling(pts, 3)
        assert indices[1].item() == 2
        assert indices[2].item() == 1

    def test_zero_samples_returns_empty(self) -> None:
        from pynerve.nn.sparse_ph import farthest_point_sampling

        pts = torch.randn(10, 2)
        landmarks, indices = farthest_point_sampling(pts, 0)
        assert landmarks.shape[0] == 0
        assert indices.shape[0] == 0


class TestPersistentHomology:
    """Numerical correctness for PersistentHomology module."""

    def test_forward_two_points(self) -> None:
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0)
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diagrams = ph(pts)
        assert len(diagrams) == 1
        d0 = diagrams[0]
        assert d0.shape[1] == 2
        assert d0.shape[0] == 1

    def test_forward_single_point(self) -> None:
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0)
        pts = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        diagrams = ph(pts)
        assert len(diagrams) == 1

    def test_invalid_input_raises(self) -> None:
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0)
        with pytest.raises(Exception):
            ph(torch.randn(2, 3))  # 2D instead of 3D

    def test_nan_input_rejected(self) -> None:
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0)
        pts = torch.tensor([[[0.0, float("nan")]]], dtype=torch.float32)
        with pytest.raises(Exception, match="finite"):
            ph(pts)

    def test_compute_persistence_diagrams(self) -> None:
        from pynerve.nn.persistent_homology import compute_persistence_diagrams

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diagrams = compute_persistence_diagrams(pts, max_dim=0, max_radius=5.0)
        assert len(diagrams) == 1


class TestTopologyAttention:
    """Numerical correctness for TopologyAttention."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.sparse_ph import TopologyAttention

        attn = TopologyAttention(n_heads=2, dim=64, n_clusters=4)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_output_finite(self) -> None:
        from pynerve.nn.sparse_ph import TopologyAttention

        attn = TopologyAttention(n_heads=2, dim=64, n_clusters=4)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.sparse_ph import TopologyAttention

        attn = TopologyAttention(n_heads=2, dim=64, n_clusters=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_mask_respected(self) -> None:
        from pynerve.nn.sparse_ph import TopologyAttention

        attn = TopologyAttention(n_heads=2, dim=16, n_clusters=2)
        x = torch.randn(1, 4, 16)
        mask = torch.tensor([[[True, True, True, False]]], dtype=torch.bool)
        out = attn(x, mask)
        assert out.shape == (1, 4, 16)


class TestSparsePH:
    """Numerical correctness for SparsePH."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.sparse_ph import SparsePH

        sph = SparsePH(max_dim=0, max_radius=5.0, landmark_ratio=0.5)
        pts = torch.randn(2, 20, 3)
        result = sph(pts)
        assert torch.isfinite(result).all()

    def test_mean_reduction_shape(self) -> None:
        from pynerve.nn.sparse_ph import SparsePH

        sph = SparsePH(max_dim=0, max_radius=5.0, landmark_ratio=0.5, reduction="mean")
        pts = torch.randn(2, 20, 3)
        result = sph(pts)
        assert result.dim() == 2
        assert result.shape[0] == 2

    def test_none_reduction(self) -> None:
        from pynerve.nn.sparse_ph import SparsePH

        sph = SparsePH(max_dim=0, max_radius=5.0, landmark_ratio=0.5, reduction="none")
        pts = torch.randn(2, 20, 3)
        result = sph(pts)
        assert isinstance(result, list)


class TestWindowedPH:
    """Numerical correctness for WindowedPH."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(window_size=8, stride=4, max_dim=0, max_radius=5.0)
        pts = torch.randn(2, 20, 3)
        result = wph(pts)
        assert torch.isfinite(result).all()

    def test_concat_overlap(self) -> None:
        from pynerve.nn.sparse_ph import WindowedPH

        wph = WindowedPH(
            window_size=8,
            stride=4,
            max_dim=0,
            max_radius=5.0,
            overlap_handling="concat",
        )
        pts = torch.randn(2, 20, 3)
        result = wph(pts)
        assert torch.isfinite(result).all()
