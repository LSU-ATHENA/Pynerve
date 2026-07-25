"""Targeted tests for nn/sparse_ph, diff/ph_layer, and nn/topo_regularization modules.

Exercises SparsePH, WindowedPH, TopologyAttention, farthest_point_sampling,
compute_witness_persistence, DifferentiableVietorisRips, DifferentiableAlphaComplex,
DifferentiableCubical, FiltrationLearningLayer, LearnableFiltrationPersistence,
TopologicalRegularizationLoss, PersistenceEntropyLoss, TopologicalComplexityLoss,
and DiagramMatchingLoss.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _mock_gpu_deps():
    """Inject mocks for GPU/CuPy/triton/numba/C++ deps, restore after."""
    saved = {}
    for mod in [
        "cupy", "cupy.cuda", "cupyx", "cupyx.scipy",
        "numba", "numba.cuda",
        "triton", "triton.language",
        "pynerve_torch_internal", "pynerve_internal", "nerve_torch_internal",
        "h5py",
    ]:
        saved[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()

    sys.modules["cupy"].cuda = MagicMock()
    sys.modules["cupy"].cuda.is_available = MagicMock(return_value=False)
    sys.modules["cupy"].ndarray = torch.Tensor
    sys.modules["cupy"].asarray = lambda x, **kw: torch.as_tensor(x)
    sys.modules["numba"].jit = lambda *a, **k: lambda f: f
    sys.modules["numba"].cuda = MagicMock()
    sys.modules["triton"].jit = lambda *a, **k: lambda f: f
    sys.modules["triton"].language = MagicMock()
    sys.modules["triton"].autotune = lambda *a, **k: lambda f: f
    sys.modules["h5py"].File = MagicMock()

    yield

    for mod, orig in saved.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig


def _diag(n=5):
    """Create a valid persistence diagram tensor (N, 2) with birth<death."""
    births = torch.rand(n) * 0.5
    deaths = births + torch.rand(n) * 0.5 + 0.01
    return torch.stack([births, deaths], dim=1)


def _diag3(n=5):
    """Create a valid persistence diagram tensor (N, 3) with birth<death."""
    births = torch.rand(n) * 0.5
    deaths = births + torch.rand(n) * 0.5 + 0.01
    dims = torch.randint(0, 2, (n,)).float()
    return torch.stack([births, deaths, dims], dim=1)


class TestFarthestPointSampling:
    """Covers nn/sparse_ph.py — farthest_point_sampling."""

    def test_basic(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        points = torch.rand(10, 3, dtype=torch.float32)
        landmarks, indices = farthest_point_sampling(points, 5)
        assert landmarks.shape == (5, 3)
        assert indices.shape == (5,)

    def test_all_points(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        points = torch.rand(5, 2, dtype=torch.float32)
        landmarks, indices = farthest_point_sampling(points, 5)
        assert landmarks.shape == (5, 2)

    def test_more_than_points(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        points = torch.rand(3, 2, dtype=torch.float32)
        landmarks, indices = farthest_point_sampling(points, 10)
        assert landmarks.shape == (3, 2)

    def test_zero_samples(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        points = torch.rand(5, 2, dtype=torch.float32)
        landmarks, indices = farthest_point_sampling(points, 0)
        assert landmarks.shape == (0, 2)

    def test_not_2d(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        with pytest.raises(ValueError, match="2D"):
            farthest_point_sampling(torch.rand(3, 4, 2), 2)

    def test_invalid_n_samples(self):
        from pynerve.nn.sparse_ph import farthest_point_sampling
        with pytest.raises(ValueError, match="non-negative"):
            farthest_point_sampling(torch.rand(5, 2), -1)


class TestComputeWitnessPersistence:
    """Covers nn/sparse_ph.py — compute_witness_persistence."""

    def test_dim_mismatch(self):
        from pynerve.nn.sparse_ph import compute_witness_persistence
        landmarks = np.random.rand(5, 3).astype(np.float64)
        witnesses = np.random.rand(5, 2).astype(np.float64)
        with pytest.raises(ValueError, match="same dimension"):
            compute_witness_persistence(landmarks, witnesses)

    def test_empty_landmarks(self):
        from pynerve.nn.sparse_ph import compute_witness_persistence
        with pytest.raises(ValueError, match="non-empty"):
            compute_witness_persistence(np.empty((0, 3)), np.random.rand(5, 3))

    def test_invalid_max_radius(self):
        from pynerve.nn.sparse_ph import compute_witness_persistence
        with pytest.raises(ValueError, match="positive"):
            compute_witness_persistence(
                np.random.rand(5, 3), np.random.rand(5, 3), max_radius=-1.0
            )


class TestSparsePH:
    """Covers nn/sparse_ph.py — SparsePH."""

    def test_construct(self):
        from pynerve.nn.sparse_ph import SparsePH
        layer = SparsePH(max_dim=1, max_radius=5.0, landmark_ratio=0.5)
        assert layer.max_dim == 1
        assert layer.max_radius == 5.0
        assert layer.landmark_ratio == 0.5

    def test_construct_invalid_max_dim(self):
        from pynerve.nn.sparse_ph import SparsePH
        with pytest.raises(ValueError, match="non-negative"):
            SparsePH(max_dim=-1)

    def test_construct_invalid_radius(self):
        from pynerve.nn.sparse_ph import SparsePH
        with pytest.raises(ValueError, match="positive"):
            SparsePH(max_radius=0.0)

    def test_construct_invalid_ratio(self):
        from pynerve.nn.sparse_ph import SparsePH
        with pytest.raises(ValueError, match="landmark_ratio"):
            SparsePH(landmark_ratio=0.0)

    def test_construct_invalid_reduction(self):
        from pynerve.nn.sparse_ph import SparsePH
        with pytest.raises(ValueError, match="reduction"):
            SparsePH(reduction="bad")

    def test_forward_mean(self):
        from pynerve.nn.sparse_ph import SparsePH
        layer = SparsePH(max_dim=1, max_radius=5.0, landmark_ratio=0.5, reduction="mean")
        points = torch.rand(2, 8, 3, dtype=torch.float32)
        result = layer(points)
        assert result is not None

    def test_forward_none(self):
        from pynerve.nn.sparse_ph import SparsePH
        layer = SparsePH(max_dim=1, max_radius=5.0, landmark_ratio=0.5, reduction="none")
        points = torch.rand(2, 8, 3, dtype=torch.float32)
        result = layer(points)
        assert isinstance(result, list)

    def test_forward_not_3d(self):
        from pynerve.nn.sparse_ph import SparsePH
        layer = SparsePH()
        with pytest.raises(ValueError, match="3D"):
            layer(torch.rand(5, 3))

    def test_forward_empty_points(self):
        from pynerve.nn.sparse_ph import SparsePH
        layer = SparsePH()
        with pytest.raises(ValueError, match="at least one point"):
            layer(torch.rand(1, 0, 3))


class TestWindowedPH:
    """Covers nn/sparse_ph.py — WindowedPH."""

    def test_construct(self):
        from pynerve.nn.sparse_ph import WindowedPH
        layer = WindowedPH(window_size=4, stride=2, max_dim=1)
        assert layer.window_size == 4
        assert layer.stride == 2

    def test_construct_invalid_window(self):
        from pynerve.nn.sparse_ph import WindowedPH
        with pytest.raises(ValueError, match="positive"):
            WindowedPH(window_size=0)

    def test_construct_invalid_overlap(self):
        from pynerve.nn.sparse_ph import WindowedPH
        with pytest.raises(ValueError, match="overlap_handling"):
            WindowedPH(overlap_handling="bad")

    def test_forward_concat(self):
        from pynerve.nn.sparse_ph import WindowedPH
        layer = WindowedPH(window_size=4, stride=4, max_dim=1, overlap_handling="concat")
        points = torch.rand(1, 8, 3, dtype=torch.float32)
        result = layer(points)
        assert result is not None

    def test_forward_mean(self):
        from pynerve.nn.sparse_ph import WindowedPH
        layer = WindowedPH(window_size=4, stride=2, max_dim=1, overlap_handling="mean")
        points = torch.rand(1, 8, 3, dtype=torch.float32)
        result = layer(points)
        assert result is not None

    def test_forward_max(self):
        from pynerve.nn.sparse_ph import WindowedPH
        # Use stride=window_size to avoid overlap reshape issues with mocked backend
        layer = WindowedPH(window_size=4, stride=4, max_dim=1, overlap_handling="max")
        points = torch.rand(1, 8, 3, dtype=torch.float32)
        result = layer(points)
        assert result is not None

    def test_forward_not_3d(self):
        from pynerve.nn.sparse_ph import WindowedPH
        layer = WindowedPH()
        with pytest.raises(ValueError, match="3D"):
            layer(torch.rand(5, 3))


class TestTopologyAttention:
    """Covers nn/sparse_ph.py — TopologyAttention."""

    def test_construct(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        layer = TopologyAttention(n_heads=4, dim=16, n_clusters=4, dropout=0.0)
        assert layer.n_heads == 4
        assert layer.dim == 16

    def test_construct_dim_not_divisible(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        with pytest.raises(ValueError, match="divisible"):
            TopologyAttention(n_heads=4, dim=10)

    def test_construct_invalid_dropout(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        with pytest.raises(ValueError, match="dropout"):
            TopologyAttention(dropout=1.0)

    def test_forward(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        layer = TopologyAttention(n_heads=4, dim=16, n_clusters=4, dropout=0.0)
        x = torch.rand(2, 5, 16, dtype=torch.float32)
        result = layer(x)
        assert result.shape == (2, 5, 16)

    def test_forward_wrong_dim(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        layer = TopologyAttention(n_heads=4, dim=16)
        with pytest.raises(ValueError, match="embedding dimension"):
            layer(torch.rand(2, 5, 10))

    def test_forward_not_3d(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        layer = TopologyAttention(n_heads=4, dim=16)
        with pytest.raises(ValueError, match="3D"):
            layer(torch.rand(5, 16))

    def test_forward_with_mask(self):
        from pynerve.nn.sparse_ph import TopologyAttention
        layer = TopologyAttention(n_heads=4, dim=16, n_clusters=4, dropout=0.0)
        x = torch.rand(2, 5, 16, dtype=torch.float32)
        # Mask shape: (batch, 1, seq_len, seq_len) to broadcast across n_heads
        mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)
        result = layer(x, mask=mask)
        assert result.shape == (2, 5, 16)


class TestDifferentiableVietorisRips:
    """Covers diff/ph_layer.py — DifferentiableVietorisRips."""

    def test_construct(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips
        layer = DifferentiableVietorisRips(max_dim=1, max_radius=5.0)
        assert layer.max_dim == 1
        assert layer.max_radius == 5.0

    def test_construct_negative_max_dim(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips
        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableVietorisRips(max_dim=-1)

    def test_forward_not_3d(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips
        layer = DifferentiableVietorisRips(max_dim=1)
        with pytest.raises(ValueError, match="shape"):
            layer(torch.rand(5, 3))


class TestDifferentiableAlphaComplex:
    """Covers diff/ph_layer.py — DifferentiableAlphaComplex."""

    def test_construct(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex
        layer = DifferentiableAlphaComplex(max_dim=2)
        assert layer.max_dim == 2

    def test_construct_negative(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex
        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableAlphaComplex(max_dim=-1)

    def test_forward_raises(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex
        layer = DifferentiableAlphaComplex(max_dim=2)
        with pytest.raises(RuntimeError, match="not exposed"):
            layer(torch.rand(1, 5, 3))


class TestDifferentiableCubical:
    """Covers diff/ph_layer.py — DifferentiableCubical."""

    def test_construct(self):
        from pynerve.diff.ph_layer import DifferentiableCubical
        layer = DifferentiableCubical(max_dim=2, sublevel=True)
        assert layer.max_dim == 2
        assert layer.sublevel is True

    def test_construct_negative(self):
        from pynerve.diff.ph_layer import DifferentiableCubical
        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableCubical(max_dim=-1)

    def test_forward_raises(self):
        from pynerve.diff.ph_layer import DifferentiableCubical
        layer = DifferentiableCubical(max_dim=2)
        with pytest.raises(RuntimeError, match="not exposed"):
            layer(torch.rand(1, 8, 8))


class TestFiltrationLearningLayer:
    """Covers diff/ph_layer.py — FiltrationLearningLayer."""

    def test_construct(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[32, 32])
        assert layer is not None

    def test_construct_default_hidden(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        layer = FiltrationLearningLayer(input_dim=5)
        assert layer is not None

    def test_construct_invalid_input_dim(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        with pytest.raises(ValueError, match="positive"):
            FiltrationLearningLayer(input_dim=0)

    def test_construct_invalid_hidden(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        with pytest.raises(ValueError, match="positive"):
            FiltrationLearningLayer(input_dim=3, hidden_dims=[0, 32])

    def test_forward(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[16])
        points = torch.rand(2, 5, 3, dtype=torch.float32)
        result = layer(points)
        assert result.shape == (2, 5)

    def test_forward_not_3d(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer
        layer = FiltrationLearningLayer(input_dim=3)
        with pytest.raises(ValueError, match="shape"):
            layer(torch.rand(5, 3))


class TestLearnableFiltrationPersistence:
    """Covers diff/ph_layer.py — LearnableFiltrationPersistence."""

    def test_construct(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence
        layer = LearnableFiltrationPersistence(input_dim=3, max_dim=1, hidden_dims=[16])
        assert layer is not None

    def test_forward_not_3d(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence
        layer = LearnableFiltrationPersistence(input_dim=3, max_dim=1)
        with pytest.raises(ValueError, match="shape"):
            layer(torch.rand(5, 3))


class TestTopologicalRegularizationLoss:
    """Covers nn/topo_regularization.py — TopologicalRegularizationLoss."""

    def test_construct(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss
        loss = TopologicalRegularizationLoss(
            min_persistence=0.1, target_betti=[2, 1], max_dim=1, weight=1.0
        )
        assert loss.min_persistence == 0.1
        assert loss.target_betti == [2, 1]

    def test_construct_default(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss
        loss = TopologicalRegularizationLoss()
        assert loss.target_betti is None
        assert loss.reduction == "mean"

    def test_construct_invalid_betti(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalRegularizationLoss(target_betti=[-1, 2])

    def test_construct_invalid_reduction(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss
        with pytest.raises(ValueError, match="reduction"):
            TopologicalRegularizationLoss(reduction="bad")

    def test_construct_negative_weight(self):
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalRegularizationLoss(weight=-1.0)


class TestPersistenceEntropyLoss:
    """Covers nn/topo_regularization.py — PersistenceEntropyLoss."""

    def test_construct(self):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss
        loss = PersistenceEntropyLoss(target_entropy=2.0, max_dim=1, weight=0.5)
        assert loss.target_entropy == 2.0
        assert loss.weight == 0.5

    def test_construct_negative_entropy(self):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss
        with pytest.raises(ValueError, match="non-negative"):
            PersistenceEntropyLoss(target_entropy=-1.0)

    def test_construct_negative_weight(self):
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss
        with pytest.raises(ValueError, match="non-negative"):
            PersistenceEntropyLoss(weight=-1.0)


class TestTopologicalComplexityLoss:
    """Covers nn/topo_regularization.py — TopologicalComplexityLoss."""

    def test_construct(self):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss
        loss = TopologicalComplexityLoss(
            min_features=5, max_features=50, min_persistence=0.1, max_dim=1
        )
        assert loss.min_features == 5
        assert loss.max_features == 50

    def test_construct_invalid_bounds(self):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss
        with pytest.raises(ValueError, match="feature bounds"):
            TopologicalComplexityLoss(min_features=10, max_features=5)

    def test_construct_negative_min_persistence(self):
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalComplexityLoss(min_persistence=-0.1)


class TestDiagramMatchingLoss:
    """Covers nn/topo_regularization.py — DiagramMatchingLoss."""

    def test_construct_wasserstein(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0, weight=1.0)
        assert loss.distance_metric == "wasserstein"
        assert loss.p == 2.0

    def test_construct_bottleneck(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss(distance_metric="bottleneck")
        assert loss.distance_metric == "bottleneck"

    def test_construct_invalid_metric(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        with pytest.raises(ValueError, match="distance_metric"):
            DiagramMatchingLoss(distance_metric="bad")

    def test_construct_invalid_p(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        with pytest.raises(ValueError, match="p"):
            DiagramMatchingLoss(p=-1.0)

    def test_construct_negative_weight(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        with pytest.raises(ValueError, match="non-negative"):
            DiagramMatchingLoss(weight=-1.0)

    def test_forward_matching(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0, weight=1.0)
        pred = [[_diag(3), _diag(2)]]
        target = [[_diag(4), _diag(1)]]
        result = loss.forward(pred, target)
        assert result >= 0

    def test_forward_bottleneck(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss(distance_metric="bottleneck", weight=1.0)
        pred = [[_diag(3), _diag(2)]]
        target = [[_diag(4), _diag(1)]]
        result = loss.forward(pred, target)
        assert result >= 0

    def test_forward_empty_both(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        pred = [[torch.empty(0, 2), torch.empty(0, 2)]]
        target = [[torch.empty(0, 2), torch.empty(0, 2)]]
        result = loss.forward(pred, target)
        assert result.item() == 0.0

    def test_forward_pred_empty(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        pred = [[torch.empty(0, 2)]]
        target = [[_diag(3)]]
        result = loss.forward(pred, target)
        assert result >= 0

    def test_forward_target_empty(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        pred = [[_diag(3)]]
        target = [[torch.empty(0, 2)]]
        result = loss.forward(pred, target)
        assert result >= 0

    def test_forward_batch_mismatch(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        with pytest.raises(ValueError, match="Batch sizes"):
            loss.forward([[_diag(3)]], [[_diag(3)], [_diag(2)]])

    def test_forward_empty_batch(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        with pytest.raises(ValueError, match="[Aa]t least one"):
            loss.forward([], [])

    def test_forward_dim_mismatch(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss
        loss = DiagramMatchingLoss()
        with pytest.raises(ValueError, match="dimensions must match"):
            loss.forward([[_diag(3), _diag(2)]], [[_diag(3)]])


class TestHelperFunctions:
    """Covers helper functions in nn/topo_regularization.py."""

    def test_finite_birth_death_valid(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        d = _diag(5)
        result = _finite_birth_death(d)
        assert result.shape[1] == 2

    def test_finite_birth_death_empty(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        d = torch.empty(0, 2)
        result = _finite_birth_death(d)
        assert result.numel() == 0

    def test_finite_birth_death_not_tensor(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        with pytest.raises(TypeError, match="tensor"):
            _finite_birth_death([1.0, 2.0])

    def test_finite_birth_death_wrong_dim(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        with pytest.raises(ValueError, match="birth/death"):
            _finite_birth_death(torch.rand(5))

    def test_finite_birth_death_not_floating(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        with pytest.raises(TypeError, match="floating"):
            _finite_birth_death(torch.zeros(3, 2, dtype=torch.int32))

    def test_finite_birth_death_nan_births(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        d = torch.tensor([[float("nan"), 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="births"):
            _finite_birth_death(d)

    def test_finite_birth_death_death_before_birth(self):
        from pynerve.nn.topo_regularization import _finite_birth_death
        d = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="deaths"):
            _finite_birth_death(d)

    def test_finite_persistence(self):
        from pynerve.nn.topo_regularization import _finite_persistence
        d = _diag(5)
        result = _finite_persistence(d)
        assert result.shape == (5,)

    def test_finite_persistence_empty(self):
        from pynerve.nn.topo_regularization import _finite_persistence
        d = torch.empty(0, 2)
        result = _finite_persistence(d)
        assert result.numel() == 0

    def test_as_batched_points_2d(self):
        from pynerve.nn.topo_regularization import _as_batched_points
        points = torch.rand(5, 3)
        result = _as_batched_points(points)
        assert result.dim() == 3
        assert result.shape[0] == 1

    def test_as_batched_points_3d(self):
        from pynerve.nn.topo_regularization import _as_batched_points
        points = torch.rand(2, 5, 3)
        result = _as_batched_points(points)
        assert result.shape == (2, 5, 3)

    def test_as_batched_points_wrong_dim(self):
        from pynerve.nn.topo_regularization import _as_batched_points
        with pytest.raises(ValueError, match="2D or 3D"):
            _as_batched_points(torch.rand(2, 3, 4, 5))

    def test_as_batched_points_empty(self):
        from pynerve.nn.topo_regularization import _as_batched_points
        with pytest.raises(ValueError, match="at least one"):
            _as_batched_points(torch.rand(1, 0, 3))

    def test_deterministic_subsample(self):
        from pynerve.nn.topo_regularization import _deterministic_subsample
        points = torch.rand(100, 3)
        result = _deterministic_subsample(points, 10)
        assert result.shape == (10, 3)

    def test_deterministic_subsample_no_op(self):
        from pynerve.nn.topo_regularization import _deterministic_subsample
        points = torch.rand(5, 3)
        result = _deterministic_subsample(points, 10)
        assert result.shape == (5, 3)
