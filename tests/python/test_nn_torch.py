"""Tests for nn/_ph_autograd.py, nn/_ph_module.py, torch/_persistence_core (_impl)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn

from pynerve._compute_core import PersistenceResult as CorePersistenceResult
from pynerve.exceptions import InvalidArgumentError, ValidationError

# -------------------------------------------------------------------- helpers


def _has_core_backend() -> bool:
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


def _has_torch_backend() -> bool:
    try:
        from pynerve.torch._persistence_validators import _torch_backend  # noqa: PLC0415

        return _torch_backend() is not None
    except ImportError:
        return False


_core_skip = pytest.mark.skipif(not _has_core_backend(), reason="pynerve_internal not available")
_torchc_skip = pytest.mark.skipif(
    not _has_torch_backend(), reason="Torch C++ backend not available"
)


def _make_point_cloud(batch=2, n_points=8, dim=3, seed=42):
    torch.manual_seed(seed)
    return torch.rand(batch, n_points, dim)


def _make_dist_matrix(n_points=4, batch=2, seed=42):
    torch.manual_seed(seed)
    pts = torch.rand(batch, n_points, 3)
    return torch.cdist(pts, pts)


# ------------------------------------------------------------------- _ph_autograd


class TestPadDiagramBatch:
    def test_all_equal_size(self):
        from pynerve.nn._ph_autograd import _pad_diagram_batch

        a = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        b = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        out = _pad_diagram_batch([a, b])
        assert out.shape == (2, 2, 2)
        assert torch.equal(out[0], a)
        assert torch.equal(out[1], b)

    def test_unequal_sizes(self):
        from pynerve.nn._ph_autograd import _pad_diagram_batch

        a = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        b = torch.tensor([[0.1, 0.2]])
        out = _pad_diagram_batch([a, b])
        assert out.shape == (2, 2, 2)
        assert out[1, 1, 0] == 0.0
        assert out[1, 1, 1] == 0.0

    def test_all_empty(self):
        from pynerve.nn._ph_autograd import _pad_diagram_batch

        a = torch.empty((0, 2))
        b = torch.empty((0, 2))
        out = _pad_diagram_batch([a, b])
        assert out.shape == (2, 0, 2)

    def test_empty_and_non_empty(self):
        from pynerve.nn._ph_autograd import _pad_diagram_batch

        a = torch.empty((0, 2))
        b = torch.tensor([[0.1, 0.2]])
        out = _pad_diagram_batch([a, b])
        assert out.shape == (2, 1, 2)
        assert out[0, 0, 0] == 0.0
        assert out[1, 0, 0] == 0.1

    def test_single_element(self):
        from pynerve.nn._ph_autograd import _pad_diagram_batch

        a = torch.tensor([[1.0, 2.0]])
        out = _pad_diagram_batch([a])
        assert out.shape == (1, 1, 2)


class TestPersistentHomologyFunction:
    @_core_skip
    def test_forward_basic(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction

        pts = _make_point_cloud(batch=1, n_points=5)
        result = PersistentHomologyFunction.apply(
            pts, 1, float("inf"), "euclidean", "standard", False, "standard"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @_core_skip
    def test_forward_raises_bad_metric(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction

        pts = _make_point_cloud(batch=1, n_points=5)
        with pytest.raises(ValueError, match="euclidean"):
            PersistentHomologyFunction.apply(
                pts, 0, 1.0, "manhattan", "standard", False, "standard"
            )

    def test_backward_no_grads(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction

        class _Ctx:
            saved_tensors = (_make_point_cloud(batch=2, n_points=4),)

        ctx = _Ctx()
        result = PersistentHomologyFunction.backward(ctx)
        assert isinstance(result, tuple)
        assert result[0] is not None
        assert result[0].shape == (2, 4, 3)

    def test_backward_zero_grads(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction

        class _Ctx:
            saved_tensors = (_make_point_cloud(batch=1, n_points=3),)

        ctx = _Ctx()
        g = torch.zeros(1, 0, 2)
        result = PersistentHomologyFunction.backward(ctx, g)
        assert result[0].shape == (1, 3, 3)

    def test_backward_produces_finite_grad(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction

        class _Ctx:
            saved_tensors = (_make_point_cloud(batch=1, n_points=3),)

        ctx = _Ctx()
        g = torch.rand(1, 2, 2, requires_grad=False) * 0.1
        result = PersistentHomologyFunction.backward(ctx, g)
        assert torch.isfinite(result[0]).all()

    @_core_skip
    def test_gradient_flow_end_to_end(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=float("inf"))
        pts = _make_point_cloud(batch=1, n_points=4).requires_grad_(True)
        diagrams = ph(pts)
        loss = diagrams[0].sum()
        loss.backward()
        assert pts.grad is not None
        assert (pts.grad.abs().sum() > 0).item()


# ------------------------------------------------------------------ _ph_module


class TestEffectiveMaxRadius:
    def test_finite_radius_returned(self):
        from pynerve.nn._ph_module import _effective_max_radius

        pts = _make_point_cloud(batch=1, n_points=3)
        assert _effective_max_radius(pts, 2.5) == 2.5

    def test_infinite_when_no_pairs(self):
        from pynerve.nn._ph_module import _effective_max_radius

        pts = torch.rand(1, 3)
        result = _effective_max_radius(pts, float("inf"))
        assert result == 0.0

    def test_infinite_computed_from_points(self):
        from pynerve.nn._ph_module import _effective_max_radius

        pts = _make_point_cloud(batch=1, n_points=4)[0]
        result = _effective_max_radius(pts, float("inf"))
        assert math.isfinite(result)
        assert result > 0


class TestResultToDiagramTensors:
    def test_all_dimensions_populated(self):
        from pynerve.nn._ph_module import _result_to_diagram_tensors

        result = CorePersistenceResult.from_dict(
            {
                "pairs": [(0.0, 1.0, 0), (0.2, 0.8, 1)],
                "betti_numbers": [1, 1],
                "max_dim": 2,
                "max_radius": 1.0,
            }
        )
        tensors = _result_to_diagram_tensors(result, 2, torch.float32, torch.device("cpu"))
        assert len(tensors) == 3
        assert tensors[0].shape == (1, 2)
        assert tensors[1].shape == (1, 2)
        assert tensors[2].shape == (0, 2)

    def test_out_of_range_dim_ignored(self):
        from pynerve.nn._ph_module import _result_to_diagram_tensors

        result = CorePersistenceResult.from_dict(
            {
                "pairs": [(0.0, 1.0, 0), (0.2, 0.8, 5)],
                "betti_numbers": [1, 0],
                "max_dim": 1,
                "max_radius": 1.0,
            }
        )
        tensors = _result_to_diagram_tensors(result, 1, torch.float64, torch.device("cpu"))
        assert len(tensors) == 2
        assert tensors[0].shape == (1, 2)
        assert tensors[1].shape == (0, 2)

    def test_preserves_dtype_and_device(self):
        from pynerve.nn._ph_module import _result_to_diagram_tensors

        result = CorePersistenceResult.from_dict(
            {"pairs": [(0.0, 0.5, 0)], "betti_numbers": [1], "max_dim": 0, "max_radius": 0.5}
        )
        tensors = _result_to_diagram_tensors(result, 0, torch.float64, torch.device("cpu"))
        assert tensors[0].dtype == torch.float64
        assert tensors[0].device == torch.device("cpu")


class TestPersistentHomology:
    @pytest.fixture(scope="class")
    def ph(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        return PersistentHomology(max_dim=1, max_radius=float("inf"))

    @_core_skip
    def test_forward_returns_list(self, ph):
        pts = _make_point_cloud()
        diagrams = ph(pts)
        assert isinstance(diagrams, list)
        assert len(diagrams) == 2

    @_core_skip
    def test_forward_tensor_shapes(self, ph):
        pts = _make_point_cloud(batch=2, n_points=8, dim=3)
        diagrams = ph(pts)
        for d in diagrams:
            assert d.dim() == 3
            assert d.shape[0] == 2
            assert d.shape[2] == 2

    @_core_skip
    def test_max_dim_zero(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph0 = PersistentHomology(max_dim=0, max_radius=float("inf"))
        diagrams = ph0(_make_point_cloud())
        assert len(diagrams) == 1

    @_core_skip
    def test_cohomology_reduction(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=1, reduction="cohomology")
        diagrams = ph(_make_point_cloud())
        assert len(diagrams) == 2

    @_core_skip
    def test_finite_max_radius(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=1.0)
        diagrams = ph(_make_point_cloud())
        assert len(diagrams) == 1

    @_core_skip
    def test_clearing_reduction(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0, reduction="clearing")
        diagrams = ph(_make_point_cloud())
        assert len(diagrams) == 1

    @_core_skip
    def test_standard_reduction(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=5.0, reduction="standard")
        diagrams = ph(_make_point_cloud())
        assert len(diagrams) == 1

    @_core_skip
    def test_gradient_flows(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=float("inf"))
        pts = _make_point_cloud(batch=1, n_points=4).requires_grad_(True)
        diagrams = ph(pts)
        loss = diagrams[0].sum()
        loss.backward()
        assert pts.grad is not None
        assert (pts.grad.abs().sum() > 0).item()

    def test_is_nn_module(self, ph):
        assert isinstance(ph, nn.Module)

    def test_invalid_max_dim(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_dim"):
            PersistentHomology(max_dim=-1)

    def test_invalid_max_radius_nan(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_radius"):
            PersistentHomology(max_radius=float("nan"))

    def test_invalid_max_radius_zero(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_radius"):
            PersistentHomology(max_radius=0.0)

    def test_invalid_max_radius_negative(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_radius"):
            PersistentHomology(max_radius=-1.0)

    def test_invalid_metric(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="metric"):
            PersistentHomology(metric="manhattan")

    def test_invalid_reduction(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="reduction"):
            PersistentHomology(reduction="bad_reduction")

    def test_invalid_memory_mode(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="memory_mode"):
            PersistentHomology(memory_mode="bad_mode")

    def test_invalid_max_memory_gb(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_memory_gb"):
            PersistentHomology(max_memory_gb=0.0)

    def test_invalid_max_memory_gb_negative(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        with pytest.raises(InvalidArgumentError, match="max_memory_gb"):
            PersistentHomology(max_memory_gb=-1.0)

    def test_defaults(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        assert ph.max_dim == 1
        assert ph.max_radius == float("inf")
        assert ph.metric == "euclidean"
        assert ph.reduction == "clearing"
        assert ph.memory_mode == "standard"
        assert ph.max_memory_gb is None

    @_core_skip
    def test_non_3d_input_raises(self, ph):
        with pytest.raises(ValidationError):
            ph(torch.randn(5, 3))

    @_core_skip
    def test_2d_input_raises(self, ph):
        with pytest.raises(ValidationError):
            ph(torch.randn(8, 5, 3, 2))

    @_core_skip
    def test_empty_batch_raises(self, ph):
        with pytest.raises(ValidationError):
            ph(torch.zeros(0, 8, 3))

    @_core_skip
    def test_empty_points_raises(self, ph):
        with pytest.raises(ValidationError):
            ph(torch.zeros(2, 0, 3))

    @_core_skip
    def test_empty_dim_raises(self, ph):
        with pytest.raises(ValidationError):
            ph(torch.zeros(2, 8, 0))

    @_core_skip
    def test_nan_input_rejected(self, ph):
        pts = torch.tensor([[[0.0, float("nan")]]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="finite"):
            ph(pts)

    @_core_skip
    def test_inf_input_rejected(self, ph):
        pts = torch.tensor([[[0.0, float("inf")]]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="finite"):
            ph(pts)

    def test_to_tensor(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        t = torch.empty(1)
        ph.to(t)
        assert ph.device == t.device
        assert ph.dtype == t.dtype

    def test_to_device_str(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.to("cpu")
        assert ph.device == torch.device("cpu")

    def test_to_dtype(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.to(dtype=torch.float64)
        assert ph.dtype == torch.float64

    def test_to_device_and_dtype_kwargs(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.to(device="cpu", dtype=torch.float64)
        assert ph.device == torch.device("cpu")
        assert ph.dtype == torch.float64

    def test_cpu(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.cpu()
        assert ph.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_default(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.cuda()
        assert ph.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_index(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.cuda(0)
        assert ph.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.cuda(torch.device("cuda:0"))
        assert ph.device.type == "cuda"

    def test_device_property(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        assert isinstance(ph.device, torch.device)

    def test_half(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.half()
        assert ph.dtype == torch.float16

    def test_float(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.double()
        ph.float()
        assert ph.dtype == torch.float32

    def test_double(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.double()
        assert ph.dtype == torch.float64

    def test_dtype_property(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        assert isinstance(ph.dtype, torch.dtype)

    def test_train(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.eval()
        ph.train()
        assert ph.training

    def test_eval(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        ph.eval()
        assert not ph.training

    def test_extra_repr(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=2, max_radius=1.5, reduction="standard")
        rep = ph.extra_repr()
        assert "max_dim=2" in rep
        assert "max_radius=1.500" in rep
        assert "standard" in rep

    def test_estimate_memory_gb(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        gb = ph._estimate_memory_gb(100, 3)
        assert gb > 0
        assert math.isfinite(gb)

    @_core_skip
    def test_extreme_memory_under_budget(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(
            max_dim=0, max_radius=float("inf"), memory_mode="extreme", max_memory_gb=1000.0
        )
        pts = _make_point_cloud(batch=1, n_points=4)
        diagrams = ph(pts)
        assert len(diagrams) == 1

    @_core_skip
    def test_extreme_memory_over_budget_raises(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(
            max_dim=0, max_radius=float("inf"), memory_mode="extreme", max_memory_gb=0.0
        )
        pts = _make_point_cloud(batch=1, n_points=100)
        with pytest.raises(RuntimeError, match="memory budget"):
            ph(pts)

    def test_init_device_dtype_kwargs(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(device=torch.device("cpu"), dtype=torch.float64)
        assert ph.device == torch.device("cpu")
        assert ph.dtype == torch.float64

    def test_to_with_tensor_picks_up_device_dtype(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology()
        t = torch.empty(1, dtype=torch.float64, device=torch.device("cpu"))
        ph.to(t)
        assert ph.device == t.device
        assert ph.dtype == t.dtype

    def test_memory_mode_extreme_default_budget(self):
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(memory_mode="extreme")
        assert ph.max_memory_gb is None

    @_core_skip
    def test_compute_persistence_diagrams(self):
        from pynerve.nn.persistent_homology import compute_persistence_diagrams

        pts = _make_point_cloud(batch=1, n_points=4)
        diagrams = compute_persistence_diagrams(pts, max_dim=0, max_radius=5.0)
        assert isinstance(diagrams, list)
        if len(diagrams) > 0:
            assert isinstance(diagrams[0], Tensor)


# ----------------------------------------------------------- _persistence_core_impl


class TestPersistenceResult:
    def test_keep_batched(self):
        from pynerve.torch._persistence_core_impl import PersistenceResult

        d = torch.rand(2, 5, 3)
        m = torch.ones(2, 5, dtype=torch.bool)
        n = torch.tensor([[3, 2], [4, 1]])
        pr = PersistenceResult(diagrams=d, mask=m, num_pairs=n, was_batched=True)
        unbatched = pr.unbatch()
        assert unbatched.diagrams.shape == d.shape
        assert unbatched.was_batched

    def test_unbatch_single(self):
        from pynerve.torch._persistence_core_impl import PersistenceResult

        d = torch.rand(1, 5, 3)
        m = torch.ones(1, 5, dtype=torch.bool)
        n = torch.tensor([[3, 2]])
        pr = PersistenceResult(diagrams=d, mask=m, num_pairs=n, was_batched=False)
        unbatched = pr.unbatch()
        assert unbatched.diagrams.dim() == 2
        assert unbatched.mask.dim() == 1
        assert unbatched.num_pairs.dim() == 1
        assert not unbatched.was_batched

    def test_unbatch_already_unbatched(self):
        from pynerve.torch._persistence_core_impl import PersistenceResult

        d = torch.rand(5, 3)
        m = torch.ones(5, dtype=torch.bool)
        n = torch.tensor([3, 2])
        pr = PersistenceResult(diagrams=d, mask=m, num_pairs=n, was_batched=False)
        unbatched = pr.unbatch()
        assert unbatched.diagrams.dim() == 2
        assert not unbatched.was_batched


class TestStackBackendParts:
    def test_single_item(self):
        from pynerve.torch._persistence_core_impl import _stack_backend_parts

        d = torch.rand(5, 3)
        m = torch.ones(5, dtype=torch.bool)
        n = torch.tensor([3, 2])
        stacked = _stack_backend_parts([d], [m], [n])
        assert stacked[0].shape == (1, 5, 3)
        assert stacked[1].shape == (1, 5)

    def test_multiple_same_size(self):
        from pynerve.torch._persistence_core_impl import _stack_backend_parts

        d1 = torch.rand(4, 3)
        d2 = torch.rand(4, 3)
        m1 = torch.ones(4, dtype=torch.bool)
        m2 = torch.ones(4, dtype=torch.bool)
        n1 = torch.tensor([2, 2])
        n2 = torch.tensor([1, 3])
        stacked = _stack_backend_parts([d1, d2], [m1, m2], [n1, n2])
        assert stacked[0].shape == (2, 4, 3)
        assert stacked[1].shape == (2, 4)

    def test_different_sizes_padded(self):
        from pynerve.torch._persistence_core_impl import _stack_backend_parts

        d1 = torch.rand(2, 3)
        d2 = torch.rand(5, 3)
        m1 = torch.ones(2, dtype=torch.bool)
        m2 = torch.ones(5, dtype=torch.bool)
        n1 = torch.tensor([1, 1])
        n2 = torch.tensor([3, 2])
        stacked = _stack_backend_parts([d1, d2], [m1, m2], [n1, n2])
        assert stacked[0].shape == (2, 5, 3)
        assert stacked[1].shape == (2, 5)
        assert (stacked[0][0, 2:, :] == 0).all()

    def test_empty_list_raises(self):
        from pynerve.torch._persistence_core_impl import _stack_backend_parts

        with pytest.raises(ValueError, match="no diagrams"):
            _stack_backend_parts([], [], [])


class TestPartsFromDiagram:
    def test_basic(self):
        from pynerve.torch._persistence_core_impl import _parts_from_diagram

        diagram = torch.zeros(2, 5, 3)
        diagram[0, 0, :2] = torch.tensor([0.0, 1.0])
        diagram[0, 0, 2] = 0.0
        diagram[0, 1, :2] = torch.tensor([0.2, 0.8])
        diagram[0, 1, 2] = 1.0

        d, m, n = _parts_from_diagram(diagram, 1)
        assert d.shape == (2, 5, 3)
        assert m.shape == (2, 5)
        assert m.dtype == torch.bool
        assert n.shape == (2, 2)


class TestWitnessDistanceMatrix:
    def test_return_shape(self):
        from pynerve.torch._persistence_core_impl import _witness_distance_matrix

        landmarks = torch.rand(2, 3, 2)
        witnesses = torch.rand(2, 5, 2)
        dist = _witness_distance_matrix(landmarks, witnesses, float("inf"))
        assert dist.shape == (2, 3, 3)
        assert dist.dtype == landmarks.dtype

    def test_diagonal_zeros(self):
        from pynerve.torch._persistence_core_impl import _witness_distance_matrix

        landmarks = _make_point_cloud(batch=1, n_points=3, dim=2)
        witnesses = _make_point_cloud(batch=1, n_points=5, dim=2)
        dist = _witness_distance_matrix(landmarks, witnesses, float("inf"))
        for i in range(3):
            assert dist[0, i, i] == 0.0

    def test_symmetric(self):
        from pynerve.torch._persistence_core_impl import _witness_distance_matrix

        landmarks = _make_point_cloud(batch=1, n_points=3, dim=2)
        witnesses = _make_point_cloud(batch=1, n_points=5, dim=2)
        dist = _witness_distance_matrix(landmarks, witnesses, 10.0)
        assert torch.allclose(dist[0], dist[0].T)

    def test_finite_radius_clamps(self):
        from pynerve.torch._persistence_core_impl import _witness_distance_matrix

        landmarks = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
        witnesses = torch.tensor([[[5.0, 5.0]]])
        dist = _witness_distance_matrix(landmarks, witnesses, 1.0)
        assert torch.isinf(dist[0, 0, 1])
        assert torch.isinf(dist[0, 1, 0])


class TestDistanceMatrixParts:
    def test_returns_tensors(self):
        from pynerve.torch._persistence_core_impl import _distance_matrix_parts

        dm = _make_dist_matrix(n_points=4, batch=1)
        d, m, n = _distance_matrix_parts(dm, 1)
        assert isinstance(d, Tensor)
        assert isinstance(m, Tensor)
        assert isinstance(n, Tensor)


class TestPythonBackend:
    @pytest.fixture(scope="class")
    def backend(self):
        from pynerve.torch._persistence_core_impl import PythonBackend

        return PythonBackend()

    def test_compute_vr(self, backend):
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_vr(pts, 1, float("inf"), "euclidean")
        assert d.dim() == 3
        assert m.dim() == 2
        assert n.dim() == 2

    def test_compute_alpha(self, backend):
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_alpha(pts, 1)
        assert isinstance(d, Tensor)
        assert isinstance(m, Tensor)
        assert isinstance(n, Tensor)

    def test_compute_witness(self, backend):
        landmarks = _make_point_cloud(batch=1, n_points=3, dim=2)
        witnesses = _make_point_cloud(batch=1, n_points=5, dim=2)
        d, m, n = backend.compute_witness(landmarks, witnesses, 1, float("inf"))
        assert isinstance(d, Tensor)
        assert isinstance(m, Tensor)
        assert isinstance(n, Tensor)

    def test_compute_distance_matrix(self, backend):
        dm = _make_dist_matrix(n_points=4, batch=1)
        d, m, n = backend.compute_distance_matrix(dm, 1)
        assert isinstance(d, Tensor)
        assert isinstance(m, Tensor)
        assert isinstance(n, Tensor)

    def test_compute_vr_rejects_bad_metric(self, backend):
        pts = _make_point_cloud(batch=1, n_points=4)
        with pytest.raises(ValueError, match="metric"):
            backend.compute_vr(pts, 0, 1.0, "bad_metric")

    def test_compute_vr_negative_radius(self, backend):
        pts = _make_point_cloud(batch=1, n_points=4)
        with pytest.raises(ValueError, match="max_radius"):
            backend.compute_vr(pts, 0, -1.0, "euclidean")

    def test_compute_vr_manhattan(self, backend):
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_vr(pts, 0, float("inf"), "manhattan")
        assert d.dim() == 3
        assert d.shape[0] == 1


class TestCoreCBackend:
    @_core_skip
    def test_instantiates(self):
        from pynerve.torch._persistence_core_impl import CoreCBackend

        backend = CoreCBackend()
        assert isinstance(backend, CoreCBackend)

    @_core_skip
    def test_compute_vr(self):
        from pynerve.torch._persistence_core_impl import CoreCBackend

        backend = CoreCBackend()
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_vr(pts, 1, float("inf"), "euclidean")
        assert d.dim() == 3

    @_core_skip
    def test_compute_witness(self):
        from pynerve.torch._persistence_core_impl import CoreCBackend

        backend = CoreCBackend()
        landmarks = _make_point_cloud(batch=1, n_points=3, dim=2)
        witnesses = _make_point_cloud(batch=1, n_points=5, dim=2)
        d, m, n = backend.compute_witness(landmarks, witnesses, 1, float("inf"))
        assert isinstance(d, Tensor)

    @_core_skip
    def test_compute_alpha(self):
        from pynerve.torch._persistence_core_impl import CoreCBackend

        backend = CoreCBackend()
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_alpha(pts, 1)
        assert isinstance(d, Tensor)

    @_core_skip
    def test_compute_distance_matrix(self):
        from pynerve.torch._persistence_core_impl import CoreCBackend

        backend = CoreCBackend()
        dm = _make_dist_matrix(n_points=4, batch=1)
        d, m, n = backend.compute_distance_matrix(dm, 1)
        assert isinstance(d, Tensor)


class TestTorchCBackend:
    @_torchc_skip
    def test_instantiates(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        assert isinstance(backend, TorchCBackend)

    @_torchc_skip
    def test_compute_vr(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        pts = _make_point_cloud(batch=1, n_points=4)
        d, m, n = backend.compute_vr(pts, 1, float("inf"), "euclidean")
        assert d.dim() == 3

    @_torchc_skip
    def test_compute_witness(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        landmarks = _make_point_cloud(batch=1, n_points=3, dim=2)
        witnesses = _make_point_cloud(batch=1, n_points=5, dim=2)
        d, m, n = backend.compute_witness(landmarks, witnesses, 1, float("inf"))
        assert isinstance(d, Tensor)

    @_torchc_skip
    def test_compute_distance_matrix_2d(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        dm = _make_dist_matrix(n_points=4, batch=1)[0]
        d, m, n = backend.compute_distance_matrix(dm, 1)
        assert isinstance(d, Tensor)

    @_torchc_skip
    def test_compute_distance_matrix_3d(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        dm = _make_dist_matrix(n_points=4, batch=2)
        d, m, n = backend.compute_distance_matrix(dm, 1)
        assert d.dim() == 3
        assert d.shape[0] == 2

    @_torchc_skip
    def test_compute_distance_matrix_bad_rank(self):
        from pynerve.torch._persistence_core_impl import TorchCBackend

        backend = TorchCBackend()
        dm = torch.rand(4)
        with pytest.raises(ValueError):
            backend.compute_distance_matrix(dm, 1)


class TestBackendDiscovery:
    def test_get_best_backend_unknown_raises(self):
        from pynerve.torch._persistence_core_impl import get_best_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_best_backend("unknown_requirement")

    def test_get_best_backend_default(self):
        from pynerve.torch._persistence_core_impl import get_best_backend

        backend = get_best_backend()
        from pynerve.torch._persistence_core_impl import PersistenceComputer

        assert isinstance(backend, PersistenceComputer)

    def test_get_persistence_backend_cached(self):
        from pynerve.torch._persistence_core_impl import (
            _persistence_backend,
            get_persistence_backend,
        )

        saved = _persistence_backend
        try:
            import pynerve.torch._persistence_core_impl as mod

            mod._persistence_backend = None
            b1 = get_persistence_backend()
            b2 = get_persistence_backend()
            assert b2 is b1
        finally:
            import pynerve.torch._persistence_core_impl as mod

            mod._persistence_backend = saved

    @_torchc_skip
    def test_get_best_backend_torch_c(self):
        from pynerve.torch._persistence_core_impl import (
            TorchCBackend,
            get_best_backend,
        )

        backend = get_best_backend("torch_c")
        assert isinstance(backend, TorchCBackend)

    @_core_skip
    def test_get_best_backend_core_c(self):
        from pynerve.torch._persistence_core_impl import (
            CoreCBackend,
            get_best_backend,
        )

        backend = get_best_backend("core_c")
        assert isinstance(backend, CoreCBackend)


class TestComputePersistenceVR:
    def test_basic_h0(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        assert result.diagrams.dim() == 3
        assert result.mask.dim() == 2

    def test_batched(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=2, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        assert result.diagrams.shape[0] == 2

    def test_unbatched_input(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)[0]
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        assert result.was_batched is True

    def test_already_batched(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=2, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        assert result.was_batched is False

    def test_cosine_metric(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0, metric="cosine")
        assert result.diagrams.dim() == 3

    def test_chebyshev_metric(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0, metric="chebyshev")
        assert result.diagrams.dim() == 3

    def test_invalid_metric(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        with pytest.raises(ValueError, match="metric"):
            compute_persistence_vr(pts, max_dim=0, metric="not_a_metric")

    def test_invalid_max_dim(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        with pytest.raises(ValueError, match="max_dim"):
            compute_persistence_vr(pts, max_dim=-1)

    def test_invalid_max_radius(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        with pytest.raises(ValueError, match="max_radius"):
            compute_persistence_vr(pts, max_radius=0.0)

    def test_result_has_fields(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        assert hasattr(result, "diagrams")
        assert hasattr(result, "mask")
        assert hasattr(result, "num_pairs")
        assert hasattr(result, "was_batched")

    def test_unbatch_method(self):
        from pynerve.torch._persistence_core_impl import compute_persistence_vr

        pts = _make_point_cloud(batch=1, n_points=4)
        result = compute_persistence_vr(pts, max_dim=0, max_radius=10.0)
        unbatched = result.unbatch()
        assert unbatched.diagrams.dim() == 2


class TestAbstractPersistenceComputer:
    def test_compute_vr_raises(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer

        comp = PersistenceComputer()
        with pytest.raises(RuntimeError, match="abstract"):
            comp.compute_vr(None, 0, 1.0, "euclidean")

    def test_compute_witness_raises(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer

        comp = PersistenceComputer()
        with pytest.raises(RuntimeError, match="abstract"):
            comp.compute_witness(None, None, 0, 1.0)

    def test_compute_alpha_raises(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer

        comp = PersistenceComputer()
        with pytest.raises(RuntimeError, match="abstract"):
            comp.compute_alpha(None, 0)

    def test_compute_distance_matrix_raises(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer

        comp = PersistenceComputer()
        with pytest.raises(RuntimeError, match="abstract"):
            comp.compute_distance_matrix(None, 0)


# -------------------------------------------------------------- _persistence_core


class TestPersistenceCoreReExports:
    def test_core_exports(self):
        from pynerve.torch._persistence_core import (
            CoreCBackend,
            PersistenceComputer,
            PersistenceResult,
            PythonBackend,
            TorchCBackend,
            compute_persistence_vr,
            get_best_backend,
            get_persistence_backend,
        )

        import pynerve.torch._persistence_core_impl as impl

        assert CoreCBackend is impl.CoreCBackend
        assert PersistenceComputer is impl.PersistenceComputer
        assert PersistenceResult is impl.PersistenceResult
        assert PythonBackend is impl.PythonBackend
        assert TorchCBackend is impl.TorchCBackend
        assert compute_persistence_vr is impl.compute_persistence_vr
        assert get_best_backend is impl.get_best_backend
        assert get_persistence_backend is impl.get_persistence_backend
