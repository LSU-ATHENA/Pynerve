from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

if TYPE_CHECKING:
    pass


class _RowEncoder(nn.Module):
    def __init__(self, out_dim: int = 8, in_dim: int = 2) -> None:
        super().__init__()
        self.output_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def _make_finite_diagram(n: int = 4, cols: int = 2):
    births = torch.linspace(0.0, 3.0, n)
    deaths = births + 1.0
    d = torch.stack([births, deaths], dim=1)
    if cols > 2:
        extra = torch.zeros(n, cols - 2)
        d = torch.cat([d, extra], dim=1)
    return d


class TestFiltrationOrderingTask:
    def test_init_negative_max_simplices_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="max_simplices must be positive"):
            FiltrationOrderingTask(enc, max_simplices=0)
        with pytest.raises(ValueError, match="max_simplices must be positive"):
            FiltrationOrderingTask(enc, max_simplices=-1)

    def test_init_valid(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        task = FiltrationOrderingTask(enc, max_simplices=50)
        assert task.encoder is enc
        assert task.max_simplices == 50

    def test_forward_wrong_dim_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        task = FiltrationOrderingTask(enc)
        with pytest.raises(ValueError, match="must be 2D"):
            task.forward(torch.randn(3), torch.randn(3, 2))
        with pytest.raises(ValueError, match="must be 2D"):
            task.forward(torch.randn(3, 2), torch.randn(3))

    def test_forward_mismatched_rows_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        task = FiltrationOrderingTask(enc)
        with pytest.raises(ValueError, match="same number of rows"):
            task.forward(torch.randn(3, 2), torch.randn(4, 2))

    def test_forward_exceeds_max_simplices_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        task = FiltrationOrderingTask(enc, max_simplices=3)
        with pytest.raises(ValueError, match="exceeds max_simplices"):
            task.forward(torch.randn(5, 2), torch.randn(5, 2))

    def test_forward_non_finite_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8)
        task = FiltrationOrderingTask(enc)
        bad = torch.tensor([[float("nan"), 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="finite values"):
            task.forward(bad, torch.randn(2, 2))
        with pytest.raises(ValueError, match="finite values"):
            task.forward(torch.randn(2, 2), torch.tensor([[float("inf"), 1.0], [0.0, 1.0]]))

    def test_forward_shape_and_finite(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8, in_dim=5)
        task = FiltrationOrderingTask(enc)
        simplices = torch.randn(5, 2)
        feats = torch.randn(5, 3)
        scores = task.forward(simplices, feats)
        assert scores.shape == (5,)
        assert scores.dtype == torch.float32

    def test_compute_loss_mismatched_shape_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8, in_dim=4)
        task = FiltrationOrderingTask(enc)
        simplices = torch.randn(5, 2)
        feats = torch.randn(5, 2)
        with pytest.raises(ValueError, match="true_order must have shape"):
            task.compute_loss(simplices, feats, torch.randn(3))

    def test_compute_loss_non_finite_raises(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8, in_dim=4)
        task = FiltrationOrderingTask(enc)
        simplices = torch.randn(5, 2)
        feats = torch.randn(5, 2)
        bad_order = torch.tensor([float("nan"), 0.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="finite values"):
            task.compute_loss(simplices, feats, bad_order)

    def test_compute_loss_scalar(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8, in_dim=5)
        task = FiltrationOrderingTask(enc)
        simplices = torch.randn(4, 2)
        feats = torch.randn(4, 3)
        true_order = torch.tensor([0.0, 1.0, 2.0, 3.0])
        loss = task.compute_loss(simplices, feats, true_order)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_loss_zero_when_all_equal_order(self) -> None:
        from pynerve.ssl._filtration import FiltrationOrderingTask

        enc = _RowEncoder(out_dim=8, in_dim=5)
        task = FiltrationOrderingTask(enc)
        simplices = torch.randn(4, 2)
        feats = torch.randn(4, 3)
        true_order = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss = task.compute_loss(simplices, feats, true_order)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)


class TestMultiTaskTopologySSL:
    def test_init_negative_max_dim_raises(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="max_dim must be non-negative"):
            MultiTaskTopologySSL(enc, max_dim=-1)

    def test_init_unknown_task_weight_raises(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="unknown task weights"):
            MultiTaskTopologySSL(enc, task_weights={"nonexistent": 0.5})

    def test_init_negative_task_weight_raises(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="must be non-negative"):
            MultiTaskTopologySSL(enc, task_weights={"completion": -0.5})

    def test_init_defaults(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        assert model.task_weights == {"completion": 1.0, "betti": 0.5, "denoising": 1.0}

    def test_init_custom_weights(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, task_weights={"completion": 2.0})
        assert model.task_weights["completion"] == 2.0
        assert model.task_weights["betti"] == 0.5
        assert model.task_weights["denoising"] == 1.0

    def test_forward_completion_shape(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        out = model.forward(d, "completion")
        assert out.shape == (4, 3)

    def test_forward_betti_shape(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, max_dim=3)
        d = _make_finite_diagram(4)
        out = model.forward(d, "betti")
        assert out.shape == (4,)
        assert (out >= 0).all()

    def test_forward_denoising_shape(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        out = model.forward(d, "denoising")
        assert out.shape == (4, 3)

    def test_forward_unknown_task_raises(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        with pytest.raises(ValueError, match="Unknown task"):
            model.forward(d, "nonexistent")

    def test_compute_multitask_loss_requires_ssl_diagram(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        bad = torch.randn(4)
        with pytest.raises(ValueError, match="must have shape"):
            model.compute_multitask_loss(bad, {})

    def test_compute_multitask_loss_completion_missing_mask(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        target = torch.tensor([[0.0, 1.0, 0.0] for _ in range(4)])
        with pytest.raises(ValueError, match="requires completion_mask"):
            model.compute_multitask_loss(d, {"completion": target})

    def test_compute_multitask_loss_completion(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        mask = torch.tensor([True, True, False, False], dtype=torch.bool)
        target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        losses = model.compute_multitask_loss(d, {"completion_mask": mask, "completion": target})
        assert "completion" in losses
        assert "total" in losses
        assert losses["total"].dim() == 0

    def test_compute_multitask_loss_betti(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, max_dim=0)
        d = _make_finite_diagram(4)
        target = torch.tensor([0.0])
        losses = model.compute_multitask_loss(d, {"betti": target})
        assert "betti" in losses
        assert "total" in losses
        assert losses["total"].dim() == 0
        assert losses["betti"].item() >= 0

    def test_compute_multitask_loss_betti_shape_mismatch(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, max_dim=0)
        d = _make_finite_diagram(4)
        with pytest.raises(ValueError, match="betti target must match"):
            model.compute_multitask_loss(d, {"betti": torch.tensor([1.0, 2.0])})

    def test_compute_multitask_loss_denoising(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        clean = d.clone()
        losses = model.compute_multitask_loss(d, {"denoising": clean})
        assert "denoising" in losses
        assert "total" in losses
        assert losses["total"].dim() == 0

    def test_compute_multitask_loss_denoising_shape_mismatch(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        with pytest.raises(ValueError, match="denoising target must match"):
            model.compute_multitask_loss(d, {"denoising": torch.randn(3, 2)})

    def test_compute_multitask_loss_all(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, max_dim=0)
        d = _make_finite_diagram(4)
        mask = torch.tensor([True, True, True, False], dtype=torch.bool)
        comp_target = torch.tensor([[0.0, 1.0, 0.0] for _ in range(4)])
        betti_target = torch.tensor([0.0])
        losses = model.compute_multitask_loss(
            d,
            {
                "completion_mask": mask,
                "completion": comp_target,
                "betti": betti_target,
                "denoising": d.clone(),
            },
        )
        assert "completion" in losses
        assert "betti" in losses
        assert "denoising" in losses
        assert "total" in losses
        assert losses["total"].dim() == 0

    def test_compute_multitask_loss_empty_tasks(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = _make_finite_diagram(4)
        losses = model.compute_multitask_loss(d, {})
        assert "total" in losses
        assert losses["total"].item() == pytest.approx(0.0, abs=1e-10)

    def test_compute_multitask_loss_non_finite_betti_target(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc, max_dim=0)
        d = _make_finite_diagram(4)
        with pytest.raises(ValueError, match="finite values"):
            model.compute_multitask_loss(d, {"betti": torch.tensor([float("nan")])})

    def test_compute_multitask_loss_empty_diagram(self) -> None:
        from pynerve.ssl._multitask import MultiTaskTopologySSL

        enc = _RowEncoder(out_dim=8)
        model = MultiTaskTopologySSL(enc)
        d = torch.empty(0, 2)
        mask = torch.empty(0, dtype=torch.bool)
        comp_target = torch.empty(0, 3)
        losses = model.compute_multitask_loss(
            d, {"completion_mask": mask, "completion": comp_target}
        )
        assert "total" in losses
        assert losses["total"].dim() == 0


class TestPersistencePredictionTask:
    def test_init_invalid_mask_ratio_raises(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            PersistencePredictionTask(enc, mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            PersistencePredictionTask(enc, mask_ratio=1.5)

    def test_init_non_positive_prediction_dim_raises(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        with pytest.raises(ValueError, match="prediction_dim must be positive"):
            PersistencePredictionTask(enc, prediction_dim=0)

    def test_init_valid(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc, mask_ratio=0.2, prediction_dim=3)
        assert task.mask_ratio == 0.2

    def test_forward_invalid_diagram_raises(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc)
        with pytest.raises((ValueError, TypeError)):
            task.forward(torch.randn(4))

    def test_forward_empty_diagram(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc)
        empty = torch.empty(0, 2)
        loss, preds, targets = task.forward(empty)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)
        assert preds.shape == (0, 2)
        assert targets.shape == (0, 2)

    def test_forward_shapes(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc, mask_ratio=0.5)
        d = _make_finite_diagram(8)
        torch.manual_seed(42)
        loss, preds, targets = task.forward(d)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert preds.ndim == 2
        assert preds.shape[1] == 2
        assert targets.ndim == 2
        assert targets.shape[1] == 2
        assert preds.shape[0] == targets.shape[0]

    def test_forward_prediction_dim_custom(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc, mask_ratio=0.3, prediction_dim=2)
        d = _make_finite_diagram(10)
        torch.manual_seed(42)
        loss, preds, targets = task.forward(d)
        assert preds.shape[1] == 2
        assert targets.shape[1] == 2
        assert preds.shape[0] == targets.shape[0]
        assert loss.dim() == 0

    def test_forward_non_finite_diagram_raises(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc)
        bad = torch.tensor([[float("nan"), 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="finite"):
            task.forward(bad)

    def test_forward_death_lt_birth_raises(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc)
        bad = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            task.forward(bad)

    def test_forward_computes_loss_on_masked_only(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc, mask_ratio=1.0)
        d = _make_finite_diagram(10)
        torch.manual_seed(42)
        loss, preds, targets = task.forward(d)
        assert preds.shape[0] == 10
        assert targets.shape[0] == 10

    def test_forward_single_pair(self) -> None:
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask

        enc = _RowEncoder(out_dim=8)
        task = PersistencePredictionTask(enc, mask_ratio=0.5)
        d = _make_finite_diagram(1)
        torch.manual_seed(42)
        loss, preds, targets = task.forward(d)
        assert preds.shape[0] == 1
        assert targets.shape[0] == 1


class TestMerge:
    def test_match_persistence_diagrams_empty(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        result = match_persistence_diagrams([])
        assert result == []

    def test_match_persistence_diagrams_single(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d = np.array([[0.0, 1.0], [0.0, 2.0]])
        result = match_persistence_diagrams([d])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], d)

    def test_match_persistence_diagrams_two(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, 2.0], [0.0, 3.0]])
        d2 = np.array([[0.1, 1.9], [0.0, 3.1]])
        result = match_persistence_diagrams([d1, d2], threshold=10.0)
        assert len(result) == 2
        assert result[0].shape[0] >= 0
        assert result[1].shape[0] >= 0

    def test_match_persistence_diagrams_with_dim(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        d2 = np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 0.0]])
        result = match_persistence_diagrams([d1, d2], dim=1, threshold=10.0)
        assert len(result) == 2
        for r in result:
            assert (r[:, 2].astype(int) == 1).all()

    def test_match_persistence_diagrams_dim_no_dim_col(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, 1.0], [0.0, 2.0]])
        d2 = np.array([[0.0, 1.0], [0.0, 2.0]])
        result = match_persistence_diagrams([d1, d2], dim=0, threshold=10.0)
        assert len(result) == 2

    def test_match_persistence_diagrams_tight_threshold(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, 1.0], [0.0, 2.0]])
        d2 = np.array([[10.0, 11.0], [10.0, 12.0]])
        result = match_persistence_diagrams([d1, d2], threshold=0.5)
        assert len(result) == 2
        assert result[1].shape[0] == 0

    def test_match_persistence_diagrams_non_finite_pairs(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, float("inf")], [0.0, 2.0]])
        d2 = np.array([[0.0, 1.0], [0.0, 2.0]])
        result = match_persistence_diagrams([d1, d2], threshold=10.0)
        assert len(result) == 2

    def test_filter_by_dim_no_dim_col(self) -> None:
        from pynerve.merge import _filter_by_dim

        d = np.array([[0.0, 1.0], [0.0, 2.0]])
        result = _filter_by_dim(d, 0)
        np.testing.assert_array_equal(result, d)

    def test_filter_by_dim_filters(self) -> None:
        from pynerve.merge import _filter_by_dim

        d = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 0.0]])
        result = _filter_by_dim(d, 0)
        assert result.shape[0] == 2
        assert (result[:, 2].astype(int) == 0).all()

    def test_bottleneck_match_both_empty(self) -> None:
        from pynerve.merge import _bottleneck_match

        ref = np.empty((0, 2))
        tgt = np.empty((0, 2))
        result = _bottleneck_match(ref, tgt, 1.0)
        result_single = _bottleneck_match(np.array([[0.0, 1.0]]), np.empty((0, 2)), 1.0)
        assert result.shape[0] == 0
        assert result_single.shape[0] == 1

    def test_bottleneck_match_identity(self) -> None:
        from pynerve.merge import _bottleneck_match

        d = np.array([[0.0, 1.0], [0.0, 2.0]])
        result = _bottleneck_match(d, d, 0.1)
        assert result.shape[0] == 2

    def test_match_persistence_diagrams_three(self) -> None:
        from pynerve.merge import match_persistence_diagrams

        d1 = np.array([[0.0, 2.0], [0.0, 3.0]])
        d2 = np.array([[0.1, 2.1], [0.0, 3.0]])
        d3 = np.array([[0.0, 2.0], [0.1, 3.1]])
        result = match_persistence_diagrams([d1, d2, d3], threshold=5.0)
        assert len(result) == 3

    def test_bottleneck_match_preserves_ref_when_no_target_match(self) -> None:
        from pynerve.merge import _bottleneck_match

        ref = np.array([[0.0, 1.0]])
        tgt = np.array([[100.0, 101.0]])
        result = _bottleneck_match(ref, tgt, 0.5)
        assert result.shape[0] == 0 or result.shape[0] == 1
