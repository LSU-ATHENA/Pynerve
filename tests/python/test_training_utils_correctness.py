"""Numerical correctness tests for training utility modules."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


class TestDiagramDistanceLoss:
    """Numerical correctness for DiagramDistanceLoss."""

    def test_identical_diagrams_zero_wasserstein(self) -> None:
        from pynerve.torch.training_utils import DiagramDistanceLoss

        loss = DiagramDistanceLoss(metric="wasserstein", p=2.0)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        val = loss(d, d)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_identical_diagrams_zero_bottleneck(self) -> None:
        from pynerve.torch.training_utils import DiagramDistanceLoss

        loss = DiagramDistanceLoss(metric="bottleneck")
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        val = loss(d, d)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.torch.training_utils import DiagramDistanceLoss

        loss = DiagramDistanceLoss(metric="wasserstein", p=2.0)
        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        val = loss(d1, d2)
        assert val.item() > 0

    def test_bottleneck_different(self) -> None:
        from pynerve.torch.training_utils import DiagramDistanceLoss

        loss = DiagramDistanceLoss(metric="bottleneck")
        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        val = loss(d1, d2)
        assert val.item() == pytest.approx(1.0, abs=1e-6)

    def test_with_persistence_diagram_object(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.training_utils import DiagramDistanceLoss

        loss = DiagramDistanceLoss(metric="wasserstein", p=2.0)
        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        val = loss(pd, pd)
        assert val.item() == pytest.approx(0.0, abs=1e-10)


class TestTopologicalRegularization:
    """Numerical correctness for TopologicalRegularization."""

    def test_l1_penalty_exact(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 5.0},
            weights={"total_persistence": 1.0},
            penalty_type="l1",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        val = reg(d)
        # total_persistence = 1+2+3 = 6, target = 5, l1 diff = 1
        assert val.item() == pytest.approx(1.0, abs=1e-10)

    def test_l2_penalty_exact(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 5.0},
            weights={"total_persistence": 1.0},
            penalty_type="l2",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        val = reg(d)
        # total_persistence = 6, target = 5, l2 diff = 1^2 = 1
        assert val.item() == pytest.approx(1.0, abs=1e-10)

    def test_smooth_penalty_large_diff(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 0.0},
            weights={"total_persistence": 1.0},
            penalty_type="smooth",
        )
        d = torch.tensor([[0.0, 5.0]], dtype=torch.float64)
        val = reg(d)
        # diff = 5, |diff| >= 1, smooth = |diff| - 0.5 = 5 - 0.5 = 4.5
        assert val.item() == pytest.approx(4.5, abs=1e-10)

    def test_smooth_penalty_small_diff(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 5.0},
            weights={"total_persistence": 1.0},
            penalty_type="smooth",
        )
        d = torch.tensor([[0.0, 5.5]], dtype=torch.float64)
        val = reg(d)
        # diff = 0.5, |diff| < 1, smooth = 0.5 * 0.5^2 = 0.125
        assert val.item() == pytest.approx(0.125, abs=1e-10)

    def test_h0_count_penalty(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"h0_count": 2.0},
            weights={"h0_count": 1.0},
            penalty_type="l1",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        val = reg(d)
        # number_of_features(d, dim=0) = 3, target = 2, l1 diff = 1
        assert val.item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_when_identical(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 3.0},
            weights={"total_persistence": 1.0},
            penalty_type="l1",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        val = reg(d)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_weighted_penalty(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 5.0},
            weights={"total_persistence": 2.5},
            penalty_type="l1",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        val = reg(d)
        # total_persistence = 6, diff = 1, weighted = 1 * 2.5 = 2.5
        assert val.item() == pytest.approx(2.5, abs=1e-10)

    def test_multi_objective(self) -> None:
        from pynerve.torch.training_utils import TopologicalRegularization

        reg = TopologicalRegularization(
            target_complexity={"total_persistence": 10.0, "mean_persistence": 2.0},
            weights={"total_persistence": 1.0, "mean_persistence": 1.0},
            penalty_type="l1",
        )
        d = torch.tensor([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]], dtype=torch.float64)
        val = reg(d)
        # total_persistence = 9, diff = 1
        # mean_persistence = 3, diff = 1
        # total = 1 + 1 = 2
        assert val.item() == pytest.approx(2.0, abs=1e-6)


class TestPersistenceCrossEntropy:
    """Numerical correctness for PersistenceCrossEntropy."""

    def test_without_diagrams_standard_ce(self) -> None:
        from pynerve.torch.training_utils import PersistenceCrossEntropy

        loss = PersistenceCrossEntropy(confidence_weighting=False)
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        val = loss(logits, targets)
        # CE = mean(-log(softmax(1,0)[0]), -log(softmax(0,1)[1]))
        # = mean(-log(exp(1)/(exp(1)+1)), -log(exp(1)/(exp(1)+1)))
        # = -log(sigmoid(1)) = -log(0.731...) = 0.313...
        expected = -math.log(math.exp(1) / (math.exp(1) + 1))
        assert val.item() == pytest.approx(expected, abs=1e-6)

    def test_with_diagrams_confidence(self) -> None:
        from pynerve.torch.training_utils import PersistenceCrossEntropy

        loss = PersistenceCrossEntropy(confidence_weighting=True, min_persistence_threshold=0.0)
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        diagrams = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]],
            dtype=torch.float64,
        )
        val = loss(logits, targets, diagrams)
        assert val.item() > 0

    def test_mean_reduction(self) -> None:
        from pynerve.torch.training_utils import PersistenceCrossEntropy

        loss = PersistenceCrossEntropy(confidence_weighting=False, reduction="mean")
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        val = loss(logits, targets)
        assert val.dim() == 0

    def test_none_reduction(self) -> None:
        from pynerve.torch.training_utils import PersistenceCrossEntropy

        loss = PersistenceCrossEntropy(confidence_weighting=False, reduction="none")
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        val = loss(logits, targets)
        assert val.shape == (2,)

    def test_with_persistence_diagram_object(self) -> None:
        from pynerve.torch import PersistenceDiagram
        from pynerve.torch.training_utils import PersistenceCrossEntropy

        loss = PersistenceCrossEntropy(confidence_weighting=True, min_persistence_threshold=0.0)
        logits = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.long)
        tensor = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        val = loss(logits, targets, pd)
        assert val.item() > 0


class TestComputeKernelSimilarity:
    """Numerical correctness for compute_kernel_similarity."""

    def test_self_similarity(self) -> None:
        from pynerve.torch.training_utils import compute_kernel_similarity

        d = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)]
        K = compute_kernel_similarity(d, d, kernel="gaussian", sigma=1.0)
        assert K.shape == (1, 1)
        assert K[0, 0].item() > 0

    def test_kernel_symmetric(self) -> None:
        from pynerve.torch.training_utils import compute_kernel_similarity

        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.0, 2.0]], dtype=torch.float64)]
        K12 = compute_kernel_similarity(d1, d2, kernel="gaussian", sigma=1.0)
        K21 = compute_kernel_similarity(d2, d1, kernel="gaussian", sigma=1.0)
        assert K12[0, 0].item() == pytest.approx(K21[0, 0].item(), abs=1e-10)

    def test_different_kernels(self) -> None:
        from pynerve.torch.training_utils import compute_kernel_similarity

        d = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        for kernel in ("gaussian", "linear", "sliced_wasserstein"):
            K = compute_kernel_similarity(d, d, kernel=kernel)
            assert torch.isfinite(K).all()
