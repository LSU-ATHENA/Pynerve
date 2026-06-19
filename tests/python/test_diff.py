"""Comprehensive tests for pynerve.diff subpackage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

SEED = 42


def _valid_diagram(n_pairs: int = 4, dims: bool = False) -> Tensor:
    torch.manual_seed(SEED)
    births = torch.rand(n_pairs) * 5
    deaths = births + torch.rand(n_pairs) * 3 + 0.1
    pairs = torch.stack([births, deaths], dim=1)
    if dims:
        d = torch.randint(0, 2, (n_pairs, 1)).float()
        return torch.cat([pairs, d], dim=1)
    return pairs


def _valid_batch_points(batch: int = 2, n: int = 8, d: int = 3) -> Tensor:
    torch.manual_seed(SEED)
    return torch.rand(batch, n, d)


# ---- _loss_helpers ----


class TestValidateNonNegativeScalar:
    def test_positive(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar

        assert _validate_non_negative_scalar("w", 3.0) == 3.0

    def test_zero(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar

        assert _validate_non_negative_scalar("w", 0.0) == 0.0

    def test_negative_raises(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar

        with pytest.raises(ValueError):
            _validate_non_negative_scalar("w", -1.0)

    def test_nan_raises(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar

        with pytest.raises(ValueError, match="finite"):
            _validate_non_negative_scalar("w", float("nan"))

    def test_inf_raises(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar

        with pytest.raises(ValueError, match="finite"):
            _validate_non_negative_scalar("w", float("inf"))


class TestValidatePositiveScalar:
    def test_positive(self):
        from pynerve.diff._loss_helpers import _validate_positive_scalar

        assert _validate_positive_scalar("t", 0.5) == 0.5

    def test_zero_raises(self):
        from pynerve.diff._loss_helpers import _validate_positive_scalar

        with pytest.raises(ValueError):
            _validate_positive_scalar("t", 0.0)

    def test_negative_raises(self):
        from pynerve.diff._loss_helpers import _validate_positive_scalar

        with pytest.raises(ValueError):
            _validate_positive_scalar("t", -0.1)

    def test_nan_raises(self):
        from pynerve.diff._loss_helpers import _validate_positive_scalar

        with pytest.raises(ValueError, match="finite"):
            _validate_positive_scalar("t", float("nan"))


class TestPersistenceValues:
    def test_valid_2col(self):
        from pynerve.diff._loss_helpers import _persistence_values

        d = torch.tensor([[0.0, 1.0], [1.0, 3.0]])
        p = _persistence_values(d)
        assert torch.allclose(p, torch.tensor([1.0, 2.0]))

    def test_valid_3col(self):
        from pynerve.diff._loss_helpers import _persistence_values

        d = torch.tensor([[0.0, 2.0, 0.0], [1.0, 4.0, 1.0]])
        p = _persistence_values(d, min_cols=3)
        assert torch.allclose(p, torch.tensor([2.0, 3.0]))

    def test_empty_returns_zeros(self):
        from pynerve.diff._loss_helpers import _persistence_values

        p = _persistence_values(torch.empty(0, 2))
        assert p.numel() == 0

    def test_one_3col_raises(self):
        from pynerve.diff._loss_helpers import _persistence_values

        with pytest.raises(ValueError):
            _persistence_values(torch.tensor([[0.0, 1.0]]), min_cols=3)

    def test_nan_birth_raises(self):
        from pynerve.diff._loss_helpers import _persistence_values

        with pytest.raises(ValueError, match="finite"):
            _persistence_values(torch.tensor([[float("nan"), 1.0]]))

    def test_death_less_than_birth_raises(self):
        from pynerve.diff._loss_helpers import _persistence_values

        with pytest.raises(ValueError, match="deaths.*births"):
            _persistence_values(torch.tensor([[2.0, 1.0]]))


class TestValidateDiagramDimensions:
    def test_valid_dimensions(self):
        from pynerve.diff._loss_helpers import _validate_diagram_dimensions

        d = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
        result = _validate_diagram_dimensions(d)
        assert torch.equal(result, torch.tensor([0, 1]))

    def test_non_integer_dim_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_dimensions

        d = torch.tensor([[0.0, 1.0, 0.5]])
        with pytest.raises(ValueError, match="integer"):
            _validate_diagram_dimensions(d)

    def test_negative_dim_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_dimensions

        d = torch.tensor([[0.0, 1.0, -1.0]])
        with pytest.raises(ValueError):
            _validate_diagram_dimensions(d)

    def test_nan_dim_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_dimensions

        d = torch.tensor([[0.0, 1.0, float("nan")]])
        with pytest.raises(ValueError, match="finite"):
            _validate_diagram_dimensions(d)


class TestValidateTargetBetti:
    def test_valid(self):
        from pynerve.diff._loss_helpers import _validate_target_betti

        _validate_target_betti(torch.tensor([2, 1]))

    def test_2d_raises(self):
        from pynerve.diff._loss_helpers import _validate_target_betti

        with pytest.raises(ValueError, match="1D"):
            _validate_target_betti(torch.tensor([[2]]))

    def test_empty_raises(self):
        from pynerve.diff._loss_helpers import _validate_target_betti

        with pytest.raises(ValueError, match="non-empty"):
            _validate_target_betti(torch.tensor([]))

    def test_negative_raises(self):
        from pynerve.diff._loss_helpers import _validate_target_betti

        with pytest.raises(ValueError, match="non-negative"):
            _validate_target_betti(torch.tensor([2, -1]))

    def test_non_finite_raises(self):
        from pynerve.diff._loss_helpers import _validate_target_betti

        with pytest.raises(ValueError):
            _validate_target_betti(torch.tensor([float("inf"), 1]))


class TestValidateDiagramSequence:
    def test_valid_list(self):
        from pynerve.diff._loss_helpers import _validate_diagram_sequence

        d = [torch.tensor([[0.0, 1.0]])]
        _validate_diagram_sequence(d, "diagrams")

    def test_empty_list_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_sequence

        with pytest.raises(ValueError, match="non-empty"):
            _validate_diagram_sequence([], "diagrams")

    def test_not_sequence_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_sequence

        with pytest.raises(TypeError, match="sequence"):
            _validate_diagram_sequence("not a list", "diagrams")

    def test_invalid_diagram_in_sequence_raises(self):
        from pynerve.diff._loss_helpers import _validate_diagram_sequence

        with pytest.raises(ValueError, match="finite"):
            _validate_diagram_sequence([torch.tensor([[float("nan"), 1.0]])], "diagrams")


# ---- _diagram_distances ----


class TestPersistenceLossSoftmin:
    def test_basic(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        x = torch.tensor([1.0, 3.0, 2.0])
        r = PersistenceLoss.softmin(x, temperature=1.0)
        assert r.item() < x.min().item()

    def test_dim_arg(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = PersistenceLoss.softmin(x, dim=0, temperature=1.0)
        assert r.shape == (2,)

    def test_gradient_flow(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        r = PersistenceLoss.softmin(x, temperature=1.0)
        r.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_temperature_zero_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        with pytest.raises(ValueError):
            PersistenceLoss.softmin(torch.tensor([1.0, 2.0]), temperature=0.0)

    def test_nan_input_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        with pytest.raises(ValueError, match="finite"):
            PersistenceLoss.softmin(torch.tensor([float("nan"), 1.0]))


class TestPersistenceLossSoftmax:
    def test_sum_to_one(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        x = torch.tensor([1.0, 2.0, 3.0])
        r = PersistenceLoss.softmax(x, temperature=1.0)
        assert r.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_temperature_effect(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        x = torch.tensor([1.0, 3.0])
        r_low = PersistenceLoss.softmax(x, temperature=0.1)
        r_high = PersistenceLoss.softmax(x, temperature=10.0)
        assert r_low[1].item() > r_high[1].item()

    def test_nan_input_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        with pytest.raises(ValueError, match="finite"):
            PersistenceLoss.softmax(torch.tensor([float("nan"), 1.0]))


class TestPersistenceLossWasserstein:
    def test_p1(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        d2 = torch.tensor([[0.1, 1.1], [0.1, 2.1]])
        r = PersistenceLoss.diagram_wasserstein(d1, d2, p=1, temperature=0.01)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_p3(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 2.0]])
        r = PersistenceLoss.diagram_wasserstein(d1, d2, p=3, temperature=0.01)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_3col_diagrams(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        d2 = torch.tensor([[0.1, 1.1, 0.0]])
        r = PersistenceLoss.diagram_wasserstein(d1, d2, temperature=0.01)
        assert r.item() >= 0

    def test_n1_empty_n2_nonempty(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.empty(0, 2)
        d2 = torch.tensor([[0.0, 1.0]])
        r = PersistenceLoss.diagram_wasserstein(d1, d2, temperature=0.01)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_n1_nonempty_n2_empty(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.empty(0, 2)
        r = PersistenceLoss.diagram_wasserstein(d1, d2, temperature=0.01)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_p_negative_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            PersistenceLoss.diagram_wasserstein(d, d, p=-1.0)

    def test_temperature_negative_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            PersistenceLoss.diagram_wasserstein(d, d, temperature=-0.1)


class TestPersistenceLossBottleneck:
    def test_n1_empty_n2_nonempty(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.empty(0, 2)
        d2 = torch.tensor([[0.0, 1.0]])
        r = PersistenceLoss.diagram_bottleneck(d1, d2, temperature=0.001)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_n1_nonempty_n2_empty(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.empty(0, 2)
        r = PersistenceLoss.diagram_bottleneck(d1, d2, temperature=0.001)
        assert r.item() > 0
        assert torch.isfinite(r)

    def test_temperature_negative_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            PersistenceLoss.diagram_bottleneck(d, d, temperature=-0.01)

    def test_gradient_flow(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 3.0]], requires_grad=True)
        r = PersistenceLoss.diagram_bottleneck(d, d.detach(), temperature=0.1)
        r.backward()
        assert d.grad is not None


class TestPersistenceLossKernel:
    def test_empty_empty(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.empty(0, 2)
        d2 = torch.empty(0, 2)
        k = PersistenceLoss.persistence_kernel(d1, d2, sigma=1.0)
        assert k.item() == pytest.approx(0.0, abs=1e-10)

    def test_3col_diagrams(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        d2 = torch.tensor([[0.1, 1.1, 0.0]])
        k = PersistenceLoss.persistence_kernel(d1, d2, sigma=1.0)
        assert k.item() > 0
        assert torch.isfinite(k)

    def test_sigma_negative_raises(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            PersistenceLoss.persistence_kernel(d, d, sigma=-1.0)

    def test_gradient_flow(self):
        from pynerve.diff._diagram_distances import PersistenceLoss

        d = torch.tensor([[0.0, 1.0]], requires_grad=True)
        k = PersistenceLoss.persistence_kernel(d, d.detach(), sigma=1.0)
        k.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


# ---- _loss_modules ----


class TestBettiNumberLossValidation:
    def test_threshold_negative_raises(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        with pytest.raises(ValueError):
            BettiNumberLoss(threshold=-0.1)

    def test_temperature_negative_raises(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        with pytest.raises(ValueError):
            BettiNumberLoss(temperature=0.0)

    def test_is_nn_module(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        assert isinstance(BettiNumberLoss(), nn.Module)

    def test_soft_step_nan_input_raises(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        loss = BettiNumberLoss()
        with pytest.raises(ValueError, match="finite"):
            loss.soft_step(torch.tensor([float("nan")]))

    def test_forward_multi_dim_betti(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.01)
        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.5, 1.0]])
        target = torch.tensor([2.0, 1.0])
        val = loss(d, target)
        assert torch.isfinite(val)

    def test_forward_dim_oob(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.01)
        d = torch.tensor([[0.0, 1.0, 3.0]])
        target = torch.tensor([1.0, 0.0])
        val = loss(d, target)
        assert torch.isfinite(val)


class TestDiagramComplexityLossValidation:
    def test_unknown_measure_raises(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        with pytest.raises(ValueError, match="unknown"):
            DiagramComplexityLoss(measure="invalid")

    def test_is_nn_module(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        assert isinstance(DiagramComplexityLoss(), nn.Module)

    def test_all_measures_coverage(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        for measure in (
            "total_persistence",
            "persistence_entropy",
            "num_features",
            "max_persistence",
        ):
            loss = DiagramComplexityLoss(measure=measure)
            val = loss(d)
            assert torch.isfinite(val)

    def test_persistence_entropy_zero_total(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="persistence_entropy")
        d = torch.tensor([[0.5, 0.5]])
        val = loss(d)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_num_features_all_below_threshold(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="num_features")
        d = torch.tensor([[0.0, 0.05]])
        val = loss(d)
        assert val.item() == 0

    def test_max_persistence_single(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="max_persistence")
        d = torch.tensor([[0.0, 5.0]])
        val = loss(d)
        assert val.item() == pytest.approx(5.0, abs=1e-10)


class TestTopologicalComplexityLossAlias:
    def test_alias_is_same_class(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss, TopologicalComplexityLoss

        assert TopologicalComplexityLoss is DiagramComplexityLoss


class TestStabilityLoss:
    def test_epsilon_negative_raises(self):
        from pynerve.diff._loss_modules import StabilityLoss

        with pytest.raises(ValueError):
            StabilityLoss(epsilon=-0.01)

    def test_num_samples_zero_raises(self):
        from pynerve.diff._loss_modules import StabilityLoss

        with pytest.raises(ValueError):
            StabilityLoss(num_samples=0)

    def test_is_nn_module(self):
        from pynerve.diff._loss_modules import StabilityLoss

        assert isinstance(StabilityLoss(), nn.Module)

    def test_forward_persistence_fn_not_callable_raises(self):
        from pynerve.diff._loss_modules import StabilityLoss

        loss = StabilityLoss()
        pts = torch.randn(10, 3)
        with pytest.raises(TypeError, match="callable"):
            loss(pts, "not_callable")

    def test_forward_mismatched_diagram_counts(self):
        from pynerve.diff._loss_modules import StabilityLoss

        loss = StabilityLoss(num_samples=2)
        call_count = 0

        def bad_fn(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [torch.tensor([[0.0, 1.0]])]
            return [torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 2.0]])]

        pts = torch.randn(10, 3)
        with pytest.raises(ValueError, match="mismatched"):
            loss(pts, bad_fn)

    def test_forward_produces_finite_output(self):
        from pynerve.diff._loss_modules import StabilityLoss

        loss = StabilityLoss(epsilon=0.01, num_samples=2)

        def stable_fn(x):
            return [torch.zeros(10, 2) + 1.0]

        pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        val = loss(pts, stable_fn)
        assert torch.isfinite(val)


class TestMultiScaleTopologyLoss:
    def test_empty_scales_raises(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        with pytest.raises(ValueError, match="non-empty"):
            MultiScaleTopologyLoss(scales=())

    def test_non_positive_scale_raises(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        with pytest.raises(ValueError):
            MultiScaleTopologyLoss(scales=(0.1, 0.0))

    def test_is_nn_module(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        assert isinstance(MultiScaleTopologyLoss(), nn.Module)

    def test_mismatched_target_length_raises(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(0.1, 0.5))
        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="match"):
            loss(d, [d])

    def test_forward_all_below_scales(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(1.0,))
        d = torch.tensor([[0.0, 0.5]])
        target = torch.tensor([[0.0, 1.0]])
        val = loss(d, [target])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_forward_produces_finite_output(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(0.1, 0.5))
        d = torch.tensor([[0.0, 2.0], [0.0, 3.0]])
        t1 = torch.tensor([[0.0, 1.0]])
        t2 = torch.tensor([[0.0, 1.5]])
        val = loss(d, [t1, t2])
        assert torch.isfinite(val)

    def test_target_empty_at_scale(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(0.1,))
        d = torch.tensor([[0.0, 2.0]])
        target = torch.empty(0, 2)
        val = loss(d, [target])
        assert val.item() == pytest.approx(0.0, abs=1e-10)


class TestLandscapeLossValidation:
    def test_n_layers_invalid_raises(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        with pytest.raises(ValueError):
            LandscapeLoss(n_layers=0)

    def test_resolution_invalid_raises(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        with pytest.raises(ValueError):
            LandscapeLoss(resolution=0)

    def test_is_nn_module(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        assert isinstance(LandscapeLoss(), nn.Module)

    def test_landscape_method(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        loss = LandscapeLoss(n_layers=3, resolution=20)
        d = torch.tensor([[0.0, 1.0]])
        land = loss.landscape(d)
        assert land.shape == (3, 20)
        assert (land >= -1e-7).all()


# ---- _composite_loss ----


class TestCompositeTopologyLoss:
    def test_weight_validation(self):
        from pynerve.diff._composite_loss import TopologyLoss

        with pytest.raises(ValueError):
            TopologyLoss(wasserstein_weight=-0.1)
        with pytest.raises(ValueError):
            TopologyLoss(betti_weight=-0.1)
        with pytest.raises(ValueError):
            TopologyLoss(complexity_weight=-0.1)
        with pytest.raises(ValueError):
            TopologyLoss(stability_weight=-0.1)

    def test_is_nn_module(self):
        from pynerve.diff._composite_loss import TopologyLoss

        assert isinstance(TopologyLoss(), nn.Module)

    def test_stability_component(self):
        from pynerve.diff._composite_loss import TopologyLoss

        loss = TopologyLoss(
            wasserstein_weight=0.0, betti_weight=0.0, complexity_weight=0.0, stability_weight=1.0
        )
        d = torch.tensor([[0.0, 1.0]])
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def fn(x):
            return [torch.zeros(5, 2) + 1.0]

        result = loss(d, d, points=pts, persistence_fn=fn)
        assert "stability" in result
        assert torch.isfinite(result["stability"])

    def test_zero_weights_produce_empty_total(self):
        from pynerve.diff._composite_loss import TopologyLoss

        loss = TopologyLoss(wasserstein_weight=0.0, betti_weight=0.0, complexity_weight=0.0)
        d = torch.tensor([[0.0, 1.0]])
        result = loss(d, d)
        assert result["total"].item() == pytest.approx(0.0, abs=1e-10)
        assert "wasserstein" not in result

    def test_missing_stability_args_skips_component(self):
        from pynerve.diff._composite_loss import TopologyLoss

        loss = TopologyLoss(
            stability_weight=1.0, wasserstein_weight=0.0, betti_weight=0.0, complexity_weight=0.0
        )
        d = torch.tensor([[0.0, 1.0]])
        result = loss(d, d)
        assert "stability" not in result
        assert result["total"].item() == pytest.approx(0.0, abs=1e-10)

    def test_missing_betti_args_skips_component(self):
        from pynerve.diff._composite_loss import TopologyLoss

        loss = TopologyLoss(betti_weight=1.0, wasserstein_weight=0.0, complexity_weight=0.0)
        d = torch.tensor([[0.0, 1.0]])
        result = loss(d, d)
        assert "betti" not in result
        assert result["total"].item() == pytest.approx(0.0, abs=1e-10)


# ---- _ph_representations ----


class TestComputePersistenceLandscapeEdgeCases:
    def test_n_layers_zero_raises(self):
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="positive"):
            compute_persistence_landscape(d, n_layers=0)

    def test_resolution_zero_raises(self):
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="positive"):
            compute_persistence_landscape(d, resolution=0)

    def test_more_layers_than_features(self):
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        land = compute_persistence_landscape(d, n_layers=5, resolution=10)
        assert land.shape == (5, 10)

    def test_x_max_equals_x_min(self):
        from pynerve.diff._ph_representations import compute_persistence_landscape

        d = torch.tensor([[0.5, 0.5]])
        land = compute_persistence_landscape(d, n_layers=1, resolution=5)
        assert land.shape == (1, 5)
        assert (land == 0).all()


class TestPersistenceImageEdgeCases:
    def test_resolution_zero_raises(self):
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            persistence_image(d, resolution=0)

    def test_sigma_zero_raises(self):
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            persistence_image(d, sigma=0.0)

    def test_sigma_infinite_raises(self):
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            persistence_image(d, sigma=float("inf"))

    def test_single_point_image(self):
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.0, 1.0]])
        img = persistence_image(d, resolution=10, sigma=0.5)
        assert img.sum().item() > 0
        assert torch.isfinite(img).all()

    def test_multiple_points(self):
        from pynerve.diff._ph_representations import persistence_image

        d = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        img = persistence_image(d, resolution=20, sigma=0.3)
        assert img.shape == (20, 20)
        assert (img >= 0).all()


# ---- ph_layer ----


class TestEffectiveMaxRadius:
    def test_finite_value_returned(self):
        from pynerve.diff.ph_layer import _effective_max_radius

        pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        r = _effective_max_radius(pts, 2.0)
        assert r == 2.0

    def test_less_than_two_points(self):
        from pynerve.diff.ph_layer import _effective_max_radius

        pts = torch.tensor([[0.0, 0.0]])
        r = _effective_max_radius(pts, float("inf"))
        assert r == 0.0

    def test_infinite_uses_max_distance(self):
        from pynerve.diff.ph_layer import _effective_max_radius

        pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        r = _effective_max_radius(pts, float("inf"))
        assert r == pytest.approx(5.0, rel=1e-6)


class TestValidatePersistenceInputs:
    def test_non_rips_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="rips"):
            _validate_persistence_inputs(pts, 0, "alpha", {})

    def test_non_3d_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(5, 2)
        with pytest.raises(ValueError, match="batch"):
            _validate_persistence_inputs(pts, 0, "rips", {})

    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="max_dim"):
            _validate_persistence_inputs(pts, -1, "rips", {})

    def test_empty_batch_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.empty(0, 5, 2)
        with pytest.raises(ValueError, match="empty"):
            _validate_persistence_inputs(pts, 0, "rips", {})

    def test_empty_points_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.empty(1, 0, 2)
        with pytest.raises(ValueError, match="empty"):
            _validate_persistence_inputs(pts, 0, "rips", {})

    def test_empty_dim_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.empty(1, 5, 0)
        with pytest.raises(ValueError, match="empty"):
            _validate_persistence_inputs(pts, 0, "rips", {})

    def test_non_finite_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.tensor([[[float("nan"), 0.0]]])
        with pytest.raises(ValueError, match="finite"):
            _validate_persistence_inputs(pts, 0, "rips", {})

    def test_max_radius_zero_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="max_radius"):
            _validate_persistence_inputs(pts, 0, "rips", {"max_radius": 0.0})

    def test_max_radius_nan_raises(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="max_radius"):
            _validate_persistence_inputs(pts, 0, "rips", {"max_radius": float("nan")})

    def test_max_radius_none_defaults_inf(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        max_radius, metric, reduction = _validate_persistence_inputs(
            pts, 0, "rips", {"max_radius": None}
        )
        assert max_radius == float("inf")
        assert metric == "euclidean"
        assert reduction == "clearing"

    def test_custom_metric_and_reduction(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        pts = torch.randn(1, 5, 2)
        max_radius, metric, reduction = _validate_persistence_inputs(
            pts, 0, "rips", {"max_radius": 1.0, "metric": "l2", "reduction": "twist"}
        )
        assert max_radius == 1.0
        assert metric == "l2"
        assert reduction == "twist"


class TestDifferentiableVietorisRips:
    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        with pytest.raises(ValueError):
            DifferentiableVietorisRips(max_dim=-1)

    def test_is_nn_module(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        assert isinstance(DifferentiableVietorisRips(), nn.Module)

    def test_forward_non_3d_raises(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        rips = DifferentiableVietorisRips(max_dim=1)
        with pytest.raises(ValueError, match="batch"):
            rips(torch.randn(5, 3))

    def test_forward_with_max_radius(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        rips = DifferentiableVietorisRips(max_dim=0, max_radius=10.0)
        assert rips.max_radius == 10.0


class TestDifferentiableAlphaComplex:
    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex

        with pytest.raises(ValueError):
            DifferentiableAlphaComplex(max_dim=-1)

    def test_is_nn_module(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex

        assert isinstance(DifferentiableAlphaComplex(), nn.Module)

    def test_forward_raises_runtime_error(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex

        alpha = DifferentiableAlphaComplex(max_dim=2)
        with pytest.raises(RuntimeError, match="differentiable alpha"):
            alpha(torch.randn(1, 5, 2))


class TestDifferentiableCubical:
    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer import DifferentiableCubical

        with pytest.raises(ValueError):
            DifferentiableCubical(max_dim=-1)

    def test_is_nn_module(self):
        from pynerve.diff.ph_layer import DifferentiableCubical

        assert isinstance(DifferentiableCubical(), nn.Module)

    def test_forward_raises_runtime_error(self):
        from pynerve.diff.ph_layer import DifferentiableCubical

        cubical = DifferentiableCubical(max_dim=2)
        with pytest.raises(RuntimeError, match="differentiable cubical"):
            cubical(torch.randn(1, 5, 5))


class TestFiltrationLearningLayerValidation:
    def test_input_dim_zero_raises(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        with pytest.raises(ValueError):
            FiltrationLearningLayer(input_dim=0)

    def test_hidden_dim_zero_raises(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        with pytest.raises(ValueError):
            FiltrationLearningLayer(input_dim=3, hidden_dims=[64, 0])

    def test_default_hidden_dims(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3)
        out = layer(_valid_batch_points(1, 5, 3))
        assert out.shape == (1, 5)

    def test_forward_batched(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=2, hidden_dims=[4])
        pts = _valid_batch_points(3, 5, 2)
        out = layer(pts)
        assert out.shape == (3, 5)
        assert torch.isfinite(out).all()

    def test_is_nn_module(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        assert isinstance(FiltrationLearningLayer(input_dim=2), nn.Module)


class TestLearnableFiltrationPersistenceConstruction:
    def test_is_nn_module(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        assert isinstance(LearnableFiltrationPersistence(input_dim=2), nn.Module)

    def test_parameters_exist(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        model = LearnableFiltrationPersistence(input_dim=3, max_dim=1, hidden_dims=[8])
        params = list(model.parameters())
        assert len(params) > 0


# ---- ph_layer_module ----


class TestDifferentiablePHFunction:
    def test_non_3d_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePHFunction

        pts = torch.randn(5, 2)
        with pytest.raises(ValueError, match="batch"):
            DifferentiablePHFunction.apply(pts, 0, 5.0, "euclidean", "clearing")

    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePHFunction

        pts = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="max_dim"):
            DifferentiablePHFunction.apply(pts, -1, 5.0, "euclidean", "clearing")


class TestDifferentiablePersistentHomology:
    def test_negative_max_dim_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        with pytest.raises(ValueError):
            DifferentiablePersistentHomology(max_dim=-1)

    def test_max_radius_nan_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        with pytest.raises(ValueError, match="max_radius"):
            DifferentiablePersistentHomology(max_radius=float("nan"))

    def test_max_radius_non_positive_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        with pytest.raises(ValueError, match="max_radius"):
            DifferentiablePersistentHomology(max_radius=-1.0)

    def test_is_nn_module(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        assert isinstance(DifferentiablePersistentHomology(), nn.Module)

    def test_forward_non_3d_raises(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        layer = DifferentiablePersistentHomology()
        with pytest.raises(ValueError, match="batch"):
            layer(torch.randn(5, 2))


class TestTopologyLossModuleEdgeCases:
    def test_negative_wasserstein_p_raises(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError):
            TopologyLoss([d], wasserstein_p=0.0)

    def test_empty_diagrams_raises(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        loss_fn = TopologyLoss([torch.tensor([[0.0, 1.0]])])
        with pytest.raises(ValueError, match="non-empty"):
            loss_fn([])

    def test_empty_pred_nonempty_target(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        d_target = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        d_pred = torch.empty(0, 2)
        loss_fn = TopologyLoss([d_target])
        val = loss_fn([d_pred])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_approximate_wasserstein_both_empty(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        loss_fn = TopologyLoss([torch.empty(0, 2)])
        val = loss_fn._approximate_wasserstein(torch.empty(0, 2), torch.empty(0, 2))
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_approximate_wasserstein_one_empty(self):
        from pynerve.diff.ph_layer_module import TopologyLoss

        loss_fn = TopologyLoss([torch.tensor([[0.0, 1.0]])])
        val = loss_fn._approximate_wasserstein(torch.tensor([[0.5, 1.5]]), torch.empty(0, 2))
        assert val.item() > 0
        assert torch.isfinite(val)


class TestTopologyRegularizer:
    def test_empty_diagrams_raises(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        with pytest.raises(ValueError, match="non-empty"):
            topology_regularizer([], [1])

    def test_negative_weight_raises(self):
        from pynerve.diff.ph_layer_module import topology_regularizer

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            topology_regularizer([d], [1], weight=-1.0)


class TestPersistencePenaltyValidation:
    def test_empty_diagrams_raises(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        with pytest.raises(ValueError, match="non-empty"):
            persistence_penalty([], min_persistence=0.1)

    def test_negative_min_persistence_raises(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            persistence_penalty([d], min_persistence=-0.1)

    def test_negative_weight_raises(self):
        from pynerve.diff.ph_layer_module import persistence_penalty

        d = torch.tensor([[0.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            persistence_penalty([d], weight=-1.0)
