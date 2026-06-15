from __future__ import annotations

import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_stability_regularizer_rejects_nonfinite_runtime_inputs(torch) -> None:
    from pynerve.training.stability_reg import StabilityRegularizer

    diagram = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=torch.float32)
    regularizer = StabilityRegularizer(epsilon=0.0, num_perturbations=1)

    loss = regularizer(
        torch.ones(2, 2),
        lambda _points: [diagram],
    )
    assert torch.isfinite(loss).item(), "expected finite stability loss"
    assert regularizer.wasserstein_distance([diagram], [diagram]).item() == 0.0, (
        f"expected 0.0, got {regularizer.wasserstein_distance([diagram], [diagram]).item()}"
    )
    with pytest.raises(ValueError, match="epsilon"):
        StabilityRegularizer(epsilon=float("nan"))
    with pytest.raises(ValueError, match="lambda_reg"):
        StabilityRegularizer(lambda_reg=float("inf"))
    with pytest.raises(ValueError, match="perturbation_magnitude"):
        regularizer.compute_theoretical_bound(float("nan"), 2)
    with pytest.raises(ValueError, match="n_points"):
        regularizer.compute_theoretical_bound(0.1, 0)
    with pytest.raises(ValueError, match="points"):
        regularizer(torch.tensor([float("nan")]), lambda _points: [diagram])
    with pytest.raises(ValueError, match="birth"):
        regularizer.wasserstein_distance(
            [torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)],
            [diagram],
        )
    with pytest.raises(ValueError, match="deaths"):
        regularizer.l2_diagram_distance(
            [torch.tensor([[1.0, 0.0]], dtype=torch.float32)],
            [diagram],
        )


def test_topology_regularizers_reject_nonfinite_inputs(torch) -> None:
    from pynerve.regularization.topology_constraints import (
        BettiConstraintLayer,
        HomotopyRegularizer,
        MorseRegularizer,
        TopologicalSmoothness,
    )

    diagram = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=torch.float32)
    features = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    assert torch.isfinite(MorseRegularizer()(torch.ones(2), torch.ones(2, 1))).item(), (
        "expected finite Morse regularizer output"
    )
    betti = BettiConstraintLayer([1, 1], lambda _x: diagram)
    _, betti_loss = betti(torch.ones(2, 1))
    assert torch.isfinite(betti_loss).item(), "expected finite Betti loss"
    smooth = TopologicalSmoothness(neighborhood_size=1)
    assert torch.isfinite(smooth(features, [diagram, diagram])).item(), (
        "expected finite topological smoothness"
    )
    assert torch.isfinite(HomotopyRegularizer()(features, features)).item(), (
        "expected finite homotopy regularizer output"
    )
    with pytest.raises(ValidationError, match="lambda_critical"):
        MorseRegularizer(lambda_critical=float("nan"))
    with pytest.raises(ValueError, match="function_values"):
        MorseRegularizer()(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="target_betti"):
        BettiConstraintLayer([float("nan")], lambda _x: diagram)
    with pytest.raises(ValueError, match="birth"):
        BettiConstraintLayer([1], lambda _x: torch.tensor([[float("nan"), 1.0, 0.0]]))(
            torch.ones(1, 1)
        )
    with pytest.raises(ValidationError, match="lambda_smooth"):
        TopologicalSmoothness(lambda_smooth=float("inf"))
    with pytest.raises(ValueError, match="features"):
        smooth(torch.tensor([[float("nan"), 0.0], [1.0, 0.0]]), [diagram, diagram])
    with pytest.raises(ValueError, match="deaths"):
        smooth._diagram_distance(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            diagram,
        )
    with pytest.raises(ValidationError, match="lambda_homotopy"):
        HomotopyRegularizer(lambda_homotopy=float("nan"))
    with pytest.raises(ValueError, match="outputs"):
        HomotopyRegularizer()(torch.tensor([float("nan")]), torch.tensor([0.0]))
