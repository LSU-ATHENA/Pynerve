from __future__ import annotations

import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_nn_topological_losses_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.nn.topo_regularization import (
        DiagramMatchingLoss,
        PersistenceEntropyLoss,
        TopologicalComplexityLoss,
        TopologicalRegularizationLoss,
    )

    diagram = torch.tensor([[0.0, 0.5], [0.0, float("inf")]], dtype=torch.float32)
    device = torch.device("cpu")
    dtype = torch.float32

    regularizer = TopologicalRegularizationLoss(min_persistence=0.1, target_betti=[1], max_dim=0)
    assert torch.isfinite(
        regularizer._compute_loss_from_diagrams([diagram], device, dtype)
    ).item(), "expected finite regularization loss"
    with pytest.raises(ValidationError, match="min_persistence"):
        TopologicalRegularizationLoss(min_persistence=float("nan"))
    with pytest.raises(ValueError, match="target_betti"):
        TopologicalRegularizationLoss(target_betti=[float("nan")])
    with pytest.raises(ValidationError, match="hidden_states"):
        regularizer(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="births"):
        regularizer._compute_loss_from_diagrams(
            [torch.tensor([[float("nan"), 1.0]])], device, dtype
        )

    entropy = PersistenceEntropyLoss()
    assert torch.isfinite(entropy._compute_entropy([diagram], device, dtype)).item(), (
        "expected finite entropy"
    )
    with pytest.raises(ValidationError, match="target_entropy"):
        PersistenceEntropyLoss(target_entropy=float("inf"))

    complexity = TopologicalComplexityLoss(min_features=0, max_features=2)
    assert complexity._count_features([diagram]) == 1, (
        f"expected 1, got {complexity._count_features([diagram])}"
    )
    with pytest.raises(ValueError, match="min_features"):
        TopologicalComplexityLoss(min_features=float("nan"))
    with pytest.raises(ValueError, match="deaths"):
        complexity._count_features([torch.tensor([[1.0, 0.0]])])

    matching = DiagramMatchingLoss()
    target = torch.tensor([[0.0, 0.25]], dtype=torch.float32)
    assert torch.isfinite(matching([[diagram]], [[target]])).item(), "expected finite matching loss"
    with pytest.raises(ValidationError, match="p"):
        DiagramMatchingLoss(p=float("nan"))
    with pytest.raises(ValueError, match="target"):
        matching([[target]], [[torch.tensor([[float("nan"), 1.0]])]])
    with pytest.raises(ValueError, match="non-empty"):
        matching([[]], [[]])


def test_diff_topology_losses_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.diff.topology_loss import (
        BettiNumberLoss,
        LandscapeLoss,
        MultiScaleTopologyLoss,
        PersistenceLoss,
        StabilityLoss,
        TopologyLoss,
    )
    from pynerve.diff.topology_loss import (
        TopologicalComplexityLoss as DiffTopologicalComplexityLoss,
    )

    diagram2 = torch.tensor([[0.0, 0.5], [0.1, 0.4]], dtype=torch.float32)
    diagram3 = torch.tensor([[0.0, 0.5, 0.0], [0.1, 0.4, 1.0]], dtype=torch.float32)

    assert torch.isfinite(PersistenceLoss.diagram_wasserstein(diagram2, diagram2)).item(), (
        "expected finite Wasserstein distance"
    )
    assert torch.isfinite(PersistenceLoss.diagram_bottleneck(diagram2, diagram2)).item(), (
        "expected finite bottleneck distance"
    )
    assert torch.isfinite(PersistenceLoss.persistence_kernel(diagram2, diagram2)).item(), (
        "expected finite persistence kernel"
    )
    with pytest.raises(ValidationError, match="temperature"):
        PersistenceLoss.softmax(torch.ones(2), temperature=float("nan"))
    with pytest.raises(ValidationError, match="sigma"):
        PersistenceLoss.persistence_kernel(diagram2, diagram2, sigma=float("inf"))
    with pytest.raises(ValidationError, match="birth"):
        PersistenceLoss.diagram_wasserstein(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32), diagram2
        )

    betti = BettiNumberLoss()
    assert torch.isfinite(betti(diagram3, torch.tensor([1.0, 1.0]))).item(), (
        "expected finite Betti number loss"
    )
    with pytest.raises(ValidationError, match="threshold"):
        BettiNumberLoss(threshold=float("nan"))
    with pytest.raises(ValidationError, match="target_betti"):
        betti(diagram3, torch.tensor([1.0, float("inf")]))
    with pytest.raises(ValueError, match="integer"):
        betti(torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32), torch.ones(1))

    complexity = DiffTopologicalComplexityLoss("max_persistence")
    assert torch.isfinite(complexity(diagram2)).item(), "expected finite complexity loss"
    with pytest.raises(ValidationError, match="deaths"):
        complexity(torch.tensor([[1.0, 0.0]], dtype=torch.float32))

    stability = StabilityLoss(epsilon=0.0, num_samples=1)
    assert torch.isfinite(stability(torch.ones(2, 2), lambda _pts: [diagram2])).item(), (
        "expected finite stability loss"
    )
    with pytest.raises(ValidationError, match="epsilon"):
        StabilityLoss(epsilon=float("nan"))
    with pytest.raises(ValidationError, match="points"):
        stability(torch.tensor([[float("nan"), 0.0]]), lambda _pts: [diagram2])
    with pytest.raises(ValueError, match="orig_diagrams"):
        stability(torch.ones(2, 2), lambda _pts: [])

    multiscale = MultiScaleTopologyLoss(scales=(0.1,))
    assert torch.isfinite(multiscale(diagram2, [diagram2])).item(), (
        "expected finite multiscale topology loss"
    )
    with pytest.raises(ValidationError, match="scales"):
        MultiScaleTopologyLoss(scales=(float("nan"),))
    with pytest.raises(ValueError, match="target_diagrams"):
        multiscale(diagram2, [])

    with pytest.raises(ValidationError, match="n_layers"):
        LandscapeLoss(n_layers=0)

    topology = TopologyLoss(wasserstein_weight=1.0, betti_weight=0.1, complexity_weight=0.01)
    losses = topology(diagram3, diagram2, target_betti=torch.tensor([1.0, 1.0]))
    assert torch.isfinite(losses["total"]).item(), "expected finite total topology loss"
    with pytest.raises(ValidationError, match="wasserstein_weight"):
        TopologyLoss(wasserstein_weight=float("nan"))


def test_stability_training_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.training.stability_reg import (
        CoherentPerturbationSampler,
        InterleavingRegularizer,
        PersistenceStabilityLoss,
        RobustTopologyTraining,
    )

    stability = PersistenceStabilityLoss(stability_weight=0.1, lipschitz_constant=1.0)
    assert torch.isfinite(stability(torch.zeros(2, 3), torch.ones(2, 3) * 0.1, 0.2)).item(), (
        "expected finite persistence stability loss"
    )
    with pytest.raises(ValidationError, match="stability_weight"):
        PersistenceStabilityLoss(stability_weight=float("nan"))
    with pytest.raises(ValidationError, match="features"):
        stability(torch.tensor([[float("nan")]]), torch.zeros(1, 1), 0.1)
    with pytest.raises(ValidationError, match="perturbation_magnitude"):
        stability(torch.zeros(1, 1), torch.zeros(1, 1), float("inf"))

    interleaving = InterleavingRegularizer()
    assert torch.isfinite(interleaving(torch.zeros(2), torch.ones(2))).item(), (
        "expected finite interleaving regularizer output"
    )
    with pytest.raises(ValidationError, match="lambda_reg"):
        InterleavingRegularizer(lambda_reg=float("nan"))
    with pytest.raises(ValidationError, match="filtration2"):
        interleaving(torch.zeros(1), torch.tensor([float("nan")]))

    sampler = CoherentPerturbationSampler(["scale"], max_magnitude=0.0, seed=0)
    assert sampler.sample_perturbation((2, 3)).shape == (2, 3), (
        f"expected (2, 3), got {sampler.sample_perturbation((2, 3)).shape}"
    )
    assert torch.equal(sampler.apply_perturbation(torch.ones(2, 3), "scale"), torch.ones(2, 3)), (
        "expected perturbation to preserve values at max_magnitude=0"
    )
    with pytest.raises(ValidationError, match="max_magnitude"):
        CoherentPerturbationSampler(max_magnitude=float("nan"))
    with pytest.raises(ValueError, match="noise_types"):
        CoherentPerturbationSampler([])
    with pytest.raises(ValueError, match="shape"):
        sampler.sample_perturbation((-1, 2))
    with pytest.raises(TypeError, match="dtype"):
        sampler.sample_perturbation((1,), dtype=torch.int64)
    with pytest.raises(ValidationError, match="points"):
        sampler.apply_perturbation(torch.tensor([[float("nan"), 0.0]]), "scale")

    with pytest.raises(TypeError, match="model"):
        RobustTopologyTraining(object(), lambda _points: [], stability_weight=0.1)
    with pytest.raises(ValidationError, match="stability_weight"):
        RobustTopologyTraining(
            torch.nn.Identity(), lambda _points: [], stability_weight=float("nan")
        )
    with pytest.raises(ValidationError, match="num_perturbations"):
        RobustTopologyTraining(torch.nn.Identity(), lambda _points: [], num_perturbations=0)
