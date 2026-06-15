from __future__ import annotations

import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_topology_dropout_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.regularization.topology_constraints import (
        PersistentDropout,
        TopologyPreservingDropout,
    )

    torch.manual_seed(0)

    persistent = PersistentDropout(p=0.2, temperature=0.5)
    assert (
        torch.isfinite(persistent(torch.ones(2, 3), torch.tensor([0.1, 0.2, 0.3]), training=True))
        .all()
        .item()
    ), "expected all finite persistent dropout output"
    with pytest.raises(ValidationError, match="p"):
        PersistentDropout(p=float("nan"))
    with pytest.raises(ValidationError, match="temperature"):
        PersistentDropout(temperature=float("nan"))
    with pytest.raises(ValidationError, match="x"):
        persistent(torch.tensor([[float("nan"), 0.0, 0.0]]), training=True)
    with pytest.raises(ValidationError, match="persistence_scores"):
        persistent(
            torch.ones(2, 3),
            torch.tensor([float("nan"), 0.2, 0.3]),
            training=True,
        )

    topology = TopologyPreservingDropout(p=0.5)
    assert torch.isfinite(topology(torch.ones(4, 2), torch.eye(4), training=True)).all().item(), (
        "expected all finite topology-preserving dropout output"
    )
    with pytest.raises(ValidationError, match="p"):
        TopologyPreservingDropout(p=float("inf"))
    with pytest.raises(ValidationError, match="activations"):
        topology(torch.tensor([[float("nan"), 0.0]]), torch.eye(1), training=True)
    with pytest.raises(ValidationError, match="adjacency"):
        topology(
            torch.ones(2, 2),
            torch.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            training=True,
        )
    with pytest.raises(ValueError, match="non-negative"):
        topology(
            torch.ones(2, 2),
            torch.tensor([[0.0, -1.0], [0.0, 0.0]]),
            training=True,
        )


def test_persistent_batch_norm_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.regularization.topology_constraints import PersistentBatchNorm

    norm = PersistentBatchNorm(2)
    assert torch.isfinite(norm(torch.ones(3, 2), torch.ones(3), training=True)).all().item(), (
        "expected all finite persistent batch norm output"
    )
    with pytest.raises(ValidationError, match="num_features"):
        PersistentBatchNorm(float("nan"))
    with pytest.raises(ValidationError, match="eps"):
        PersistentBatchNorm(2, eps=float("nan"))
    with pytest.raises(ValidationError, match="x"):
        norm(torch.tensor([[float("nan"), 0.0]]), training=True)
    with pytest.raises(ValidationError, match="persistence_scores"):
        norm(torch.ones(3, 2), torch.tensor([1.0, float("inf"), 1.0]), training=True)
    with pytest.raises(ValueError, match="non-negative"):
        norm(torch.ones(3, 2), torch.tensor([1.0, -1.0, 1.0]), training=True)
    with torch.no_grad():
        norm.running_var[0] = -1.0
    with pytest.raises(ValueError, match="running_var"):
        norm(torch.ones(3, 2), training=False)


def test_persistent_dropout_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.regularization.persistent_dropout import (
        AdaptivePersistentDropout,
        CurricularPersistentDropout,
        FeaturePersistenceTracker,
        MultiScalePersistentDropout,
        StructuredPersistentDropout,
    )

    torch.manual_seed(0)

    tracker = FeaturePersistenceTracker(2)
    tracker.update(torch.tensor([0.1, 0.2]))
    assert tracker.get_top_k_persistent(1).shape == (1,), (
        f"expected (1,), got {tracker.get_top_k_persistent(1).shape}"
    )
    with pytest.raises(ValidationError, match="momentum"):
        FeaturePersistenceTracker(2, momentum=float("nan"))
    with pytest.raises(ValidationError, match="feature_importance"):
        tracker.update(torch.tensor([float("nan"), 1.0]))
    with pytest.raises(ValueError, match="k"):
        tracker.get_top_k_persistent(-1)

    adaptive = AdaptivePersistentDropout(2, adaptation_epochs=2, min_persistence_to_keep=0.0)
    assert torch.isfinite(adaptive(torch.ones(3, 2))).all().item(), (
        "expected all finite adaptive dropout output"
    )
    adaptive.update_persistence(torch.tensor([[0.3, 0.4], [0.4, 0.3]]))
    with pytest.raises(ValidationError, match="min_persistence_to_keep"):
        AdaptivePersistentDropout(2, min_persistence_to_keep=float("nan"))
    with pytest.raises(ValidationError, match="feature_gradients"):
        adaptive.update_persistence(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValidationError, match="x"):
        adaptive(torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(ValidationError, match="scales"):
        MultiScalePersistentDropout(2, scales=[float("nan")], p_per_scale=[0.5])
    with pytest.raises(ValueError, match="at least one scale"):
        MultiScalePersistentDropout(2, scales=[], p_per_scale=[])
    multiscale = MultiScalePersistentDropout(2, scales=[0.1], p_per_scale=[0.3])
    assert torch.isfinite(multiscale(torch.ones(1, 2), torch.tensor([0.1, 0.2]))).all().item(), (
        "expected all finite multiscale dropout output"
    )
    with pytest.raises(ValidationError, match="feature_scales"):
        multiscale(torch.ones(1, 2), torch.tensor([float("inf"), 0.2]))

    structured = StructuredPersistentDropout(1, 2)
    assert torch.isfinite(structured(torch.ones(1, 2))).all().item(), (
        "expected all finite structured dropout output"
    )
    with pytest.raises(ValidationError, match="p_group"):
        StructuredPersistentDropout(1, 2, p_group=float("nan"))
    with torch.no_grad():
        structured.group_persistence[0] = float("nan")
    with pytest.raises(ValidationError, match="group_persistence"):
        structured(torch.ones(1, 2))

    curricular = CurricularPersistentDropout(2, warmup_epochs=0, full_epochs=1)
    assert torch.isfinite(curricular(torch.ones(1, 2))).all().item(), (
        "expected all finite curricular dropout output"
    )
    with pytest.raises(ValueError, match="warmup_epochs"):
        CurricularPersistentDropout(2, warmup_epochs=float("nan"))
    with pytest.raises(ValidationError, match="feature_importance"):
        curricular.update_ranks(torch.tensor([0.0, float("inf")]))
