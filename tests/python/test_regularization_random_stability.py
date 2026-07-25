"""Targeted tests for regularization, random, stability training, and curriculum modules.

Exercises FeaturePersistenceTracker, AdaptivePersistentDropout, MultiScalePersistentDropout,
StructuredPersistentDropout, CurricularPersistentDropout, PRNGKey, seed/manual_seed/split,
PersistenceStabilityLoss, InterleavingRegularizer, CoherentPerturbationSampler,
MorseRegularizer, BettiConstraintLayer, TopologicalSmoothness, HomotopyRegularizer,
ComplexityMeasure, CurriculumConfig, and complexity computation functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from _test_helpers import make_diag_3d

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestFeaturePersistenceTracker:
    """Covers regularization/persistent_dropout.py — FeaturePersistenceTracker."""

    def test_construct(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=10, momentum=0.9)
        assert tracker.num_features == 10
        assert tracker.momentum == 0.9
        assert tracker.persistence_scores.shape == (10,)

    def test_construct_invalid_num_features(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        with pytest.raises(ValueError, match="positive"):
            FeaturePersistenceTracker(num_features=0)

    def test_construct_invalid_momentum(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        with pytest.raises(ValueError, match="momentum"):
            FeaturePersistenceTracker(num_features=10, momentum=1.5)

    def test_update(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5, momentum=0.5)
        importance = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        tracker.update(importance)
        assert tracker.update_count.item() == 1
        assert tracker.persistence_scores[0].item() == pytest.approx(0.5)

    def test_update_wrong_shape(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5)
        with pytest.raises(ValueError, match="entries"):
            tracker.update(torch.rand(3))

    def test_update_wrong_dim(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5)
        with pytest.raises(ValueError, match="entries"):
            tracker.update(torch.rand(5, 3))

    def test_get_persistence_ranking(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5, momentum=0.0)
        tracker.update(torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0]))
        ranking = tracker.get_persistence_ranking()
        assert ranking[0].item() == 1  # highest score

    def test_get_top_k_persistent(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5, momentum=0.0)
        tracker.update(torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0]))
        top3 = tracker.get_top_k_persistent(3)
        assert top3.shape[0] == 3
        assert 1 in top3.tolist()

    def test_get_top_k_zero(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5)
        top0 = tracker.get_top_k_persistent(0)
        assert top0.shape[0] == 0

    def test_get_top_k_invalid(self):
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker
        tracker = FeaturePersistenceTracker(num_features=5)
        with pytest.raises(ValueError, match="non-negative"):
            tracker.get_top_k_persistent(-1)


class TestAdaptivePersistentDropout:
    """Covers regularization/persistent_dropout.py — AdaptivePersistentDropout."""

    def test_construct(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=10, p_initial=0.5, p_final=0.2)
        assert layer.p_initial == 0.5
        assert layer.p_final == 0.2
        assert layer.current_epoch == 0

    def test_construct_invalid_p(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        with pytest.raises(ValueError, match="p_initial"):
            AdaptivePersistentDropout(num_features=10, p_initial=1.0)

    def test_construct_invalid_p_final(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        with pytest.raises(ValueError, match="p_final"):
            AdaptivePersistentDropout(num_features=10, p_final=-0.1)

    def test_forward_not_training(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        x = torch.rand(3, 4)
        result = layer(x, training=False)
        assert torch.equal(result, x)

    def test_forward_training(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4, p_initial=0.1, p_final=0.1)
        x = torch.rand(3, 4)
        result = layer(x, training=True)
        assert result.shape == x.shape

    def test_forward_wrong_features(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        with pytest.raises(ValueError, match="features"):
            layer(torch.rand(3, 5), training=True)

    def test_update_persistence_1d(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        layer.update_persistence(torch.rand(4))

    def test_update_persistence_2d(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        layer.update_persistence(torch.rand(8, 4))

    def test_update_persistence_wrong_dim(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        with pytest.raises(ValueError, match="features"):
            layer.update_persistence(torch.rand(3))

    def test_step_epoch(self):
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout
        layer = AdaptivePersistentDropout(num_features=4)
        assert layer.current_epoch == 0
        layer.step_epoch()
        assert layer.current_epoch == 1
        layer.step_epoch()
        assert layer.current_epoch == 2


class TestMultiScalePersistentDropout:
    """Covers regularization/persistent_dropout.py — MultiScalePersistentDropout."""

    def test_construct(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        layer = MultiScalePersistentDropout(num_features=10)
        assert len(layer.scales) == 3
        assert len(layer.p_per_scale) == 3

    def test_construct_custom(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        layer = MultiScalePersistentDropout(
            num_features=10, scales=(0.5, 1.0), p_per_scale=(0.3, 0.5)
        )
        assert layer.scales == (0.5, 1.0)

    def test_construct_mismatched_lengths(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        with pytest.raises(ValueError, match="same length"):
            MultiScalePersistentDropout(num_features=10, scales=(0.5,), p_per_scale=(0.3, 0.5))

    def test_construct_empty_scales(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        with pytest.raises(ValueError, match="at least one"):
            MultiScalePersistentDropout(num_features=10, scales=(), p_per_scale=())

    def test_forward_not_training(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        layer = MultiScalePersistentDropout(num_features=4)
        x = torch.rand(3, 4)
        result = layer(x, torch.tensor([0.1, 0.5, 1.0, 0.01]), training=False)
        assert torch.equal(result, x)

    def test_forward_training(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        layer = MultiScalePersistentDropout(num_features=4)
        x = torch.rand(3, 4)
        scales = torch.tensor([0.1, 0.5, 1.0, 0.01])
        result = layer(x, scales, training=True)
        assert result.shape == x.shape

    def test_forward_wrong_shape(self):
        from pynerve.regularization.persistent_dropout import MultiScalePersistentDropout
        layer = MultiScalePersistentDropout(num_features=4)
        with pytest.raises(ValueError, match="shape"):
            layer(torch.rand(3), torch.tensor([0.1, 0.5, 1.0, 0.01]), training=True)


class TestStructuredPersistentDropout:
    """Covers regularization/persistent_dropout.py — StructuredPersistentDropout."""

    def test_construct(self):
        from pynerve.regularization.persistent_dropout import StructuredPersistentDropout
        layer = StructuredPersistentDropout(num_groups=3, features_per_group=4)
        assert layer.num_groups == 3
        assert layer.features_per_group == 4

    def test_forward_not_training(self):
        from pynerve.regularization.persistent_dropout import StructuredPersistentDropout
        layer = StructuredPersistentDropout(num_groups=3, features_per_group=4)
        x = torch.rand(2, 12)
        result = layer(x, training=False)
        assert torch.equal(result, x)

    def test_forward_training(self):
        from pynerve.regularization.persistent_dropout import StructuredPersistentDropout
        layer = StructuredPersistentDropout(num_groups=3, features_per_group=4)
        x = torch.rand(2, 12)
        result = layer(x, training=True)
        assert result.shape == x.shape

    def test_forward_wrong_features(self):
        from pynerve.regularization.persistent_dropout import StructuredPersistentDropout
        layer = StructuredPersistentDropout(num_groups=3, features_per_group=4)
        with pytest.raises(ValueError, match="shape"):
            layer(torch.rand(2, 10), training=True)


class TestCurricularPersistentDropout:
    """Covers regularization/persistent_dropout.py — CurricularPersistentDropout."""

    def test_construct(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=10, warmup_epochs=5, full_epochs=50)
        assert layer.p_start == 0.1
        assert layer.p_end == 0.5
        assert layer.warmup_epochs == 5

    def test_construct_invalid_full_le_warmup(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        with pytest.raises(ValueError, match="full_epochs"):
            CurricularPersistentDropout(num_features=10, warmup_epochs=50, full_epochs=50)

    def test_forward_not_training(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4)
        x = torch.rand(3, 4)
        result = layer(x, training=False)
        assert torch.equal(result, x)

    def test_forward_training_warmup(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4, warmup_epochs=5, full_epochs=50)
        x = torch.rand(3, 4)
        result = layer(x, training=True)
        assert result.shape == x.shape

    def test_forward_training_after_warmup(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4, warmup_epochs=2, full_epochs=10)
        layer.current_epoch = 5
        x = torch.rand(3, 4)
        result = layer(x, training=True)
        assert result.shape == x.shape

    def test_forward_training_after_full(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4, warmup_epochs=2, full_epochs=10)
        layer.current_epoch = 15
        x = torch.rand(3, 4)
        result = layer(x, training=True)
        assert result.shape == x.shape

    def test_step_epoch(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4)
        layer.step_epoch()
        assert layer.current_epoch == 1

    def test_update_ranks(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4)
        layer.update_ranks(torch.tensor([1.0, 5.0, 3.0, 2.0]))
        assert layer.persistence_history is not None

    def test_update_ranks_wrong_shape(self):
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout
        layer = CurricularPersistentDropout(num_features=4)
        with pytest.raises(ValueError, match="entries"):
            layer.update_ranks(torch.tensor([1.0, 2.0]))


class TestPRNGKey:
    """Covers random.py — PRNGKey, seed, manual_seed, split, etc."""

    def test_construct(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        assert k.seed == 42
        assert k.counter == 0

    def test_construct_negative_counter(self):
        from pynerve.random import PRNGKey
        with pytest.raises(Exception, match="counter"):
            PRNGKey(seed=42, counter=-1)

    def test_repr(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42, counter=3)
        assert "42" in repr(k)
        assert "3" in repr(k)

    def test_split(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        keys = k.split(3)
        assert len(keys) == 3
        assert all(isinstance(k, PRNGKey) for k in keys)

    def test_split_default(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        k1, k2 = k.split()
        assert k1.seed != k2.seed

    def test_iter(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        k1, k2 = k
        assert isinstance(k1, PRNGKey)

    def test_normal(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.normal(shape=(5,))
        assert result.shape == (5,)
        assert isinstance(result, np.ndarray)

    def test_normal_as_tensor(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.normal(shape=(3,), as_tensor=True)
        assert torch.is_tensor(result)

    def test_uniform(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.uniform(low=0.0, high=1.0, shape=(10,))
        assert result.shape == (10,)
        assert (result >= 0).all() and (result < 1).all()

    def test_uniform_invalid_bounds(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        with pytest.raises(Exception, match="high"):
            k.uniform(low=1.0, high=0.0)

    def test_randint(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.randint(0, 10, size=5)
        assert isinstance(result, np.ndarray)

    def test_randint_single_arg(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.randint(10)
        # With size=1 (default), numpy returns a 0-d or 1-element array
        assert result is not None

    def test_randint_invalid(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        with pytest.raises(Exception, match="high"):
            k.randint(5, 3)

    def test_choice_int(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.choice(10, size=3)
        assert len(result) == 3

    def test_choice_sequence(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.choice([1, 2, 3, 4, 5], size=2)
        assert len(result) == 2

    def test_choice_no_replace(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.choice(5, size=5, replace=False)
        assert len(np.unique(result)) == 5

    def test_choice_with_weights(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        p = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = k.choice(5, size=3, p=p)
        assert all(r == 2 for r in result)

    def test_choice_empty(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        with pytest.raises(Exception, match="non-empty"):
            k.choice([])

    def test_permutation_int(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        result = k.permutation(10)
        assert len(result) == 10

    def test_permutation_array(self):
        from pynerve.random import PRNGKey
        k = PRNGKey(seed=42)
        arr = np.array([1, 2, 3, 4, 5])
        result = k.permutation(arr)
        assert sorted(result.tolist()) == [1, 2, 3, 4, 5]


class TestGlobalRNG:
    """Covers random.py — seed, manual_seed, key, split, next_key, ReproducibleContext."""

    def test_seed(self):
        from pynerve.random import seed, key
        k = seed(42)
        assert k.seed == 42
        assert key().seed == 42

    def test_seed_none(self):
        from pynerve.random import seed
        k = seed(None)
        assert k.seed is not None

    def test_manual_seed(self):
        from pynerve.random import manual_seed, key
        k = manual_seed(123)
        assert k.seed == 123
        assert key().seed == 123

    def test_key_auto_init(self):
        import pynerve.random as rnd
        rnd._global_key = None
        k = rnd.key()
        assert k is not None

    def test_global_split(self):
        from pynerve.random import seed, split
        seed(42)
        keys = split(3)
        assert len(keys) == 3

    def test_next_key(self):
        from pynerve.random import seed, next_key
        seed(42)
        k = next_key()
        assert isinstance(k, type(seed(42)))

    def test_reproducible_context(self):
        from pynerve.random import seed, ReproducibleContext, key
        seed(999)
        with ReproducibleContext(seed=0) as ctx:
            assert ctx.seed == 0
            assert key().seed == 0
        assert key().seed == 999

    def test_reproducible_alias(self):
        from pynerve.random import reproducible, ReproducibleContext
        assert reproducible is ReproducibleContext


class TestPersistenceStabilityLoss:
    """Covers training/_stability_training.py — PersistenceStabilityLoss."""

    def test_construct(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss
        loss = PersistenceStabilityLoss(stability_weight=0.1, lipschitz_constant=10.0)
        assert loss.stability_weight == 0.1
        assert loss.lipschitz_constant == 10.0

    def test_construct_negative_weight(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss
        with pytest.raises(ValueError, match="non-negative"):
            PersistenceStabilityLoss(stability_weight=-0.1)

    def test_forward(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss
        loss = PersistenceStabilityLoss(stability_weight=0.1, lipschitz_constant=1.0)
        features = torch.rand(4, 10)
        perturbed = features + torch.randn(4, 10) * 0.01
        result = loss.forward(features, perturbed, perturbation_magnitude=0.01)
        assert result >= 0

    def test_forward_shape_mismatch(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss
        loss = PersistenceStabilityLoss()
        with pytest.raises(ValueError, match="matching shapes"):
            loss.forward(torch.rand(4, 10), torch.rand(3, 10), 0.01)

    def test_forward_with_extractor(self):
        from pynerve.training._stability_training import PersistenceStabilityLoss
        extractor = nn.Linear(10, 5)
        loss = PersistenceStabilityLoss(feature_extractor=extractor)
        features = torch.rand(4, 10)
        perturbed = features + torch.randn(4, 10) * 0.01
        result = loss.forward(features, perturbed, 0.01)
        assert result is not None


class TestInterleavingRegularizer:
    """Covers training/_stability_training.py — InterleavingRegularizer."""

    def test_construct(self):
        from pynerve.training._stability_training import InterleavingRegularizer
        reg = InterleavingRegularizer(lambda_reg=0.05)
        assert reg.lambda_reg == 0.05

    def test_construct_negative(self):
        from pynerve.training._stability_training import InterleavingRegularizer
        with pytest.raises(ValueError, match="non-negative"):
            InterleavingRegularizer(lambda_reg=-0.1)

    def test_forward(self):
        from pynerve.training._stability_training import InterleavingRegularizer
        reg = InterleavingRegularizer(lambda_reg=0.1)
        f1 = torch.rand(5, 3)
        f2 = torch.rand(5, 3)
        result = reg.forward(f1, f2)
        assert result >= 0

    def test_forward_shape_mismatch(self):
        from pynerve.training._stability_training import InterleavingRegularizer
        reg = InterleavingRegularizer()
        with pytest.raises(ValueError, match="matching shapes"):
            reg.forward(torch.rand(5, 3), torch.rand(4, 3))


class TestCoherentPerturbationSampler:
    """Covers training/_stability_training.py — CoherentPerturbationSampler."""

    def test_construct_default(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler()
        assert "gaussian" in sampler.noise_types

    def test_construct_custom(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["gaussian"], max_magnitude=0.5)
        assert sampler.noise_types == ["gaussian"]
        assert sampler.max_magnitude == 0.5

    def test_construct_empty(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        with pytest.raises(ValueError, match="non-empty"):
            CoherentPerturbationSampler(noise_types=[])

    def test_construct_unknown_type(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        with pytest.raises(ValueError, match="unknown noise types"):
            CoherentPerturbationSampler(noise_types=["bad_type"])

    def test_sample_perturbation(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(seed=42)
        result = sampler.sample_perturbation((5, 3))
        assert result.shape == (5, 3)

    def test_sample_perturbation_dtype(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(seed=42)
        with pytest.raises(TypeError, match="floating point"):
            sampler.sample_perturbation((5,), dtype=torch.int32)

    def test_apply_perturbation_gaussian(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["gaussian"], seed=42)
        points = torch.rand(10, 3)
        result = sampler.apply_perturbation(points, noise_type="gaussian")
        assert result.shape == points.shape

    def test_apply_perturbation_uniform(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["uniform"], seed=42)
        points = torch.rand(10, 3)
        result = sampler.apply_perturbation(points, noise_type="uniform")
        assert result.shape == points.shape

    def test_apply_perturbation_scale(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["scale"], seed=42)
        points = torch.rand(10, 3)
        result = sampler.apply_perturbation(points, noise_type="scale")
        assert result.shape == points.shape

    def test_apply_perturbation_rotation(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["rotation"], seed=42, max_magnitude=1.0)
        points = torch.rand(10, 3)
        result = sampler.apply_perturbation(points, noise_type="rotation")
        assert result.shape == points.shape

    def test_apply_perturbation_random(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(seed=42)
        points = torch.rand(10, 3)
        result = sampler.apply_perturbation(points)
        assert result.shape == points.shape

    def test_apply_perturbation_not_tensor(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler()
        with pytest.raises(TypeError, match="tensor"):
            sampler.apply_perturbation([1, 2, 3])

    def test_apply_perturbation_unknown_type(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        sampler = CoherentPerturbationSampler(noise_types=["gaussian"])
        with pytest.raises(ValueError, match="unknown noise type"):
            sampler.apply_perturbation(torch.rand(5, 3), noise_type="bad")


class TestMorseRegularizer:
    """Covers regularization/_topology_regularizers.py — MorseRegularizer."""

    def test_construct(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        reg = MorseRegularizer(lambda_critical=0.1, lambda_morse=0.05, critical_threshold=0.01)
        assert reg.lambda_critical == 0.1
        assert reg.threshold == 0.01

    def test_construct_negative(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        with pytest.raises(ValueError, match="non-negative"):
            MorseRegularizer(lambda_critical=-0.1)

    def test_construct_zero_threshold(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        with pytest.raises(ValueError, match="positive"):
            MorseRegularizer(critical_threshold=0)

    def test_forward_no_gradient(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        reg = MorseRegularizer()
        result = reg.forward(torch.rand(10))
        assert result.item() == 0.0

    def test_forward_with_gradient(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        reg = MorseRegularizer(lambda_critical=1.0, lambda_morse=0.5, critical_threshold=0.5)
        fn_vals = torch.rand(10)
        grad_vals = torch.rand(10, 3) * 0.1  # small gradients -> near critical
        result = reg.forward(fn_vals, grad_vals)
        assert result >= 0

    def test_forward_not_tensor(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        reg = MorseRegularizer()
        with pytest.raises(TypeError, match="tensor"):
            reg.forward([1.0, 2.0])

    def test_forward_empty(self):
        from pynerve.regularization._topology_regularizers import MorseRegularizer
        reg = MorseRegularizer()
        with pytest.raises(ValueError, match="non-empty"):
            reg.forward(torch.empty(0))


class TestHomotopyRegularizer:
    """Covers regularization/_topology_regularizers.py — HomotopyRegularizer."""

    def test_construct(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        reg = HomotopyRegularizer(lambda_homotopy=0.01)
        assert reg.lambda_homotopy == 0.01

    def test_construct_negative(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        with pytest.raises(ValueError, match="non-negative"):
            HomotopyRegularizer(lambda_homotopy=-0.1)

    def test_forward(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        reg = HomotopyRegularizer(lambda_homotopy=0.1)
        current = torch.rand(5, 3)
        target = torch.rand(5, 3)
        result = reg.forward(current, target)
        assert result >= 0

    def test_forward_shape_mismatch(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        reg = HomotopyRegularizer()
        with pytest.raises(ValueError, match="same shape"):
            reg.forward(torch.rand(5, 3), torch.rand(4, 3))

    def test_forward_empty(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        reg = HomotopyRegularizer()
        with pytest.raises(ValueError, match="non-empty"):
            reg.forward(torch.empty(0), torch.empty(0))


class TestBettiConstraintLayer:
    """Covers regularization/_topology_regularizers.py — BettiConstraintLayer."""

    def test_construct(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        mock_fn = MagicMock(return_value=make_diag_3d(5))
        layer = BettiConstraintLayer(target_betti=[2, 1], persistence_fn=mock_fn)
        assert layer.lambda_constraint == 0.1

    def test_construct_empty_betti(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        with pytest.raises(ValueError, match="non-empty"):
            BettiConstraintLayer(target_betti=[], persistence_fn=MagicMock())

    def test_construct_negative_betti(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        with pytest.raises(ValueError, match="non-negative"):
            BettiConstraintLayer(target_betti=[-1, 2], persistence_fn=MagicMock())

    def test_construct_not_callable(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        with pytest.raises(TypeError, match="callable"):
            BettiConstraintLayer(target_betti=[1], persistence_fn=None)

    def test_forward(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        mock_fn = MagicMock(return_value=make_diag_3d(5))
        layer = BettiConstraintLayer(target_betti=[2, 1], persistence_fn=mock_fn)
        x = torch.rand(5, 3)
        result_x, result_loss = layer.forward(x)
        assert torch.equal(result_x, x)
        assert result_loss >= 0

    def test_forward_empty_diagram(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        mock_fn = MagicMock(return_value=torch.empty(0, 3))
        layer = BettiConstraintLayer(target_betti=[2, 1], persistence_fn=mock_fn)
        x = torch.rand(5, 3)
        _, loss = layer.forward(x)
        assert loss >= 0


class TestTopologicalSmoothness:
    """Covers regularization/_topology_regularizers.py — TopologicalSmoothness."""

    def test_construct(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        reg = TopologicalSmoothness(lambda_smooth=0.1, neighborhood_size=3)
        assert reg.lambda_smooth == 0.1
        assert reg.neighborhood_size == 3

    def test_construct_negative(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        with pytest.raises(ValueError, match="non-negative"):
            TopologicalSmoothness(lambda_smooth=-0.1)

    def test_forward(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        reg = TopologicalSmoothness(lambda_smooth=0.1, neighborhood_size=2)
        features = torch.rand(4, 5)
        diagrams = [make_diag_3d(3) for _ in range(4)]
        result = reg.forward(features, diagrams)
        assert result >= 0

    def test_forward_single_sample(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        reg = TopologicalSmoothness()
        features = torch.rand(1, 5)
        diagrams = [make_diag_3d(3)]
        result = reg.forward(features, diagrams)
        assert result.item() == 0.0

    def test_forward_wrong_features_dim(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        reg = TopologicalSmoothness()
        with pytest.raises(ValueError, match="shape"):
            reg.forward(torch.rand(5), [make_diag_3d(3) for _ in range(5)])

    def test_forward_mismatched_length(self):
        from pynerve.regularization._topology_regularizers import TopologicalSmoothness
        reg = TopologicalSmoothness()
        with pytest.raises(ValueError, match="length"):
            reg.forward(torch.rand(4, 5), [make_diag_3d(3) for _ in range(3)])


class TestCurriculumConfig:
    """Covers training/curriculum.py — CurriculumConfig and ComplexityMeasure."""

    def test_default_config(self):
        from pynerve.training.curriculum import CurriculumConfig
        cfg = CurriculumConfig()
        assert cfg.num_stages == 5
        assert cfg.schedule == "linear"

    def test_custom_config(self):
        from pynerve.training.curriculum import CurriculumConfig, ComplexityMeasure
        cfg = CurriculumConfig(
            complexity_measure=ComplexityMeasure.NUM_FEATURES,
            num_stages=3,
            schedule="exponential",
        )
        assert cfg.num_stages == 3
        assert cfg.schedule == "exponential"

    def test_invalid_num_stages(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="positive"):
            CurriculumConfig(num_stages=0)

    def test_invalid_schedule(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="schedule"):
            CurriculumConfig(schedule="bad")

    def test_invalid_stage_ratio(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="stage_ratio"):
            CurriculumConfig(stage_ratio=0.0)

    def test_invalid_stage_ratio_high(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="stage_ratio"):
            CurriculumConfig(stage_ratio=1.5)

    def test_negative_threshold(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="non-negative"):
            CurriculumConfig(persistence_threshold=-1.0)

    def test_invalid_warmup(self):
        from pynerve.training.curriculum import CurriculumConfig
        with pytest.raises(ValueError, match="positive"):
            CurriculumConfig(warmup_epochs=0)

    def test_complexity_measure_values(self):
        from pynerve.training.curriculum import ComplexityMeasure
        assert ComplexityMeasure.TOTAL_PERSISTENCE.value == "total_persistence"
        assert ComplexityMeasure.NUM_FEATURES.value == "num_features"
        assert ComplexityMeasure.MAX_PERSISTENCE.value == "max_persistence"
        assert ComplexityMeasure.PERSISTENCE_ENTROPY.value == "persistence_entropy"
        assert ComplexityMeasure.BETTI_TOTAL.value == "betti_total"
        assert ComplexityMeasure.HOMOLOGY_DIMENSION.value == "max_homology_dim"
        assert ComplexityMeasure.MORSE_COMPLEXITY.value == "morse_complexity"


class TestRobustTopologyTraining:
    """Covers training/_stability_training.py — RobustTopologyTraining."""

    def test_construct(self):
        from pynerve.training._stability_training import RobustTopologyTraining
        model = nn.Linear(10, 2)
        persistence_fn = MagicMock(return_value=[make_diag_3d(5)])
        trainer = RobustTopologyTraining(
            model=model, persistence_fn=persistence_fn, stability_weight=0.1, num_perturbations=2
        )
        assert trainer.model is model

    def test_construct_not_module(self):
        from pynerve.training._stability_training import RobustTopologyTraining
        with pytest.raises(TypeError, match="nn.Module"):
            RobustTopologyTraining(model="not_a_model", persistence_fn=lambda x: x)

    def test_construct_not_callable(self):
        from pynerve.training._stability_training import RobustTopologyTraining
        with pytest.raises(TypeError, match="callable"):
            RobustTopologyTraining(model=nn.Linear(10, 2), persistence_fn=None)

    def test_construct_invalid_num_perturbations(self):
        from pynerve.training._stability_training import RobustTopologyTraining
        with pytest.raises(ValueError, match="positive"):
            RobustTopologyTraining(
                model=nn.Linear(10, 2), persistence_fn=lambda x: x, num_perturbations=0
            )

    def test_training_step(self):
        from pynerve.training._stability_training import RobustTopologyTraining

        class DiagramModel(nn.Module):
            """Model that accepts a list of valid persistence diagrams."""
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(15, 2)  # 5 * 3 = 15
            def forward(self, diagrams):
                if isinstance(diagrams, list):
                    flat = diagrams[0].reshape(-1)
                    return self.linear(flat.unsqueeze(0))
                return self.linear(diagrams.reshape(-1).unsqueeze(0))

        model = DiagramModel()
        # persistence_fn returns valid diagrams (birth < death) as a list
        persistence_fn = lambda points: [make_diag_3d(5)]
        trainer = RobustTopologyTraining(
            model=model, persistence_fn=persistence_fn, stability_weight=0.01, num_perturbations=1
        )
        points = torch.rand(5, 3)
        target = torch.rand(1, 2)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = trainer.training_step(points, target, loss_fn, optimizer)
        assert "prediction_loss" in result
        assert "stability_loss" in result
        assert "total_loss" in result
