"""Persistence-aware dropout layers."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._constants import EPS
from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int


def _validate_probability(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return parsed


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


class FeaturePersistenceTracker(nn.Module):
    """Track running feature-persistence scores."""

    def __init__(self, num_features: int, momentum: float = 0.9):
        """Initialise the feature persistence tracker.

        :param num_features: Number of features to track.
        :param momentum: EMA momentum for updating persistence scores
            (in ``[0, 1]``).
        :raises ValueError: If *num_features* is not positive or
            *momentum* is not in ``[0, 1]``.
        """
        super().__init__()
        num_features = _validate_positive_int(num_features, "num_features")
        momentum = _finite_scalar(momentum, "momentum")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("persistence_scores", torch.zeros(num_features))
        self.persistence_scores: torch.Tensor
        self.register_buffer("update_count", torch.zeros(1))
        self.update_count: torch.Tensor

    def update(self, feature_importance: torch.Tensor) -> None:
        """Update running scores from a feature-importance vector."""
        _validate_finite_tensor(feature_importance, "feature_importance")
        if feature_importance.dim() != 1 or feature_importance.shape[0] != self.num_features:
            raise ValueError(f"feature_importance must have {self.num_features} entries")
        importance = feature_importance.detach().to(self.persistence_scores)
        self.persistence_scores.mul_(self.momentum).add_(importance, alpha=1.0 - self.momentum)
        self.update_count += 1

    def get_persistence_ranking(self) -> torch.Tensor:
        """Return indices of features sorted by descending persistence.

        :returns: Tensor of feature indices ranked from most to least
            persistent.
        """
        return torch.argsort(self.persistence_scores, descending=True)

    def get_top_k_persistent(self, k: int) -> torch.Tensor:
        """Return indices of the *k* most persistent features.

        :param k: Number of top persistent features to select.
        :returns: Tensor of indices for the top *k* features.
        :raises ValueError: If *k* is not a non-negative integer.
        """
        k = _validate_non_negative_int("k", k)
        ranking = self.get_persistence_ranking()
        return ranking[:k]


class AdaptivePersistentDropout(nn.Module):
    """Dropout with a learned feature-persistence schedule."""

    def __init__(
        self,
        num_features: int,
        p_initial: float = 0.5,
        p_final: float = 0.2,
        adaptation_epochs: int = 100,
        min_persistence_to_keep: float = 0.1,
    ):
        """Initialise the adaptive persistent dropout.

        :param num_features: Number of features to apply dropout to.
        :param p_initial: Initial dropout probability
            (in ``[0, 1)``).
        :param p_final: Final dropout probability after adaptation
            (in ``[0, 1)``).
        :param adaptation_epochs: Number of epochs over which
            dropout probability linearly decays.
        :param min_persistence_to_keep: Minimum persistence score
            required for a feature to be exempt from dropout.
        :raises ValueError: If any parameter fails validation.
        """
        super().__init__()
        num_features = _validate_positive_int(num_features, "num_features")
        p_initial = _validate_probability("p_initial", p_initial)
        p_final = _validate_probability("p_final", p_final)
        adaptation_epochs = _validate_positive_int(adaptation_epochs, "adaptation_epochs")
        min_persistence_to_keep = _finite_scalar(min_persistence_to_keep, "min_persistence_to_keep")
        if min_persistence_to_keep < 0:
            raise ValueError("min_persistence_to_keep must be non-negative")

        self.p_initial = p_initial
        self.p_final = p_final
        self.adaptation_epochs = adaptation_epochs
        self.min_persistence = min_persistence_to_keep

        self.tracker = FeaturePersistenceTracker(num_features)
        self.current_epoch = 0
        self.feature_persistence = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply persistence-aware dropout to the input features.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :param training: If ``False``, returns *x* unchanged.
        :returns: Dropout-masked and scaled tensor with the same shape
            as *x*.
        :raises ValueError: If *x* has the wrong feature dimension
            or contains non-finite values.
        """
        if not training:
            return x
        _validate_finite_tensor(x, "x")
        if x.shape[-1] != self.feature_persistence.numel():
            raise ValueError(f"x must have {self.feature_persistence.numel()} features")
        _validate_finite_tensor(self.feature_persistence, "feature_persistence")
        _validate_finite_tensor(self.tracker.persistence_scores, "persistence_scores")

        if self.current_epoch < self.adaptation_epochs:
            alpha = self.current_epoch / self.adaptation_epochs
            p = self.p_initial * (1 - alpha) + self.p_final * alpha
        else:
            p = self.p_final

        keep_probs = torch.sigmoid(self.feature_persistence).to(x.device)
        persistence_scores = self.tracker.persistence_scores.to(x.device)
        persistent_mask = persistence_scores >= self.min_persistence
        keep_probs = torch.where(persistent_mask, keep_probs, keep_probs * (1 - p))
        current_avg_keep = keep_probs.mean()
        if current_avg_keep > 0:
            scale = (1 - p) / current_avg_keep
            keep_probs = (keep_probs * scale).clamp(0, 1)

        mask = torch.bernoulli(keep_probs).to(x.device)
        scale = 1 / (keep_probs.clamp(min=EPS))

        return x * mask * scale

    def update_persistence(self, feature_gradients: torch.Tensor) -> None:
        """Update running persistence scores from feature gradients.

        :param feature_gradients: Gradients or importance values of
            shape ``(num_features,)`` or ``(batch, num_features)``.
            When batched, the L2 norm across the batch is used.
        :raises ValueError: If *feature_gradients* has the wrong
            dimension or contains non-finite values.
        """
        _validate_finite_tensor(feature_gradients, "feature_gradients")
        expected = self.feature_persistence.numel()
        if feature_gradients.dim() == 0 or feature_gradients.shape[-1] != expected:
            raise ValueError(f"feature_gradients must have {expected} features")
        if feature_gradients.dim() == 1:
            grad_norms = feature_gradients.detach().abs()
        else:
            grad_norms = feature_gradients.detach().reshape(-1, expected).norm(dim=0)
        self.tracker.update(grad_norms)

        with torch.no_grad():
            self.feature_persistence.copy_(self.tracker.persistence_scores)

    def step_epoch(self) -> None:
        """Increment the internal epoch counter.

        Called once per training epoch to advance the dropout
        probability schedule.
        """
        self.current_epoch += 1


class MultiScalePersistentDropout(nn.Module):
    """Apply persistence-aware dropout at configured feature scales."""

    def __init__(
        self,
        num_features: int,
        scales: Sequence[float] | None = None,
        p_per_scale: Sequence[float] | None = None,
    ):
        """Initialise the multi-scale persistent dropout.

        :param num_features: Number of features to apply dropout to.
        :param scales: Sequence of scale values (default:
            ``(0.01, 0.1, 0.5)``).
        :param p_per_scale: Dropout probabilities per scale (default:
            ``(0.3, 0.5, 0.7)``).  Must match *scales* length.
        :raises ValueError: If scales and probabilities have
            mismatched lengths or any value fails validation.
        """
        super().__init__()
        num_features = _validate_positive_int(num_features, "num_features")
        scales = tuple((0.01, 0.1, 0.5) if scales is None else scales)
        p_per_scale = tuple((0.3, 0.5, 0.7) if p_per_scale is None else p_per_scale)
        if len(scales) != len(p_per_scale):
            raise ValueError("scales and p_per_scale must have the same length")
        if not scales:
            raise ValueError("at least one scale is required")
        validated_scales = []
        for i, scale in enumerate(scales):
            scale = _finite_scalar(scale, f"scales[{i}]")
            if scale <= 0:
                raise ValueError("scales must be positive")
            validated_scales.append(scale)
        validated_probabilities = []
        for i, probability in enumerate(p_per_scale):
            validated_probabilities.append(_validate_probability(f"p_per_scale[{i}]", probability))

        self.scales = tuple(validated_scales)
        self.p_per_scale = tuple(validated_probabilities)
        self.scale_trackers = nn.ModuleList(
            [FeaturePersistenceTracker(num_features) for _ in scales]
        )

    def forward(
        self, x: torch.Tensor, feature_scales: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Apply scale-specific persistence-aware dropout.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :param feature_scales: Per-feature scale assignment of shape
            ``(num_features,)``.
        :param training: If ``False``, returns *x* unchanged.
        :returns: Dropout-masked tensor with the same shape as *x*.
        :raises ValueError: If *x* or *feature_scales* have wrong shape
            or contain non-finite values.
        """
        if not training:
            return x
        _validate_finite_tensor(x, "x")
        _validate_finite_tensor(feature_scales, "feature_scales")
        expected = self.scale_trackers[0].num_features
        if x.dim() != 2 or x.shape[-1] != expected:
            raise ValueError(f"x must have shape (batch_size, {expected})")
        if feature_scales.dim() != 1 or feature_scales.shape[0] != x.shape[-1]:
            raise ValueError("feature_scales must match the feature dimension")
        feature_scales = feature_scales.to(x.device)

        output = x.clone()

        for i, (scale, p) in enumerate(zip(self.scales, self.p_per_scale, strict=True)):
            scale_mask = (feature_scales >= scale * 0.5) & (feature_scales < scale * 1.5)

            if scale_mask.any():
                scale_features = x[:, scale_mask]
                persistence = cast(torch.Tensor, self.scale_trackers[i].persistence_scores)[
                    scale_mask
                ].to(x.device)
                keep_probs = 1 - p * (1 - torch.sigmoid(persistence))
                dropout_p = float((1 - keep_probs.mean()).clamp(0.0, 1.0).item())

                output[:, scale_mask] = F.dropout(scale_features, p=dropout_p, training=training)

        return output


class StructuredPersistentDropout(nn.Module):
    """Dropout over feature groups."""

    def __init__(
        self,
        num_groups: int,
        features_per_group: int,
        p_group: float = 0.3,
        p_within_group: float = 0.5,
    ):
        """Initialise the structured persistent dropout.

        :param num_groups: Number of feature groups.
        :param features_per_group: Number of features in each group.
        :param p_group: Probability of dropping an entire group
            (in ``[0, 1)``).
        :param p_within_group: Probability of dropping a feature
            within a retained group (in ``[0, 1)``).
        :raises ValueError: If any parameter fails validation.
        """
        super().__init__()
        num_groups = _validate_positive_int(num_groups, "num_groups")
        features_per_group = _validate_positive_int(features_per_group, "features_per_group")
        p_group = _validate_probability("p_group", p_group)
        p_within_group = _validate_probability("p_within_group", p_within_group)

        self.num_groups = num_groups
        self.features_per_group = features_per_group
        self.p_group = p_group
        self.p_within = p_within_group
        self.group_persistence = nn.Parameter(torch.zeros(num_groups))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply structured dropout over feature groups.

        :param x: Input tensor of shape
            ``(batch_size, num_groups * features_per_group)``.
        :param training: If ``False``, returns *x* unchanged.
        :returns: Dropout-masked and scaled tensor with the same shape
            as *x*.
        :raises ValueError: If *x* has the wrong shape or contains
            non-finite values.
        """
        if not training:
            return x
        _validate_finite_tensor(x, "x")

        expected_features = self.num_groups * self.features_per_group
        if x.dim() != 2 or x.shape[-1] != expected_features:
            raise ValueError(f"x must have shape (batch_size, {expected_features})")
        _validate_finite_tensor(self.group_persistence, "group_persistence")

        batch_size = x.shape[0]
        x_grouped = x.view(batch_size, self.num_groups, self.features_per_group)

        group_keep_probs = torch.sigmoid(self.group_persistence).to(x.device)
        group_keep_probs = (group_keep_probs * (1 - self.p_group)).clamp(0.1, 0.9)

        group_mask = torch.bernoulli(group_keep_probs).to(x.device)
        group_mask = group_mask.unsqueeze(0).unsqueeze(-1)

        within_mask = torch.bernoulli(
            torch.full(
                (batch_size, self.num_groups, self.features_per_group),
                1 - self.p_within,
                device=x.device,
                dtype=x.dtype,
            )
        )

        mask = group_mask * within_mask
        x_dropped = x_grouped * mask
        expected_keep = (group_keep_probs.mean() * (1 - self.p_within)).clamp(min=EPS)
        scale = 1 / expected_keep

        return x_dropped.view(batch_size, -1) * scale


class CurricularPersistentDropout(nn.Module):
    """Persistence dropout with an epoch curriculum."""

    def __init__(
        self,
        num_features: int,
        p_start: float = 0.1,
        p_end: float = 0.5,
        warmup_epochs: int = 10,
        full_epochs: int = 100,
    ):
        """Initialise the curricular persistent dropout.

        :param num_features: Number of features to apply dropout to.
        :param p_start: Dropout probability at the start of training
            (in ``[0, 1)``).
        :param p_end: Dropout probability at the end of training
            (in ``[0, 1)``).
        :param warmup_epochs: Number of warmup epochs with
            rank-based feature dropout.
        :param full_epochs: Total number of epochs in the curriculum.
        :raises ValueError: If any parameter fails validation or
            *full_epochs* is not greater than *warmup_epochs*.
        """
        super().__init__()
        num_features = _validate_positive_int(num_features, "num_features")
        p_start = _validate_probability("p_start", p_start)
        p_end = _validate_probability("p_end", p_end)
        warmup_epochs = _validate_non_negative_int("warmup_epochs", warmup_epochs)
        full_epochs = _validate_positive_int(full_epochs, "full_epochs")
        if full_epochs <= warmup_epochs:
            raise ValueError("full_epochs must be greater than warmup_epochs")

        self.p_start = p_start
        self.p_end = p_end
        self.warmup_epochs = warmup_epochs
        self.full_epochs = full_epochs

        self.current_epoch = 0
        self.register_buffer("feature_ranks", torch.arange(num_features).float())
        self.feature_ranks: torch.Tensor
        self.register_buffer("persistence_history", torch.zeros(num_features))
        self.persistence_history: torch.Tensor

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply curriculum-scheduled persistent dropout.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :param training: If ``False``, returns *x* unchanged.
        :returns: Dropout-masked and scaled tensor with the same shape
            as *x*.
        :raises ValueError: If *x* has the wrong shape or contains
            non-finite values.
        """
        if not training:
            return x
        _validate_finite_tensor(x, "x")
        expected = self.feature_ranks.numel()
        if x.dim() != 2 or x.shape[-1] != expected:
            raise ValueError(f"x must have shape (batch_size, {expected})")
        _validate_finite_tensor(self.feature_ranks, "feature_ranks")

        feature_ranks = self.feature_ranks.to(x.device)
        if self.current_epoch < self.warmup_epochs and self.warmup_epochs > 0:
            p = self.p_start
            rank_threshold = int(len(feature_ranks) * (1 - self.current_epoch / self.warmup_epochs))

            keep_probs = torch.ones_like(feature_ranks)
            low_rank_mask = feature_ranks > rank_threshold
            keep_probs[low_rank_mask] = 1 - p

        elif self.current_epoch < self.full_epochs:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.full_epochs - self.warmup_epochs
            )
            p = self.p_start + (self.p_end - self.p_start) * progress
            keep_probs = torch.ones_like(feature_ranks) * (1 - p)

        else:
            p = self.p_end
            keep_probs = torch.ones_like(feature_ranks) * (1 - p)

        mask = torch.bernoulli(keep_probs).to(x.device)
        mask = mask.unsqueeze(0).expand_as(x)
        scale = 1 / (keep_probs.mean().clamp(min=EPS))

        return x * mask * scale

    def step_epoch(self) -> None:
        """Increment the internal epoch counter.

        Called once per training epoch to advance the curriculum
        schedule.
        """
        self.current_epoch += 1

    def update_ranks(self, feature_importance: torch.Tensor) -> None:
        """Update feature ranks from the given importance scores.

        :param feature_importance: Importance values of shape
            ``(num_features,)``.
        :raises ValueError: If *feature_importance* has the wrong
            shape or contains non-finite values.
        """
        _validate_finite_tensor(feature_importance, "feature_importance")
        expected = self.persistence_history.numel()
        if feature_importance.dim() != 1 or feature_importance.shape[0] != expected:
            raise ValueError(f"feature_importance must have {expected} entries")
        feature_importance = feature_importance.detach().to(self.persistence_history)
        self.persistence_history = 0.9 * self.persistence_history + 0.1 * feature_importance
        self.feature_ranks = torch.argsort(
            torch.argsort(self.persistence_history, descending=True)
        ).float()


__all__ = [
    "FeaturePersistenceTracker",
    "AdaptivePersistentDropout",
    "MultiScalePersistentDropout",
    "StructuredPersistentDropout",
    "CurricularPersistentDropout",
]
