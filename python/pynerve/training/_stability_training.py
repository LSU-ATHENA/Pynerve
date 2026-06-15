"""Stability losses, perturbations, and training wrappers."""

from __future__ import annotations

from collections.abc import Callable
from math import pi
from numbers import Integral
from typing import Any

import numpy as np

import torch
from torch import nn

from .._constants import EPS
from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int
from ._stability_regularizer import StabilityRegularizer


def _validate_non_negative_scalar(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        raise ValueError("shape must be non-empty")
    parsed: list[int] = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 0:
            raise ValueError("shape entries must be non-negative integers")
        parsed.append(int(dim))
    return tuple(parsed)


def _validate_floating_tensor(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(value, name)
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")


class PersistenceStabilityLoss(nn.Module):
    """Loss that penalizes instability of feature representations under perturbations.

    Computes the mean violation of a Lipschitz condition: the feature-space
    distance between original and perturbed inputs should be bounded by the
    Lipschitz constant times the perturbation magnitude.

    :param stability_weight: Weight applied to the stability term.
    :param lipschitz_constant: Expected Lipschitz constant of the feature extractor.
    :param feature_extractor: Optional module to extract features before comparison.
    """

    def __init__(
        self,
        stability_weight: float = 0.1,
        lipschitz_constant: float = 10.0,
        feature_extractor: nn.Module | None = None,
    ):
        super().__init__()
        stability_weight = _validate_non_negative_scalar("stability_weight", stability_weight)
        lipschitz_constant = _validate_non_negative_scalar("lipschitz_constant", lipschitz_constant)
        self.stability_weight = stability_weight
        self.lipschitz_constant = lipschitz_constant
        self.feature_extractor = feature_extractor

    def forward(
        self,
        features: torch.Tensor,
        perturbed_features: torch.Tensor,
        perturbation_magnitude: float,
    ) -> torch.Tensor:
        """Compute the stability loss.

        :param features: Original feature vectors.
        :param perturbed_features: Perturbed feature vectors.
        :param perturbation_magnitude: Magnitude of the applied perturbation.
        :returns: Scalar stability loss tensor.
        :raises ValueError: If shapes do not match.
        """
        _validate_floating_tensor("features", features)
        _validate_floating_tensor("perturbed_features", perturbed_features)
        if features.shape != perturbed_features.shape:
            raise ValueError("features and perturbed_features must have matching shapes")
        perturbation_magnitude = _validate_non_negative_scalar(
            "perturbation_magnitude", perturbation_magnitude
        )

        if self.feature_extractor is not None:
            features = self.feature_extractor(features)
            perturbed_features = self.feature_extractor(perturbed_features)
            _validate_floating_tensor("features", features)
            _validate_floating_tensor("perturbed_features", perturbed_features)

        feature_diff = torch.norm(features - perturbed_features, dim=-1)
        expected_diff = self.lipschitz_constant * perturbation_magnitude
        violation = torch.relu(feature_diff - expected_diff)
        return self.stability_weight * violation.mean()


class InterleavingRegularizer(nn.Module):
    """Regularizer that penalizes differences between two filtrations.

    Useful for enforcing interleaving-like stability between pairs of
    function values on a shared domain.

    :param lambda_reg: Regularization strength.
    """

    def __init__(self, lambda_reg: float = 0.05):
        super().__init__()
        lambda_reg = _validate_non_negative_scalar("lambda_reg", lambda_reg)
        self.lambda_reg = lambda_reg

    def forward(self, filtration1: torch.Tensor, filtration2: torch.Tensor) -> torch.Tensor:
        """Compute the interleaving regularization loss.

        :param filtration1: First filtration tensor.
        :param filtration2: Second filtration tensor.
        :returns: Scalar regularization loss.
        :raises ValueError: If the filtration shapes do not match.
        """
        _validate_floating_tensor("filtration1", filtration1)
        _validate_floating_tensor("filtration2", filtration2)
        if filtration1.shape != filtration2.shape:
            raise ValueError("filtrations must have matching shapes")
        diff = torch.abs(filtration1 - filtration2)
        return self.lambda_reg * diff.mean()


class CoherentPerturbationSampler:
    """Generate structured perturbations for stability-based training.

    Supports Gaussian noise, uniform noise, multiplicative scaling, and 3D rotations.
    All perturbations maintain coherence by using the same random state.

    :param noise_types: Allowed noise types (``"gaussian"``, ``"uniform"``, ``"scale"``, ``"rotation"``).
    :param max_magnitude: Maximum perturbation magnitude.
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``noise_types`` is empty or contains unknown types.
    """

    def __init__(
        self,
        noise_types: list[str] | None = None,
        max_magnitude: float = 0.1,
        seed: int | None = None,
    ):
        max_magnitude = _validate_non_negative_scalar("max_magnitude", max_magnitude)
        self.noise_types = (
            ["gaussian", "uniform", "scale"] if noise_types is None else list(noise_types)
        )
        if not self.noise_types:
            raise ValueError("noise_types must be non-empty")
        valid = {"gaussian", "uniform", "scale", "rotation"}
        unknown = set(self.noise_types) - valid
        if unknown:
            raise ValueError(f"unknown noise types: {sorted(unknown)}")
        self.max_magnitude = max_magnitude
        self.rng = np.random.default_rng(seed)
        self._torch_generators: dict[str, torch.Generator] = {}

    def _torch_generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._torch_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            seed = int(self.rng.integers(0, 2**63 - 1, dtype=np.int64))
            generator.manual_seed(seed)
            self._torch_generators[key] = generator
        return generator

    def sample_perturbation(
        self,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample a perturbation tensor without applying it.

        :param shape: Output tensor shape.
        :param device: Target device (defaults to ``"cpu"``).
        :param dtype: Floating-point dtype for the perturbation.
        :returns: A perturbation tensor of the given shape.
        :raises TypeError: If ``dtype`` is not floating-point.
        """
        shape = _validate_shape(shape)
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        device = torch.device("cpu") if device is None else torch.device(device)
        generator = self._torch_generator(device)
        noise_type = str(self.rng.choice(self.noise_types))
        magnitude = float(self.rng.uniform(0, self.max_magnitude))

        if noise_type == "gaussian":
            return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * magnitude
        if noise_type == "uniform":
            return (
                (torch.rand(*shape, device=device, dtype=dtype, generator=generator) - 0.5)
                * 2
                * magnitude
            )
        if noise_type == "scale":
            return torch.full(shape, float(magnitude), device=device, dtype=dtype)
        return torch.zeros(*shape, device=device, dtype=dtype)

    def apply_perturbation(
        self, points: torch.Tensor, noise_type: str | None = None
    ) -> torch.Tensor:
        """Apply a perturbation to a point cloud.

        :param points: A tensor of points (floating-point).
        :param noise_type: Specific noise type, or None for random selection.
        :returns: The perturbed point cloud.
        :raises TypeError: If ``points`` is not a tensor or not floating-point.
        :raises ValueError: If ``noise_type`` is unknown.
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a tensor")
        _validate_floating_tensor("points", points)
        if noise_type is None:
            noise_type = str(self.rng.choice(self.noise_types))
        if noise_type not in self.noise_types:
            raise ValueError(f"unknown noise type: {noise_type}")

        generator = self._torch_generator(points.device)
        magnitude = float(self.rng.uniform(0, self.max_magnitude))

        if noise_type == "gaussian":
            noise = (
                torch.randn(
                    points.shape,
                    device=points.device,
                    dtype=points.dtype,
                    generator=generator,
                )
                * magnitude
            )
            return points + noise

        elif noise_type == "uniform":
            noise = (
                (
                    torch.rand(
                        points.shape,
                        device=points.device,
                        dtype=points.dtype,
                        generator=generator,
                    )
                    - 0.5
                )
                * 2
                * magnitude
            )
            return points + noise

        elif noise_type == "scale":
            if points.dim() < 2:
                return points * (1.0 + magnitude)
            center = points.mean(dim=-2, keepdim=True)
            scale = 1.0 + magnitude
            return center + (points - center) * scale

        elif noise_type == "rotation":
            if points.dim() >= 2 and points.shape[-1] == 3:
                axis = torch.randn(
                    3,
                    device=points.device,
                    dtype=points.dtype,
                    generator=generator,
                )
                axis = axis / (axis.norm() + EPS)
                angle = (
                    torch.rand(
                        1,
                        device=points.device,
                        dtype=points.dtype,
                        generator=generator,
                    )
                    * 2
                    * pi
                    * magnitude
                )
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                center = points.mean(dim=-2, keepdim=True)
                centered = points - center
                axis_view = axis.view(*([1] * (centered.dim() - 1)), 3)
                cross = torch.cross(axis_view.expand_as(centered), centered, dim=-1)
                dot = (centered * axis_view.expand_as(centered)).sum(dim=-1, keepdim=True)
                rotated: torch.Tensor = (
                    centered * cos_a
                    + cross * sin_a
                    + axis_view.expand_as(centered) * dot * (1 - cos_a)
                )
                return center + rotated

        return points


class RobustTopologyTraining:
    """High-level wrapper for training models with stability regularization.

    Combines task-specific prediction losses with topological stability
    regularization to produce models that are robust to input perturbations.

    :param model: The PyTorch model to train.
    :param persistence_fn: A callable that maps input points to persistence diagrams.
    :param stability_weight: Weight for the stability regularization term.
    :param num_perturbations: Number of perturbations to use in the regularizer.
    :raises TypeError: If ``model`` is not an ``nn.Module`` or ``persistence_fn`` is not callable.
    :raises ValueError: If ``num_perturbations`` is not positive.
    """

    def __init__(
        self,
        model: nn.Module,
        persistence_fn: Callable[..., Any],
        stability_weight: float = 0.1,
        num_perturbations: int = 3,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not callable(persistence_fn):
            raise TypeError("persistence_fn must be callable")
        stability_weight = _validate_non_negative_scalar("stability_weight", stability_weight)
        num_perturbations = _validate_positive_int(num_perturbations, "num_perturbations")
        self.model = model
        self.persistence_fn = persistence_fn
        self.stability_reg = StabilityRegularizer(
            epsilon=0.01,
            num_perturbations=num_perturbations,
            lambda_reg=stability_weight,
        )
        self.perturbation_sampler = CoherentPerturbationSampler()

    def training_step(
        self,
        points: torch.Tensor,
        target: torch.Tensor,
        prediction_loss_fn: Callable[..., torch.Tensor],
        optimizer: Any,
    ) -> dict[str, float]:
        """Perform one training step with stability regularization.

        :param points: Input point cloud.
        :param target: Ground-truth targets for the prediction loss.
        :param prediction_loss_fn: A callable loss function for predictions.
        :param optimizer: A PyTorch optimizer.
        :returns: Dictionary with ``prediction_loss``, ``stability_loss``, and ``total_loss``.
        :raises TypeError: If ``prediction_loss_fn`` is not callable.
        """
        _validate_floating_tensor("points", points)
        _validate_finite_tensor(target, "target")
        if not callable(prediction_loss_fn):
            raise TypeError("prediction_loss_fn must be callable")
        optimizer.zero_grad(set_to_none=True)

        diagrams = self.persistence_fn(points)
        predictions = self.model(diagrams)
        _validate_finite_tensor(predictions, "predictions")
        pred_loss = prediction_loss_fn(predictions, target)
        _validate_finite_tensor(pred_loss, "prediction_loss")

        stab_loss = self.stability_reg(points, self.persistence_fn)
        total_loss = pred_loss + stab_loss

        total_loss.backward()
        optimizer.step()

        return {
            "prediction_loss": pred_loss.item(),
            "stability_loss": stab_loss.item(),
            "total_loss": total_loss.item(),
        }


__all__ = [
    "PersistenceStabilityLoss",
    "InterleavingRegularizer",
    "CoherentPerturbationSampler",
    "RobustTopologyTraining",
]
