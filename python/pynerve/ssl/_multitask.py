"""Multi-task self-supervised learning for persistence diagrams."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_embedding as _encode_embedding,
)
from .._torch_diagrams import (
    encode_diagram_rows as _encode_rows,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)
from .._validation import validate_finite_scalar as _finite_scalar
from ._completion import _validate_mask
from ._validation import _validate_ssl_diagram


class MultiTaskTopologySSL(nn.Module):
    """Multi-task self-supervised learning for persistence diagrams.

    Combines diagram completion, Betti-number prediction, and denoising
    into a single model with a shared encoder.  Task weights control the
    relative importance of each loss term.
    """

    def __init__(
        self,
        encoder: nn.Module,
        max_dim: int = 3,
        task_weights: dict[str, float] | None = None,
    ):
        """Initialise the multi-task model.

        :param encoder: Shared encoder module for all tasks.
        :param max_dim: Maximum homology dimension for the Betti-number
            prediction head.
        :param task_weights: Optional dict mapping task names
            (``"completion"``, ``"betti"``, ``"denoising"``) to
            non-negative scalar weights.  Unknown keys raise an error.
        :raises ValueError: If *max_dim* is negative, if an unrecognised
            task key is supplied, or if any weight is negative.
        """
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")

        encoder_dim = _encoder_output_dim(encoder)
        self.encoder = encoder
        self.completion_decoder = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.betti_predictor = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_dim + 1),
        )
        self.denoising_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.task_weights = {
            "completion": 1.0,
            "betti": 0.5,
            "denoising": 1.0,
        }
        if task_weights is not None:
            unknown = set(task_weights) - set(self.task_weights)
            if unknown:
                raise ValueError(f"unknown task weights: {sorted(unknown)}")
            for name, weight in task_weights.items():
                parsed = _finite_scalar(weight, f"task_weights[{name}]")
                if parsed < 0:
                    raise ValueError("task weights must be non-negative")
                task_weights[name] = parsed
            self.task_weights.update(task_weights)

    def forward(self, diagram: torch.Tensor, task: str) -> torch.Tensor:
        """Run the model on a specific task.

        :param diagram: Persistence-diagram tensor of shape
            ``(n_pairs, cols)``.
        :param task: One of ``"completion"``, ``"betti"``, or
            ``"denoising"``.
        :returns: Task-specific tensor output.
        :raises ValueError: If *task* is not one of the recognised task
            names.
        """
        if task == "completion":
            return cast(torch.Tensor, self.completion_decoder(_encode_rows(self.encoder, diagram)))
        if task == "betti":
            return cast(
                torch.Tensor,
                F.softplus(self.betti_predictor(_encode_embedding(self.encoder, diagram))),
            )
        if task == "denoising":
            return cast(torch.Tensor, self.denoising_head(_encode_rows(self.encoder, diagram)))
        raise ValueError(f"Unknown task: {task}")

    def compute_multitask_loss(
        self, diagram: torch.Tensor, tasks_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute the weighted multi-task loss.

        :param diagram: Persistence-diagram tensor of shape
            ``(n_pairs, cols)``.
        :param tasks_data: Dict mapping task names to targets.
            ``"completion"`` requires ``completion_mask`` and
            ``completion`` tensors.  ``"betti"`` requires a ``betti``
            tensor.  ``"denoising"`` requires a ``denoising`` tensor.
        :returns: Dict mapping task names (and ``"total"``) to their
            scalar loss tensors.
        :raises ValueError: If required data is missing or shapes are
            incompatible.
        """
        _validate_ssl_diagram(diagram)
        losses: dict[str, torch.Tensor] = {}
        total_loss = diagram.new_zeros(())

        if "completion" in tasks_data:
            mask = tasks_data.get("completion_mask")
            if mask is None:
                raise ValueError("completion task requires completion_mask")
            _validate_mask(mask, diagram.shape[0])
            target = tasks_data["completion"]
            pred = self.forward(diagram, "completion")
            if (
                target.dim() != 2
                or target.shape[0] != diagram.shape[0]
                or target.shape[1] < pred.shape[1]
            ):
                raise ValueError("completion target must have shape (n_pairs, at least pred_cols)")
            if not torch.isfinite(target[:, : pred.shape[1]]).all().item():
                raise ValueError("completion target must contain only finite values")
            missing = ~mask
            if missing.any():
                comp_loss = F.mse_loss(pred[missing], target[missing, : pred.shape[1]])
                losses["completion"] = comp_loss
                total_loss = total_loss + self.task_weights["completion"] * comp_loss

        if "betti" in tasks_data:
            pred_betti = self.forward(diagram, "betti")
            target_betti = tasks_data["betti"].to(pred_betti)
            if pred_betti.shape != target_betti.shape:
                raise ValueError("betti target must match predicted shape")
            if not torch.isfinite(target_betti).all().item():
                raise ValueError("betti target must contain only finite values")
            betti_loss = F.mse_loss(pred_betti, target_betti)
            losses["betti"] = betti_loss
            total_loss = total_loss + self.task_weights["betti"] * betti_loss

        if "denoising" in tasks_data:
            clean = tasks_data["denoising"]
            if clean.shape != diagram.shape:
                raise ValueError("denoising target must match diagram shape")
            _validate_ssl_diagram(clean)
            pred = self.forward(diagram, "denoising")
            denoise_loss = F.mse_loss(pred[:, :2], clean[:, :2])
            losses["denoising"] = denoise_loss
            total_loss = total_loss + self.task_weights["denoising"] * denoise_loss

        losses["total"] = total_loss
        return losses
