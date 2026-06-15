"""Completion task for persistence diagrams."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import encode_diagram_rows as _encode_rows
from .._validation import validate_finite_scalar as _finite_scalar
from ._validation import _validate_ssl_diagram


def _validate_mask(mask: torch.Tensor, n_pairs: int) -> None:
    if mask.shape != (n_pairs,) or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor with shape (n_pairs,)")


class TopologyCompletionModel(nn.Module):
    """Predicts missing entries in partially observed persistence diagrams.

    Takes a diagram with some pairs masked as missing and reconstructs the
    complete diagram using an encoder-decoder architecture.  The model learns
    topological structure from the unmasked portion and infers the hidden
    birth/death coordinates.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, completion_threshold: float = 0.1):
        """Initialise the completion model.

        :param encoder: Module that encodes a persistence-diagram tensor
            of shape ``(n_pairs, cols)`` into a latent representation.
        :param decoder: Module that decodes the latent representation back
            to diagram-space coordinates.
        :param completion_threshold: Non-negative threshold used by
            downstream logic to gate which predictions are trusted.
        :raises ValueError: If *completion_threshold* is negative.
        """
        super().__init__()
        completion_threshold = _finite_scalar(completion_threshold, "completion_threshold")
        if completion_threshold < 0:
            raise ValueError("completion_threshold must be non-negative")

        self.encoder = encoder
        self.decoder = decoder
        self.threshold = completion_threshold

    def forward(
        self, partial_diagram: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete a partially observed persistence diagram.

        :param partial_diagram: Input diagram tensor of shape
            ``(n_pairs, cols)``.  Pairs flagged by *mask* need not be
            meaningful.
        :param mask: Boolean tensor of shape ``(n_pairs,)`` where
            ``True`` indicates an observed (trustworthy) pair and
            ``False`` indicates a pair to be completed.
        :returns: A tuple ``(completed_diagram, encoding)``.
            *completed_diagram* has the same shape as *partial_diagram*
            with masked entries replaced by decoder predictions.
            *encoding* is the latent representation.
        :raises ValueError: If the decoder output shape is incompatible
            with the partial diagram.
        """
        _validate_ssl_diagram(partial_diagram)
        _validate_mask(mask, partial_diagram.shape[0])

        encoding = _encode_rows(self.encoder, partial_diagram)
        completed = self.decoder(encoding)
        if (
            completed.dim() != 2
            or completed.shape[0] != partial_diagram.shape[0]
            or completed.shape[1] < partial_diagram.shape[1]
        ):
            raise ValueError("decoder must return shape (n_pairs, at least diagram_cols)")

        result = partial_diagram.clone()
        result[~mask] = completed[~mask, : partial_diagram.shape[1]]
        return result, encoding

    def compute_loss(
        self, partial: torch.Tensor, mask: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the mean squared error over the masked pairs.

        :param partial: Partially-observed diagram tensor of shape
            ``(n_pairs, cols)``.
        :param mask: Boolean tensor of shape ``(n_pairs,)`` where
            ``True`` indicates an observed pair.
        :param target: Ground-truth complete diagram with the same shape
            as *partial*.
        :returns: Scalar MSE loss computed only over the birth/death
            columns of the masked (reconstructed) pairs, or zero if no
            pairs are masked.
        :raises ValueError: If *target* does not match the shape of
            *partial*.
        """
        if target.shape != partial.shape:
            raise ValueError("target must have the same shape as partial")
        _validate_ssl_diagram(target)

        completed, _ = self.forward(partial, mask)
        missing = ~mask
        if missing.any():
            return F.mse_loss(completed[missing, :2], target[missing, :2])
        return partial.new_zeros(())
