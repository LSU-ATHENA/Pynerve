"""Filtration ordering learning task."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_rows as _encode_rows,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)


class FiltrationOrderingTask(nn.Module):
    """Learns the filtration ordering of a set of simplices.

    Given a shuffled set of simplices and per-simplex feature vectors, the
    model predicts a scalar score for each simplex.  The relative ordering
    of these scores is trained to match the true filtration order via a
    pairwise ranking loss.
    """

    def __init__(self, encoder: nn.Module, max_simplices: int = 100):
        """Initialise the filtration-ordering task.

        :param encoder: Module that encodes an augmented simplex tensor
            into per-simplex embeddings.
        :param max_simplices: Maximum number of simplices allowed in a
            single batch element.
        :raises ValueError: If *max_simplices* is not positive.
        """
        super().__init__()
        if max_simplices <= 0:
            raise ValueError("max_simplices must be positive")

        self.encoder = encoder
        self.max_simplices = max_simplices
        self.order_predictor = nn.Sequential(
            nn.Linear(_encoder_output_dim(encoder), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, shuffled_simplices: torch.Tensor, simplex_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict a filtration score for each simplex.

        :param shuffled_simplices: Tensor of shape ``(n_simplices, d)``
            representing the shuffled simplices.
        :param simplex_features: Tensor of shape ``(n_simplices, f)``
            with per-simplex feature vectors.
        :returns: Float tensor of shape ``(n_simplices,)`` containing a
            scalar ordering score per simplex.
        :raises ValueError: If inputs have incompatible shapes, exceed
            *max_simplices*, or contain non-finite values.
        """
        if shuffled_simplices.dim() != 2 or simplex_features.dim() != 2:
            raise ValueError("shuffled_simplices and simplex_features must be 2D")
        if shuffled_simplices.shape[0] != simplex_features.shape[0]:
            raise ValueError("simplex inputs must have the same number of rows")
        if shuffled_simplices.shape[0] > self.max_simplices:
            raise ValueError("number of simplices exceeds max_simplices")
        if (
            not torch.isfinite(shuffled_simplices.to(dtype=torch.float32)).all().item()
            or not torch.isfinite(simplex_features).all().item()
        ):
            raise ValueError("simplex inputs must contain only finite values")

        encoder_input = torch.cat(
            [
                shuffled_simplices.to(dtype=simplex_features.dtype),
                simplex_features.to(device=shuffled_simplices.device),
            ],
            dim=-1,
        )
        row_embeddings = _encode_rows(self.encoder, encoder_input)
        return cast(torch.Tensor, self.order_predictor(row_embeddings).squeeze(-1))

    def compute_loss(
        self, shuffled: torch.Tensor, features: torch.Tensor, true_order: torch.Tensor
    ) -> torch.Tensor:
        """Compute a pairwise ranking loss against the true filtration order.

        :param shuffled: Shuffled simplex tensor of shape
            ``(n_simplices, d)``.
        :param features: Per-simplex feature tensor of shape
            ``(n_simplices, f)``.
        :param true_order: Ground-truth ordering scores of shape
            ``(n_simplices,)``.
        :returns: Scalar margin-based ranking loss, or zero if no
            valid precedence pairs exist.
        :raises ValueError: If *true_order* shape does not match the
            prediction shape or contains non-finite values.
        """
        pred_scores = self.forward(shuffled, features)
        if true_order.shape != pred_scores.shape:
            raise ValueError("true_order must have shape (n_simplices,)")

        order = true_order.to(device=pred_scores.device)
        if not torch.isfinite(order.to(dtype=torch.float32)).all().item():
            raise ValueError("true_order must contain only finite values")
        precedence = order.unsqueeze(1) < order.unsqueeze(0)
        if not precedence.any():
            return pred_scores.new_zeros(())

        diff = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)
        return F.relu(1.0 - diff[precedence]).mean()
