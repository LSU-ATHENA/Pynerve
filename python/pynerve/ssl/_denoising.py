"""Denoising task for persistence diagrams."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_rows as _encode_rows,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)
from .._validation import validate_finite_scalar as _finite_scalar
from ._validation import _validate_ssl_diagram


class TopologyDenoising(nn.Module):
    """Removes noise from persistence diagrams.

    Predicts additive corrections for the birth/death coordinates of each
    diagram pair and a per-pair noise-score that indicates the probability
    that the pair itself is noise.
    """

    def __init__(self, encoder: nn.Module, noise_threshold: float = 0.1):
        """Initialise the denoising model.

        :param encoder: Module that encodes a persistence-diagram tensor
            of shape ``(n_pairs, cols)`` into per-pair embeddings.
        :param noise_threshold: Non-negative threshold used by downstream
            logic to decide which pairs are classified as noise.
        :raises ValueError: If *noise_threshold* is negative.
        """
        super().__init__()
        noise_threshold = _finite_scalar(noise_threshold, "noise_threshold")
        if noise_threshold < 0:
            raise ValueError("noise_threshold must be non-negative")

        self.encoder = encoder
        self.threshold = noise_threshold
        self.denoiser = nn.Sequential(
            nn.Linear(_encoder_output_dim(encoder), 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, noisy_diagram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise a persistence diagram.

        :param noisy_diagram: Persistence-diagram tensor of shape
            ``(n_pairs, cols)`` with potentially corrupted coordinates.
        :returns: A tuple ``(denoised_diagram, noise_scores)``.
            *denoised_diagram* has the same shape as the input with
            corrected birth/death columns.  *noise_scores* is a
            ``(n_pairs,)`` tensor of per-pair noise probabilities (from
            a sigmoid).
        :raises ValueError: If the denoiser head returns an unexpected
            number of rows.
        """
        _validate_ssl_diagram(noisy_diagram)
        corrections = self.denoiser(_encode_rows(self.encoder, noisy_diagram))
        if corrections.shape[0] != noisy_diagram.shape[0]:
            raise ValueError("denoiser must return one row per diagram pair")

        denoised = noisy_diagram.clone()
        denoised[:, :2] = denoised[:, :2] + corrections[:, :2]
        noise_scores = torch.sigmoid(corrections[:, 2])
        return denoised, noise_scores

    def compute_loss(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noise_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined reconstruction and optional binary cross-entropy loss.

        :param noisy: Noisy persistence-diagram tensor of shape
            ``(n_pairs, cols)``.
        :param clean: Ground-truth clean diagram with the same shape as
            *noisy*.
        :param noise_labels: Optional boolean tensor of shape
            ``(n_pairs,)`` indicating whether each pair is true noise.
            When provided, binary cross-entropy is added to the recon-
            struction loss.
        :returns: Scalar loss.  If *noise_labels* is ``None`` only the
            MSE reconstruction term is returned.
        :raises ValueError: If *clean* does not match the shape of
            *noisy*, or if *noise_labels* has an incompatible shape or
            contains non-finite values.
        """
        if clean.shape != noisy.shape:
            raise ValueError("clean must have the same shape as noisy")
        _validate_ssl_diagram(clean)

        denoised, noise_scores = self.forward(noisy)
        recon_loss = F.mse_loss(denoised[:, :2], clean[:, :2])
        if noise_labels is None:
            return recon_loss
        if noise_labels.shape != noise_scores.shape:
            raise ValueError("noise_labels must have shape (n_pairs,)")
        if not torch.isfinite(noise_labels).all().item():
            raise ValueError("noise_labels must contain only finite values")
        return recon_loss + F.binary_cross_entropy(noise_scores, noise_labels.to(noise_scores))
