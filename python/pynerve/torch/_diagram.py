"""Persistence diagram tensor container and batching utilities."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from ..exceptions import ShapeError, ValidationError
from ._statistics_core import _validate_stat_diagram
from ._vectorization_basis import _validate_positive_finite


class PersistenceDiagram:
    """Persistence diagram wrapper with batched tensor support."""

    def __init__(
        self,
        diagrams: torch.Tensor,
        mask: torch.Tensor | None = None,
        num_pairs: torch.Tensor | None = None,
        birth_idx: torch.Tensor | None = None,
        death_idx: torch.Tensor | None = None,
    ) -> None:
        if diagrams.dim() not in {2, 3}:
            raise ShapeError(
                f"Expected diagrams to be 2D or 3D, got {diagrams.dim()}D",
                parameter="diagrams",
                actual_ndim=diagrams.dim(),
            )
        if diagrams.shape[-1] != 3:
            raise ShapeError(
                f"Expected diagrams[..., 3] with (birth, death, dim), got last dimension {diagrams.shape[-1]}",
                parameter="diagrams",
                actual_shape=tuple(diagrams.shape),
            )
        diagrams = _validate_stat_diagram(diagrams)
        if diagrams.numel() > 0:
            dims = diagrams[..., 2]
            if not torch.isfinite(dims).all().item():
                raise ValidationError("diagram dimensions must be finite")
            if not (dims >= 0).all().item():
                raise ValidationError("diagram dimensions must be non-negative")
            if not (dims == torch.floor(dims)).all().item():
                raise ValidationError("diagram dimensions must be integers")

        if diagrams.dim() == 2:
            diagrams = diagrams.unsqueeze(0)

        self._diagrams = diagrams
        self._batch_size = diagrams.shape[0]
        self._max_pairs = diagrams.shape[1]

        if mask is None:
            mask = torch.ones(
                self._batch_size,
                self._max_pairs,
                dtype=torch.bool,
                device=diagrams.device,
            )
        elif mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[:2] != diagrams.shape[:2]:
            raise ShapeError(
                f"Mask shape {tuple(mask.shape)} does not match diagram shape {tuple(diagrams.shape[:2])}",
                parameter="mask",
                expected_shape=tuple(diagrams.shape[:2]),
                actual_shape=tuple(mask.shape),
            )
        self._mask = mask.to(dtype=torch.bool, device=diagrams.device)

        self._num_pairs = num_pairs
        self._birth_idx = birth_idx
        self._death_idx = death_idx

    @property
    def diagrams(self) -> torch.Tensor:
        return self._diagrams

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @property
    def num_pairs(self) -> torch.Tensor | None:
        return self._num_pairs

    def tensor(self) -> torch.Tensor:
        return self._diagrams

    def births(self, apply_mask: bool = True) -> torch.Tensor:
        births = self._diagrams[..., 0]
        if apply_mask:
            return births[self._mask]
        return births

    def deaths(self, apply_mask: bool = True) -> torch.Tensor:
        deaths = self._diagrams[..., 1]
        if apply_mask:
            return deaths[self._mask]
        return deaths

    def dimensions(self) -> torch.Tensor:
        return self._diagrams[..., 2].long()

    def to(self, device: str | torch.device) -> PersistenceDiagram:
        return PersistenceDiagram(
            self._diagrams.to(device),
            self._mask.to(device) if self._mask is not None else None,
            self._num_pairs.to(device) if self._num_pairs is not None else None,
            self._birth_idx.to(device) if self._birth_idx is not None else None,
            self._death_idx.to(device) if self._death_idx is not None else None,
        )

    def to_dtype(self, dtype: torch.dtype) -> PersistenceDiagram:
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValidationError(
                "dtype must be a floating-point dtype",
                parameter="dtype",
                expected="floating-point dtype",
                actual=str(dtype),
            )
        return PersistenceDiagram(
            self._diagrams.to(dtype),
            self._mask,
            self._num_pairs,
            self._birth_idx,
            self._death_idx,
        )

    @property
    def device(self) -> torch.device:
        return self._diagrams.device

    @property
    def dtype(self) -> torch.dtype:
        return self._diagrams.dtype

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_pairs(self) -> int:
        return self._max_pairs

    def get_batch_item(self, idx: int) -> PersistenceDiagram:
        if idx < 0 or idx >= self._batch_size:
            raise IndexError(f"Batch index {idx} out of range for batch size {self._batch_size}")
        return PersistenceDiagram(
            self._diagrams[idx : idx + 1],
            self._mask[idx : idx + 1] if self._mask is not None else None,
            self._num_pairs[idx : idx + 1] if self._num_pairs is not None else None,
            self._birth_idx[idx : idx + 1] if self._birth_idx is not None else None,
            self._death_idx[idx : idx + 1] if self._death_idx is not None else None,
        )

    def total_persistence(self, p: float = 2.0) -> torch.Tensor:
        p = _validate_positive_finite(p, "p")
        births = self._diagrams[..., 0]
        deaths = self._diagrams[..., 1]
        finite = torch.isfinite(deaths) & self._mask
        pers = torch.where(finite, deaths - births, torch.zeros_like(deaths))
        if p == 1.0:
            return pers.sum(dim=-1)
        return (pers**p).sum(dim=-1) ** (1.0 / p)

    def persistence_entropy(self) -> torch.Tensor:
        births = self._diagrams[..., 0]
        deaths = self._diagrams[..., 1]
        finite = torch.isfinite(deaths) & self._mask
        pers = torch.where(finite, deaths - births, torch.zeros_like(deaths))

        pers_sum = pers.sum(dim=-1, keepdim=True)
        pers_normalized = pers / (pers_sum + 1e-10)
        entropy = -(pers_normalized * torch.log(pers_normalized + 1e-10)).sum(dim=-1)
        entropy = torch.where(pers_sum.squeeze(-1) > 0, entropy, torch.zeros_like(entropy))
        return entropy

    def filter_by_dimension(self, dim: int) -> PersistenceDiagram:
        dim = int(dim)
        if dim < 0:
            raise ValidationError(
                "dim must be non-negative",
                parameter="dim",
                expected=">= 0",
                actual=str(dim),
            )
        dims = self._diagrams[..., 2].long()
        dim_mask = (dims == dim) & self._mask
        return PersistenceDiagram(
            self._diagrams,
            dim_mask,
            self._num_pairs,
            self._birth_idx,
            self._death_idx,
        )

    def __repr__(self) -> str:
        valid_pairs = self._mask.sum(dim=-1)
        return (
            "PersistenceDiagram("
            f"batch={self._batch_size}, max_pairs={self._max_pairs}, "
            f"valid={valid_pairs.tolist()}, device={self.device}, dtype={self.dtype})"
        )


def batch_diagrams(diagrams: list[PersistenceDiagram]) -> PersistenceDiagram:
    """Batch multiple diagrams into a single padded batch."""
    if not diagrams:
        raise ValidationError("Cannot batch empty list of diagrams")
    device = diagrams[0].device
    dtype = diagrams[0].dtype
    for diagram in diagrams:
        if diagram.device != device:
            raise ValidationError("all diagrams must be on the same device")
        if diagram.dtype != dtype:
            raise ValueError("all diagrams must have the same dtype")

    diag_list = [d.diagrams for d in diagrams]
    max_pairs = max(d.shape[1] for d in diag_list)

    padded_diags: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for diagram, d in zip(diagrams, diag_list, strict=True):
        n_pairs = d.shape[1]
        if n_pairs < max_pairs:
            padding = torch.zeros(
                d.shape[0],
                max_pairs - n_pairs,
                3,
                dtype=d.dtype,
                device=d.device,
            )
            padded = torch.cat([d, padding], dim=1)
            mask = torch.cat(
                [
                    diagram.mask,
                    torch.zeros(
                        d.shape[0],
                        max_pairs - n_pairs,
                        dtype=torch.bool,
                        device=d.device,
                    ),
                ],
                dim=1,
            )
        else:
            padded = d
            mask = diagram.mask
        padded_diags.append(padded)
        masks.append(mask)

    batched_diags = torch.cat(padded_diags, dim=0)
    batched_mask = torch.cat(masks, dim=0)

    if all(d.num_pairs is not None for d in diagrams):
        batched_num_pairs: Tensor | None = torch.cat(
            [cast(Tensor, d.num_pairs) for d in diagrams], dim=0
        )
    else:
        batched_num_pairs = None

    return PersistenceDiagram(batched_diags, batched_mask, batched_num_pairs)


def unbatch_diagrams(diagram: PersistenceDiagram) -> list[PersistenceDiagram]:
    """Split a batched diagram into per-item diagrams."""
    return [diagram.get_batch_item(i) for i in range(diagram.batch_size)]
