"""Torch helpers for persistence-diagram modules."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError(
        "pynerve._torch_diagrams requires PyTorch. Install it with: pip install pynerve[torch]"
    ) from None


def encoder_output_dim(encoder: nn.Module, default: int = 64) -> int:
    return int(getattr(encoder, "output_dim", default))


def validate_diagram(diagram: torch.Tensor, min_cols: int = 2, name: str = "diagram") -> None:
    if not isinstance(diagram, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if diagram.dim() != 2 or diagram.shape[1] < min_cols:
        raise ValueError(f"{name} must have shape (n_pairs, at least {min_cols})")


def birth_death(diagram: torch.Tensor) -> torch.Tensor:
    validate_diagram(diagram, min_cols=2, name="diagram")
    return diagram[:, :2]


def persistence(diagram: torch.Tensor) -> torch.Tensor:
    pairs = birth_death(diagram)
    return pairs[:, 1] - pairs[:, 0]


def encode_diagram_rows(
    encoder: nn.Module, diagram: torch.Tensor, min_cols: int = 2
) -> torch.Tensor:
    validate_diagram(diagram, min_cols=min_cols, name="diagram")
    encoded = encoder(diagram.unsqueeze(0))
    if not isinstance(encoded, torch.Tensor):
        raise TypeError("encoder must return a tensor")

    n_pairs = diagram.shape[0]
    if encoded.dim() == 3:
        if encoded.shape[0] != 1:
            raise ValueError("batched encoder output must preserve one input diagram")
        encoded = encoded.squeeze(0)
    elif encoded.dim() == 2 and encoded.shape[0] == 1:
        encoded = encoded.expand(n_pairs, -1)
    elif encoded.dim() == 1:
        encoded = encoded.unsqueeze(0).expand(n_pairs, -1)

    if encoded.dim() != 2 or encoded.shape[0] != n_pairs:
        raise ValueError("encoder output must be graph-level or have one row per pair")
    return encoded


def encode_diagram_embedding(
    encoder: nn.Module, diagram: torch.Tensor, min_cols: int = 2
) -> torch.Tensor:
    rows = encode_diagram_rows(encoder, diagram, min_cols=min_cols)
    if rows.shape[0] == 0:
        return rows.new_zeros((encoder_output_dim(encoder),))
    return rows.mean(dim=0)
