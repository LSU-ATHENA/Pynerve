"""Nerve numeric constants for numerical stability."""

from __future__ import annotations

EPS: float = 1e-8
EPS_1e_6: float = 1e-6
EPS_1e_9: float = 1e-9
EPS_1e_5: float = 1e-5

__all__ = [
    "EPS",
    "EPS_1e_5",
    "EPS_1e_6",
    "EPS_1e_9",
]
