"""Auto-detection of diagram and point-cloud file formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._formats_files import (
    load_csv,
    load_json,
    load_off,
    load_ply,
    save_csv,
    save_json,
    save_off,
)


def _format_key(filepath: Path, format_hint: str | None) -> str:
    value = format_hint if format_hint is not None else filepath.suffix
    key = value.lower().strip()
    return key[1:] if key.startswith(".") else key


def auto_load(
    filepath: str | Path,
    format_hint: str | None = None,
) -> Any:
    """Load a diagram or point-cloud file, auto-detecting format by extension or hint.

    :param filepath: Path to the input file.
    :param format_hint: Explicit format override (e.g. ``"csv"``, ``"json"``, ``"off"``).
        If ``None``, inferred from the file extension.
    :returns: Parsed data -- list of ``(birth, death, dimension)`` tuples for CSV/JSON,
        ``ndarray`` for OFF/PLY/NPY.
    :raises ValueError: If the format cannot be detected.
    """
    filepath = Path(filepath)
    key = _format_key(filepath, format_hint)

    if key in {"csv", "txt"}:
        return load_csv(filepath)
    if key == "json":
        return load_json(filepath)
    if key == "off":
        return load_off(filepath)
    if key == "ply":
        return load_ply(filepath)
    if key == "npy":
        return np.load(filepath)
    raise ValueError(
        f"Cannot auto-detect format for: {filepath}. Supported: csv, txt, json, off, ply, npy"
    )


def auto_save(
    data: Any,
    filepath: str | Path,
    format_hint: str | None = None,
) -> None:
    """Save a diagram or point-cloud file, auto-detecting format by extension or hint.

    :param data: Data to save -- list of ``(birth, death, dimension)`` tuples for CSV/JSON,
        ``ndarray`` for OFF/NPY.
    :param filepath: Output file path.
    :param format_hint: Explicit format override (e.g. ``"csv"``, ``"json"``, ``"off"``).
        If ``None``, inferred from the file extension.
    :returns: None.
    :raises ValueError: If the format cannot be detected.
    """
    filepath = Path(filepath)
    key = _format_key(filepath, format_hint)

    if key in {"csv", "txt"}:
        save_csv(data, filepath)
    elif key == "json":
        save_json(data, filepath)
    elif key == "off":
        save_off(data, filepath)
    elif key == "npy":
        np.save(filepath, data)
    else:
        raise ValueError(
            f"Cannot auto-detect format for: {filepath}. Supported: csv, txt, json, off, npy"
        )


__all__ = [
    "auto_load",
    "auto_save",
]
