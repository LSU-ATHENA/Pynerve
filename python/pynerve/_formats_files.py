"""File format loaders and savers for persistence diagrams and point clouds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import parse_nonnegative_int
from .exceptions import InvalidArgumentError, ValidationError

Diagram = list[tuple[float, float, int]]


def _parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("inf")
    return float(value)


def _validate_diagram_entry(birth: float, death: float, dim: int) -> tuple[float, float, int]:
    birth = float(birth)
    death = float(death)
    dim = parse_nonnegative_int(dim, "dimension")
    if not np.isfinite(birth):
        raise ValidationError("diagram births must be finite", parameter="birth")
    if np.isnan(death) or np.isneginf(death):
        raise ValidationError("diagram deaths must be finite or +inf", parameter="death")
    if np.isfinite(death) and death < birth:
        raise ValidationError(
            "diagram finite deaths must be greater than or equal to births", parameter="death"
        )
    return birth, death, dim


def _validate_diagram(diagram: Diagram) -> Diagram:
    return [_validate_diagram_entry(birth, death, dim) for birth, death, dim in diagram]


def _parse_diagram_entry(entry: Any) -> tuple[float, float, int]:
    if isinstance(entry, dict):
        return _validate_diagram_entry(
            _parse_float(entry["birth"]),
            _parse_float(entry["death"]),
            entry.get("dimension", 0),
        )
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        dim = entry[2] if len(entry) > 2 else 0
        return _validate_diagram_entry(_parse_float(entry[0]), _parse_float(entry[1]), dim)
    raise ValidationError(
        "diagram entries must be mappings or birth/death sequences", parameter="entry"
    )


def _is_csv_header(row: list[str], birth_col: int, death_col: int, dim_col: int | None) -> bool:
    birth = row[birth_col].strip().lower()
    death = row[death_col].strip().lower()
    if birth != "birth" or death != "death":
        return False
    if dim_col is None:
        return True
    return row[dim_col].strip().lower() in {"dimension", "dim"}


def load_csv(
    filepath: str | Path,
    birth_col: int = 0,
    death_col: int = 1,
    dim_col: int | None = 2,
) -> Diagram:
    """Load a persistence diagram from a CSV file.

    :param filepath: Path to the input CSV file.
    :param birth_col: Zero-indexed column index for birth values. Defaults to 0.
    :param death_col: Zero-indexed column index for death values. Defaults to 1.
    :param dim_col: Zero-indexed column index for dimension values. Pass ``None`` to read only
        birth and death (dimension defaults to 0). Defaults to 2.
    :returns: List of ``(birth, death, dimension)`` tuples.
    :raises ValueError: If a row has too few columns or contains invalid numeric values.
    """
    birth_col = parse_nonnegative_int(birth_col, "birth_col")
    death_col = parse_nonnegative_int(death_col, "death_col")
    if dim_col is not None:
        dim_col = parse_nonnegative_int(dim_col, "dim_col")
    required_col = max(birth_col, death_col, dim_col if dim_col is not None else 0)
    result: Diagram = []
    filepath = Path(filepath)

    with filepath.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_number, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) <= required_col:
                raise ValueError(
                    f"CSV row {row_number} must contain birth, death, and dimension columns"
                )

            try:
                birth = float(row[birth_col])
                death = _parse_float(row[death_col])
                dim = (
                    parse_nonnegative_int(row[dim_col], "dimension")
                    if dim_col is not None and len(row) > dim_col
                    else 0
                )
            except ValueError as exc:
                if row_number == 1 and _is_csv_header(row, birth_col, death_col, dim_col):
                    continue
                raise ValueError(f"CSV row {row_number} contains invalid diagram values") from exc

            result.append(_validate_diagram_entry(birth, death, dim))

    return result


def save_csv(
    diagram: Diagram,
    filepath: str | Path,
    header: bool = True,
) -> None:
    """Save a persistence diagram to a CSV file.

    :param diagram: List of ``(birth, death, dimension)`` tuples.
    :param filepath: Output file path.
    :param header: Whether to write a ``birth,death,dimension`` header row. Defaults to True.
    :returns: None.
    :raises ValidationError: If any diagram entry is invalid.
    """
    diagram = _validate_diagram(diagram)
    filepath = Path(filepath)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if header:
            writer.writerow(["birth", "death", "dimension"])

        for birth, death, dim in diagram:
            writer.writerow([birth, death, dim])


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load a persistence diagram from a JSON file (Nerve schema).

    :param filepath: Path to the input JSON file.
    :returns: Dictionary with a ``"diagram"`` key and optionally a ``"diagrams"`` key,
        each containing parsed ``(birth, death, dimension)`` tuples.
    :raises InvalidArgumentError: If the JSON root is not a ``dict``.
    """
    filepath = Path(filepath)
    with filepath.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise InvalidArgumentError("JSON diagram file must contain an object", parameter="filepath")
    if "diagram" in data:
        data["diagram"] = [_parse_diagram_entry(entry) for entry in data["diagram"]]
    if "diagrams" in data:
        if "diagram" in data:
            data["diagrams"] = [_parse_diagram_entry(entry) for entry in data["diagrams"]]
        else:
            data["diagram"] = [_parse_diagram_entry(entry) for entry in data["diagrams"]]

    return data


def save_json(
    diagram: Diagram,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a persistence diagram to a JSON file (Nerve schema).

    :param diagram: List of ``(birth, death, dimension)`` tuples.
    :param filepath: Output file path.
    :param metadata: Optional metadata dict to include under a ``"metadata"`` key.
    :returns: None.
    :raises ValidationError: If any diagram entry is invalid.
    """
    diagram = _validate_diagram(diagram)
    data: dict[str, Any] = {
        "format": "nerve_v1",
        "diagram": [
            {
                "birth": float(birth),
                "death": None if np.isinf(death) else float(death),
                "dimension": int(dim),
            }
            for birth, death, dim in diagram
        ],
    }

    if metadata:
        data["metadata"] = metadata

    filepath = Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_off(filepath: str | Path) -> np.ndarray:
    """Load vertex coordinates from an OFF file.

    :param filepath: Path to the input OFF file.
    :returns: ``(N, 3)`` array of vertex coordinates.
    :raises InvalidArgumentError: If the file is not a valid OFF file, the header is missing,
        or vertex data is malformed.
    """
    filepath = Path(filepath)
    with filepath.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    lines = [line for line in lines if line and not line.startswith("#")]
    if len(lines) < 2:
        raise InvalidArgumentError("Not a valid OFF file", parameter="filepath")

    if lines[0] != "OFF":
        raise InvalidArgumentError("Not a valid OFF file", parameter="filepath")

    counts = lines[1].split()
    if len(counts) < 3:
        raise InvalidArgumentError("OFF header missing counts", parameter="filepath")
    n_vertices = parse_nonnegative_int(counts[0], "n_vertices")
    if len(lines) < 2 + n_vertices:
        raise InvalidArgumentError(
            "OFF file ended before all vertices were read", parameter="filepath"
        )

    points = []
    for i in range(2, 2 + n_vertices):
        coords = lines[i].split()
        if len(coords) < 3:
            raise InvalidArgumentError(
                "OFF vertex rows must contain at least x,y,z", parameter="filepath"
            )
        point = [float(value) for value in coords[:3]]
        if not np.isfinite(point).all():
            raise InvalidArgumentError(
                "OFF vertex coordinates must be finite", parameter="filepath"
            )
        points.append(point)

    return np.array(points)


def save_off(points: np.ndarray, filepath: str | Path, faces: np.ndarray | None = None) -> None:
    """Save vertex coordinates (and optionally faces) to an OFF file.

    :param points: ``(N, 3+)`` array of vertex coordinates. Only the first three columns are used.
    :param filepath: Output file path.
    :param faces: Optional ``(M, K)`` array of face vertex indices.
    :returns: None.
    :raises InvalidArgumentError: If ``points`` is not 2-D, has fewer than 3 columns, or contains
        non-finite values; if a face index references a non-existent vertex.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise InvalidArgumentError(
            "points must be a 2D array with at least three columns", parameter="points"
        )
    if not np.isfinite(points[:, :3]).all():
        raise InvalidArgumentError(
            "points must contain only finite coordinates", parameter="points"
        )
    n_points = points.shape[0]
    n_faces = len(faces) if faces is not None else 0
    filepath = Path(filepath)

    with filepath.open("w", encoding="utf-8") as f:
        f.write("OFF\n")
        f.write(f"{n_points} {n_faces} 0\n")

        for point in points:
            f.write(" ".join(map(str, point[:3])) + "\n")

        if faces is not None:
            for face in faces:
                indices = [parse_nonnegative_int(vertex, "face index") for vertex in face]
                if any(vertex >= n_points for vertex in indices):
                    raise InvalidArgumentError(
                        "face indices must reference existing points", parameter="faces"
                    )
                f.write(f"{len(indices)} " + " ".join(map(str, indices)) + "\n")


def save_ply(points: np.ndarray, filepath: str | Path, as_ascii: bool = True) -> None:
    """Save vertex coordinates to a PLY file.

    :param points: ``(N, 3+)`` array of vertex coordinates. Only the first three columns are used.
    :param filepath: Output file path.
    :param as_ascii: Whether to write ASCII PLY (vs. binary). Defaults to True.
    :returns: None.
    :raises InvalidArgumentError: If ``points`` is not 2-D, has fewer than 3 columns, or contains
        non-finite values.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise InvalidArgumentError(
            "points must be a 2D array with at least three columns", parameter="points"
        )
    if not np.isfinite(points[:, :3]).all():
        raise InvalidArgumentError(
            "points must contain only finite coordinates", parameter="points"
        )
    filepath = Path(filepath)
    n = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0" if as_ascii else "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with filepath.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def load_ply(filepath: str | Path) -> np.ndarray:
    """Load vertex coordinates from a PLY file.

    :param filepath: Path to the input PLY file.
    :returns: ``(N, 3)`` array of vertex coordinates.
    :raises InvalidArgumentError: If the file is not a valid PLY file, is missing ``x``/``y``/``z``
        properties, or vertex data is malformed.
    """
    filepath = Path(filepath)
    with filepath.open(encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n_vertices = 0
    properties: list[str] = []
    if not lines or lines[0].strip() != "ply":
        raise InvalidArgumentError("Not a valid PLY file", parameter="filepath")

    while i < len(lines):
        line = lines[i].strip()

        if line == "end_header":
            i += 1
            break

        if line.startswith("element vertex"):
            n_vertices = parse_nonnegative_int(line.split()[-1], "n_vertices")

        if line.startswith("property "):
            properties.append(line.split()[-1])

        i += 1

    if not {"x", "y", "z"}.issubset(properties):
        raise InvalidArgumentError("PLY file missing x,y,z properties", parameter="filepath")
    coordinate_indices = [properties.index(name) for name in ("x", "y", "z")]
    if len(lines) < i + n_vertices:
        raise InvalidArgumentError(
            "PLY file ended before all vertices were read", parameter="filepath"
        )

    points = []
    for j in range(n_vertices):
        fields = lines[i + j].split()
        if len(fields) <= max(coordinate_indices):
            raise InvalidArgumentError(
                "PLY vertex row has fewer fields than the header declares", parameter="filepath"
            )
        coords = [float(fields[index]) for index in coordinate_indices]
        if not np.isfinite(coords).all():
            raise InvalidArgumentError(
                "PLY vertex coordinates must be finite", parameter="filepath"
            )
        points.append(coords)

    return np.array(points)


__all__ = [
    "load_csv",
    "save_csv",
    "load_json",
    "save_json",
    "load_off",
    "save_off",
    "load_ply",
    "save_ply",
]
