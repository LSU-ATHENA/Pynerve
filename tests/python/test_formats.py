"""Tests for format converters and file serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pynerve.exceptions import ValidationError
from pynerve.formats import (
    Diagram,
    DiagramLike,
    load_csv,
    load_diagrams,
    load_json,
    save_csv,
    save_diagrams,
    save_json,
)


class TestDiagramTypes:
    def test_diagram_type_alias(self):
        d: Diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        assert len(d) == 2

    def test_diagram_like_numpy(self):
        arr = np.array([(0.0, 1.0, 0), (0.5, 2.0, 1)])
        assert isinstance(arr, DiagramLike) is True

    def test_diagram_like_list(self):
        lst: DiagramLike = [(0.0, 1.0, 0)]
        assert isinstance(lst, list)


class TestSaveLoadCSV:
    def test_save_and_load_csv_roundtrip(self):
        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1), (0.3, float("inf"), 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            loaded = load_csv(path)
            assert len(loaded) == 3
            np.testing.assert_array_almost_equal(
                np.array([(0.0, 1.0, 0), (0.5, 2.0, 1), (0.3, float("inf"), 0)]),
                np.array(loaded),
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_csv_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv([], path)
            loaded = load_csv(path)
            assert len(loaded) == 0
        finally:
            Path(path).unlink(missing_ok=True)


class TestSaveLoadJSON:
    def test_save_and_load_json_roundtrip(self):
        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json(diagram, path)
            loaded = load_json(path)
            assert loaded["format"] == "nerve_v1"
            assert len(loaded["diagram"]) == 2
            assert loaded["diagram"][0] == (0.0, 1.0, 0)
            assert loaded["diagram"][1] == (0.5, 2.0, 1)
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadDiagramsDispatch:
    def test_load_csv_dispatch(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = load_diagrams(path)
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json_dispatch(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json(diagram, path)
            result = load_diagrams(path)
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            Path(path).write_text("x")
            with pytest.raises(ValidationError, match="extension"):
                load_diagrams(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestSaveDiagrams:
    def test_save_diagrams_csv(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_diagrams(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_diagrams_json(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_diagrams(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_diagrams_numpy_array(self):
        diagram = np.array([(0.0, 1.0, 0), (0.5, 2.0, 1)])
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_diagrams(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_diagrams_numpy_1d_raises(self):
        diagram = np.array([0.0, 1.0, 0.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValidationError, match="2D"):
                save_diagrams(diagram, path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValidationError, match="extension"):
                save_diagrams([(0.0, 1.0, 0)], path)
        finally:
            Path(path).unlink(missing_ok=True)
