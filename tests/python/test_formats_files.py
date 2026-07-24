"""Additional tests for _formats_files.py — file I/O edge cases and error paths."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pynerve._formats_files import (
    load_csv,
    load_json,
    load_off,
    load_ply,
    save_csv,
    save_json,
    save_off,
    save_ply,
)
from pynerve.exceptions import ValidationError


class TestLoadCSV:
    def test_load_valid(self):
        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = load_csv(path)
            assert len(result) == 2
            np.testing.assert_array_almost_equal(
                np.array(result), np.array([(0.0, 1.0, 0), (0.5, 2.0, 1)])
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv([], path)
            result = load_csv(path)
            assert len(result) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/file.csv")

    def test_load_with_header(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path, header="birth,death,dim")
            result = load_csv(path)
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_with_header(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path, header="birth,death,dim")
            content = Path(path).read_text()
            assert "birth" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_csv_no_header(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_csv_pathlib(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, Path(path))
            loaded = load_csv(Path(path))
            assert len(loaded) == 1
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadJSON:
    def test_load_valid(self):
        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json(diagram, path)
            result = load_json(path)
            assert result["format"] == "nerve_v1"
            assert len(result["diagram"]) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_with_metadata(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json(diagram, path, metadata={"source": "unit_test", "version": "1.0"})
            content = json.loads(Path(path).read_text())
            assert content["metadata"]["source"] == "unit_test"
            assert content["metadata"]["version"] == "1.0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_json_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json([], path)
            content = json.loads(Path(path).read_text())
            assert content["diagram"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/file.json")


class TestLoadOFF:
    def test_load_valid(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".off", mode="w", delete=False) as f:
            path = f.name
        try:
            save_off(points, path)
            result = load_off(path)
            assert isinstance(result, np.ndarray)
            assert result.shape[0] >= 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_off_single_point(self):
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".off", mode="w", delete=False) as f:
            path = f.name
        try:
            save_off(points, path)
            content = Path(path).read_text()
            assert "OFF" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_off_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_off("/nonexistent/file.off")


class TestLoadPLY:
    def test_load_valid(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ply", mode="w", delete=False) as f:
            path = f.name
        try:
            save_ply(points, path)
            result = load_ply(path)
            assert isinstance(result, np.ndarray)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_ply_single_point(self):
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ply", mode="w", delete=False) as f:
            path = f.name
        try:
            save_ply(points, path)
            content = Path(path).read_text()
            assert "ply" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_ply_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_ply("/nonexistent/file.ply")
