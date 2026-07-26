"""Tests for pynerve/_formats_files.py -- CSV, JSON, OFF, PLY loaders/savers."""

from __future__ import annotations

import json
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
from pynerve.exceptions import InvalidArgumentError, ValidationError

DIAGRAM = [(0.0, 1.0, 0), (2.0, 5.0, 0), (0.5, 3.0, 1)]


class TestLoadCsv:
    def test_basic(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("0.0,1.0,0\n2.0,5.0,0\n0.5,3.0,1\n")
        result = load_csv(p)
        assert len(result) == 3

    def test_header_skip(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("birth,death,dim\n0.0,1.0,0\n")
        result = load_csv(p)
        assert len(result) == 1

    def test_inf_death(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("0.0,inf,0\n")
        result = load_csv(p)
        assert result[0][1] == float("inf")

    def test_empty_row_skip(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("0.0,1.0,0\n\n2.0,5.0,0\n")
        result = load_csv(p)
        assert len(result) == 2

    def test_bad_value_raises(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("x,y,z\n")
        with pytest.raises(ValueError, match="invalid diagram"):
            load_csv(p)

    def test_too_few_cols_raises(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("1.0\n")
        with pytest.raises(ValueError, match="must contain"):
            load_csv(p)


class TestSaveCsv:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.csv"
        save_csv(DIAGRAM, p)
        assert p.exists()
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_no_header(self, tmp_path):
        p = tmp_path / "out.csv"
        save_csv(DIAGRAM, p, header=False)
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 3


class TestLoadJson:
    def test_basic(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"diagram": [{"birth": 0.0, "death": 1.0, "dimension": 0}]}))
        result = load_json(p)
        assert "diagram" in result
        assert len(result["diagram"]) == 1

    def test_list_entry(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"diagram": [[0.0, 1.0, 0]]}))
        result = load_json(p)
        assert len(result["diagram"]) == 1

    def test_not_dict_raises(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text("[1,2,3]")
        with pytest.raises(InvalidArgumentError, match="object"):
            load_json(p)

    def test_diagrams_key(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"diagrams": [[0.0, 1.0, 0]]}))
        result = load_json(p)
        assert "diagram" in result


class TestSaveJson:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.json"
        save_json(DIAGRAM, p)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["format"] == "nerve_v1"
        assert len(data["diagram"]) == 3

    def test_with_metadata(self, tmp_path):
        p = tmp_path / "out.json"
        save_json(DIAGRAM, p, metadata={"tool": "pynerve"})
        data = json.loads(p.read_text())
        assert data["metadata"]["tool"] == "pynerve"


class TestLoadOff:
    def test_basic(self, tmp_path):
        p = tmp_path / "m.off"
        p.write_text("OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n")
        result = load_off(p)
        assert result.shape == (3, 3)

    def test_not_off_header(self, tmp_path):
        p = tmp_path / "bad.off"
        p.write_text("NOTOFF\n")
        with pytest.raises(InvalidArgumentError, match="Not a valid OFF"):
            load_off(p)

    def test_too_short(self, tmp_path):
        p = tmp_path / "m.off"
        p.write_text("OFF\n100 0 0\n")
        with pytest.raises(InvalidArgumentError, match="ended before"):
            load_off(p)

    def test_missing_counts(self, tmp_path):
        p = tmp_path / "m.off"
        p.write_text("OFF\n0 0\n")
        with pytest.raises(InvalidArgumentError, match="missing counts"):
            load_off(p)


class TestSaveOff:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.off"
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        save_off(points, p)
        assert p.exists()

    def test_non_2d_raises(self, tmp_path):
        with pytest.raises(InvalidArgumentError, match="2D array"):
            save_off(np.array([1.0, 2.0, 3.0]), Path("out.off"))

    def test_too_few_cols_raises(self, tmp_path):
        with pytest.raises(InvalidArgumentError, match="three columns"):
            save_off(np.array([[1.0, 2.0]]), Path("out.off"))


class TestSavePly:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.ply"
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        save_ply(points, p)
        assert p.exists()
        content = p.read_text()
        assert "ply" in content
        assert "element vertex 2" in content

    def test_non_2d_raises(self):
        with pytest.raises(InvalidArgumentError, match="2D array"):
            save_ply(np.array([1.0, 2.0]), Path("out.ply"))


class TestLoadPly:
    def test_basic(self, tmp_path):
        p = tmp_path / "m.ply"
        p.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "0.0 0.0 0.0\n"
            "1.0 1.0 1.0\n"
        )
        result = load_ply(p)
        assert result.shape == (2, 3)

    def test_not_ply_header(self, tmp_path):
        p = tmp_path / "bad.ply"
        p.write_text("notply\n")
        with pytest.raises(InvalidArgumentError, match="Not a valid PLY"):
            load_ply(p)

    def test_too_short_raises(self, tmp_path):
        p = tmp_path / "m.ply"
        p.write_text("ply\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n")
        with pytest.raises(InvalidArgumentError, match="ended before"):
            load_ply(p)
