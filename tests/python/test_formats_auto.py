"""Tests for _formats_auto.py -- auto-detection of diagram and point-cloud file formats."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pynerve._formats_auto import _format_key, auto_load, auto_save
from pynerve._formats_files import save_csv, save_json


class TestFormatKey:
    def test_with_extension(self):
        assert _format_key(Path("data.csv"), None) == "csv"

    def test_with_leading_dot(self):
        assert _format_key(Path("data.csv"), ".json") == "json"

    def test_with_hint_no_dot(self):
        assert _format_key(Path("data.csv"), "json") == "json"

    def test_with_hint_uppercase(self):
        assert _format_key(Path("data.csv"), "JSON") == "json"

    def test_with_hint_whitespace(self):
        assert _format_key(Path("data.csv"), "  csv  ") == "csv"

    def test_no_extension_no_hint(self):
        assert _format_key(Path("data"), None) == ""

    def test_none_hint_uses_extension(self):
        assert _format_key(Path("file.txt"), None) == "txt"


class TestAutoLoad:
    def test_load_csv(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = auto_load(path)
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_txt_as_csv(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = auto_load(path)
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            save_json(diagram, path)
            result = auto_load(path)
            assert isinstance(result, dict)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_npy(self):
        arr = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            result = auto_load(path)
            np.testing.assert_array_equal(result, arr)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            Path(path).write_text("x")
            with pytest.raises(ValueError, match="auto-detect"):
                auto_load(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_explicit_format_hint(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = auto_load(path, format_hint="csv")
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_format_hint_overrides_extension(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = f.name
        try:
            save_csv(diagram, path)
            result = auto_load(path, format_hint="csv")
            assert len(result) == 1
        finally:
            Path(path).unlink(missing_ok=True)


class TestAutoSave:
    def test_save_csv(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            auto_save(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_txt_as_csv(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            auto_save(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_json(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            auto_save(diagram, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_npy(self):
        arr = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            auto_save(arr, path)
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded, arr)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_off(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".off", delete=False) as f:
            path = f.name
        try:
            auto_save(arr, path)
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_extension_raises(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="auto-detect"):
                auto_save(diagram, path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_format_hint_overrides_extension(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            auto_save(diagram, path, format_hint="csv")
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)
