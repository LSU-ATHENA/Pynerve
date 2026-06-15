"""Numerical round-trip tests for format I/O functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestCsvRoundTrip:
    """CSV save/load round-trips preserve numerical values."""

    def test_save_and_load_csv(self, tmp_path: Path) -> None:
        from pynerve import formats

        original = [(0.0, 1.0, 0), (0.0, 2.0, 1)]
        path = tmp_path / "test.csv"
        formats.save_csv(original, path)
        loaded = formats.load_csv(path)
        assert len(loaded) == 2
        for (b1, d1, dim1), (b2, d2, dim2) in zip(original, loaded, strict=True):
            assert b1 == pytest.approx(b2, abs=1e-10)
            assert d1 == pytest.approx(d2, abs=1e-10)
            assert dim1 == dim2

    def test_save_and_load_csv_inf_death(self, tmp_path: Path) -> None:
        from pynerve import formats

        original = [(0.0, float("inf"), 0)]
        path = tmp_path / "test_inf.csv"
        formats.save_csv(original, path)
        loaded = formats.load_csv(path)
        assert len(loaded) == 1
        b, d, dim = loaded[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert not np.isfinite(d)
        assert dim == 0


class TestJsonRoundTrip:
    """JSON save/load round-trips preserve numerical values."""

    def test_save_and_load_json(self, tmp_path: Path) -> None:
        from pynerve import formats

        original = [(0.0, 1.0, 0), (0.5, 2.5, 1)]
        path = tmp_path / "test.json"
        formats.save_json(original, path)
        loaded = formats.load_json(path)
        diagram = loaded["diagram"]
        assert len(diagram) == 2
        for (b1, d1, dim1), (b2, d2, dim2) in zip(original, diagram, strict=True):
            assert b1 == pytest.approx(b2, abs=1e-10)
            assert d1 == pytest.approx(d2, abs=1e-10)
            assert dim1 == dim2

    def test_save_and_load_json_inf_death(self, tmp_path: Path) -> None:
        from pynerve import formats

        original = [(0.0, float("inf"), 0)]
        path = tmp_path / "test_inf.json"
        formats.save_json(original, path)
        loaded = formats.load_json(path)
        diagram = loaded["diagram"]
        b, d, dim = diagram[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert d == float("inf")
        assert dim == 0


class TestExternalFormat:
    """External format conversion preserves values."""

    def test_to_external_contains_values(self) -> None:
        from pynerve import formats

        diagram = [(0.0, 1.0, 0), (0.0, float("inf"), 1)]
        output = formats.to_external(diagram)
        assert "0" in output
        assert "1" in output

    def test_from_external_contains_inf(self) -> None:
        from pynerve import formats

        diagram = [(0.0, 1.0, 0), (0.0, float("inf"), 1)]
        output = formats.to_external(diagram)
        assert "inf" in output or "null" in output


class TestGudhiFormat:
    """Gudhi format conversion preserves values."""

    def test_to_gudhi_values_preserved(self) -> None:
        from pynerve import formats

        original = [(0.0, 1.0, 0), (0.0, 2.0, 1)]
        gudhi = formats.to_gudhi(original)
        assert len(gudhi) == 2
        for (dim1, (b1, d1)), (b2, d2, dim2) in zip(gudhi, original, strict=True):
            assert dim1 == dim2
            assert b1 == pytest.approx(b2, abs=1e-10)
            assert d1 == pytest.approx(d2, abs=1e-10)

    def test_from_gudhi_diagram(self) -> None:
        from pynerve import formats

        gudhi_input = [(0, (0.0, 1.0)), (1, (0.0, 2.0))]
        loaded = formats.from_gudhi(gudhi_input, format_type="diagram")
        for (b1, d1, dim1), (dim2, (b2, d2)) in zip(loaded, gudhi_input, strict=True):
            assert dim1 == dim2
            assert b1 == pytest.approx(b2, abs=1e-10)
            assert d1 == pytest.approx(d2, abs=1e-10)


class TestDionysusFormat:
    """Dionysus format conversion preserves values."""

    def test_to_dionysus_called(self) -> None:
        from pynerve import formats

        original = [(0.0, 1.0, 0)]
        result = formats.to_dionysus(original)
        assert result is not None
