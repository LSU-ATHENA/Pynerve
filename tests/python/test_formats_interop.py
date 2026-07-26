"""Tests for _formats_interop.py -- third-party TDA library format converters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pynerve._formats_interop import (
    _as_diagram_array,
    _from_gudhi_diagram,
    _from_gudhi_simplex_tree,
    _resolve_simplex_iter,
    _validate_diagram,
    _validate_diagram_entry,
    from_dionysus,
    from_external,
    from_giotto,
    from_gudhi,
    from_sktda,
    to_dionysus,
    to_external,
    to_gudhi,
)
from pynerve.exceptions import InvalidArgumentError, ValidationError


class TestValidateDiagramEntry:
    def test_valid_entry(self):
        result = _validate_diagram_entry(0.0, 1.0, 0)
        assert result == (0.0, 1.0, 0)

    def test_valid_with_inf_death(self):
        result = _validate_diagram_entry(0.0, float("inf"), 1)
        assert result == (0.0, float("inf"), 1)

    def test_nan_birth_raises(self):
        with pytest.raises(ValidationError, match="births must be finite"):
            _validate_diagram_entry(float("nan"), 1.0, 0)

    def test_nan_death_raises(self):
        with pytest.raises(ValidationError, match="deaths must be finite"):
            _validate_diagram_entry(0.0, float("nan"), 0)

    def test_neginf_death_raises(self):
        with pytest.raises(ValidationError, match="deaths must be finite"):
            _validate_diagram_entry(0.0, float("-inf"), 0)

    def test_death_below_birth_raises(self):
        with pytest.raises(ValidationError, match="deaths must be greater"):
            _validate_diagram_entry(2.0, 1.0, 0)

    def test_negative_dim_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            _validate_diagram_entry(0.0, 1.0, -1)

    def test_death_equal_birth_allowed(self):
        result = _validate_diagram_entry(1.0, 1.0, 0)
        assert result == (1.0, 1.0, 0)


class TestValidateDiagram:
    def test_valid_list(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        result = _validate_diagram(diagram)
        assert result == diagram

    def test_empty_list(self):
        result = _validate_diagram([])
        assert result == []

    def test_invalid_entry_raises(self):
        with pytest.raises(ValidationError, match="births must be finite"):
            _validate_diagram([(float("nan"), 1.0, 0)])


class TestAsDiagramArray:
    def test_valid_2d(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64)
        result = _as_diagram_array(arr)
        assert result.shape == (2, 3)

    def test_empty_array(self):
        arr = np.empty((0, 3), dtype=np.float64)
        result = _as_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_list_input(self):
        result = _as_diagram_array([[0.0, 1.0, 0]])
        assert result.shape == (1, 3)

    def test_1d_raises(self):
        with pytest.raises(Exception):
            _as_diagram_array(np.array([0.0, 1.0]))


class TestResolveSimplexIter:
    def test_list_input(self):
        data = [((0, 1), 0.5), ((1, 2), 1.0)]
        result = _resolve_simplex_iter(data)
        assert result is data

    def test_tuple_input(self):
        data = (((0, 1), 0.5),)
        result = _resolve_simplex_iter(data)
        assert result is data

    def test_invalid_object_raises(self):
        with pytest.raises(ValidationError, match="simplex_tree"):
            _resolve_simplex_iter(object())


class TestFromGudhiDiagram:
    def test_two_element_format(self):
        data = [(0, (0.0, 1.0)), (1, (1.0, 2.0))]
        result = _from_gudhi_diagram(data)
        assert len(result) == 2
        assert result[0] == (0.0, 1.0, 0)
        assert result[1] == (1.0, 2.0, 1)

    def test_three_element_format(self):
        data = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        result = _from_gudhi_diagram(data)
        assert len(result) == 2

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="GUDHI diagram"):
            _from_gudhi_diagram([(0.0,)])

    def test_empty_list(self):
        result = _from_gudhi_diagram([])
        assert result == []


class TestFromGudhiSimplexTree:
    def test_valid_simplices(self):
        data = [((0, 1), 0.5), ((1, 2), 1.0)]
        result = _from_gudhi_simplex_tree(data)
        assert len(result) == 2
        assert result[0] == ((0, 1), 0.5)

    def test_invalid_entry_raises(self):
        with pytest.raises(ValidationError, match="simplex entries"):
            _from_gudhi_simplex_tree([(0,)])

    def test_nan_filtration_raises(self):
        with pytest.raises(ValidationError, match="filtration values"):
            _from_gudhi_simplex_tree([((0, 1), float("nan"))])


class TestFromGudhi:
    def test_diagram_format(self):
        data = [(0, (0.0, 1.0))]
        result = from_gudhi(data, format_type="diagram")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_simplex_tree_format(self):
        data = [((0, 1), 0.5)]
        result = from_gudhi(data, format_type="simplex_tree")
        assert len(result) == 1
        assert result[0] == ((0, 1), 0.5)

    def test_unknown_format_raises(self):
        with pytest.raises(InvalidArgumentError, match="format_type"):
            from_gudhi([], format_type="unknown")


class TestFromExternal:
    def test_valid_external_output(self):
        data = {"dgms": [[(0.0, 1.0)], [(1.0, 2.0)]]}
        result = from_external(data)
        assert len(result) == 2
        assert result[0] == (0.0, 1.0, 0)
        assert result[1] == (1.0, 2.0, 1)

    def test_missing_dgms_key_raises(self):
        with pytest.raises(ValidationError, match="dgms"):
            from_external({})

    def test_empty_dgms(self):
        result = from_external({"dgms": []})
        assert result == []

    def test_multiple_dimensions(self):
        data = {"dgms": [[(0.0, 1.0), (0.5, 2.0)], [], [(2.0, 3.0)]]}
        result = from_external(data)
        assert len(result) == 3
        assert result[0][2] == 0
        assert result[2][2] == 2


class TestFromDionysus:
    def test_with_birth_death_attrs(self):
        class DgmPoint:
            def __init__(self, b, d):
                self.birth = b
                self.death = d

        dgm = [[DgmPoint(0.0, 1.0), DgmPoint(1.0, 2.0)]]
        result = from_dionysus(dgm)
        assert len(result) == 2
        assert result[0] == (0.0, 1.0, 0)

    def test_with_indexable_points(self):
        dgm = [[(0.0, 1.0), (1.0, 2.0)]]
        result = from_dionysus(dgm)
        assert len(result) == 2

    def test_invalid_point_raises(self):
        with pytest.raises((ValidationError, TypeError), match="Dionysus|Dion"):
            from_dionysus([[[0]]])  # single-element list: has len() but no birth/death attrs

    def test_empty_diagrams(self):
        result = from_dionysus([])
        assert result == []


class TestFromGiotto:
    def test_valid_array(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64)
        result = from_giotto(arr)
        assert len(result) == 2
        assert result[0] == (0.0, 1.0, 0)

    def test_empty_array(self):
        result = from_giotto(np.empty((0, 3), dtype=np.float64))
        assert result == []

    def test_list_input(self):
        result = from_giotto([[0.0, 1.0, 0]])
        assert len(result) == 1


class TestFromSktda:
    def test_valid_object(self):
        class SktdaResult:
            dgms = [[(0.0, 1.0)], [(1.0, 2.0)]]

        result = from_sktda(SktdaResult())
        assert len(result) == 2

    def test_missing_dgms_raises(self):
        with pytest.raises(ValidationError, match="dgms"):
            from_sktda(object())

    def test_empty_dgms(self):
        class SktdaResult:
            dgms = []

        result = from_sktda(SktdaResult())
        assert result == []


class TestToGudhi:
    def test_valid_diagram(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        result = to_gudhi(diagram)
        assert len(result) == 2
        assert result[0] == (0, (0.0, 1.0))
        assert result[1] == (1, (1.0, 2.0))

    def test_empty_diagram(self):
        result = to_gudhi([])
        assert result == []

    def test_invalid_entry_raises(self):
        with pytest.raises(ValidationError):
            to_gudhi([(float("nan"), 1.0, 0)])


class TestToExternal:
    def test_valid_diagram_to_json(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        result = to_external(diagram)
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["format"] == "nerve_diagram"
        assert len(data["diagrams"]) == 2

    def test_valid_diagram_to_file(self):
        diagram = [(0.0, 1.0, 0)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = to_external(diagram, filepath=path)
            assert result == path
            content = json.loads(Path(path).read_text())
            assert content["format"] == "nerve_diagram"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_diagram(self):
        result = to_external([])
        data = json.loads(result)
        assert data["diagrams"] == []

    def test_inf_death_becomes_none(self):
        diagram = [(0.0, float("inf"), 0)]
        result = to_external(diagram)
        data = json.loads(result)
        assert data["diagrams"][0]["death"] is None


class TestToDionysus:
    def test_valid_diagram(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 1), (0.5, 1.5, 0)]
        result = to_dionysus(diagram)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_empty_diagram(self):
        result = to_dionysus([])
        assert result == {}

    def test_dim_groups(self):
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 0), (2.0, 3.0, 1)]
        result = to_dionysus(diagram)
        assert result[0] == [(0.0, 1.0), (1.0, 2.0)]
        assert result[1] == [(2.0, 3.0)]
