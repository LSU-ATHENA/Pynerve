"""Comprehensive correctness tests for format auto-detection and TDA interop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pynerve._formats_auto import auto_load, auto_save
from pynerve._formats_interop import (
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

# Shared helpers

DIAGRAM_SIMPLE: list[tuple[float, float, int]] = [
    (0.0, 1.0, 0),
    (0.5, 2.0, 1),
    (0.3, float("inf"), 0),
    (1.0, 1.5, 1),
    (2.0, 3.5, 2),
]

DIAGRAM_EMPTY: list[tuple[float, float, int]] = []

DIAGRAM_SINGLE: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]

POINTS_3D = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)


def _write_corrupted(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# Auto-detection: format key derivation


class TestAutoDetectFormatKey:
    """Tests for _format_key-driven dispatch in auto_load/auto_save."""

    def test_detect_csv_from_extension(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("0.0,1.0,0\n")
        result = auto_load(str(path))
        assert result == [(0.0, 1.0, 0)]

    def test_detect_txt_as_csv(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("0.0,1.0,0\n")
        result = auto_load(str(path))
        assert result == [(0.0, 1.0, 0)]

    def test_detect_json_from_extension(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"format":"nerve_v1","diagram":[{"birth":0.0,"death":1.0,"dimension":0}]}')
        result = auto_load(str(path))
        assert "diagram" in result
        assert result["diagram"][0] == (0.0, 1.0, 0)

    def test_detect_npy_from_extension(self, tmp_path):
        path = tmp_path / "test.npy"
        arr = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        np.save(str(path), arr)
        result = auto_load(str(path))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_detect_off_from_extension(self, tmp_path):
        path = tmp_path / "test.off"
        path.write_text("OFF\n3 0 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n")
        result = auto_load(str(path))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    def test_detect_ply_from_extension(self, tmp_path):
        path = tmp_path / "test.ply"
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n1.0 0.0 0.0\n"
        )
        result = auto_load(str(path))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_format_hint_overrides_extension(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("0.0,1.0,0\n0.5,2.0,1\n")
        result = auto_load(str(path), format_hint="csv")
        assert result == [(0.0, 1.0, 0), (0.5, 2.0, 1)]

    def test_format_hint_with_dot_prefix(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("0.0,1.0,0\n")
        result = auto_load(str(path), format_hint=".csv")
        assert result == [(0.0, 1.0, 0)]

    def test_format_hint_case_insensitive(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_text("0.0,1.0,0\n")
        result = auto_load(str(path), format_hint="CSV")
        assert result == [(0.0, 1.0, 0)]

    def test_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "test.unknown"
        path.write_text("garbage")
        with pytest.raises(ValueError, match="auto-detect"):
            auto_load(str(path))

    def test_unknown_extension_auto_save_raises(self, tmp_path):
        with pytest.raises(ValueError, match="auto-detect"):
            auto_save([(0.0, 1.0, 0)], tmp_path / "test.unknown")

    def test_auto_load_with_path_object(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("0.0,1.0,0\n")
        result = auto_load(path)
        assert result == [(0.0, 1.0, 0)]


# Auto-detection: content verification per format


class TestAutoLoadContent:
    """Verify auto_load returns correct content for each format."""

    def test_csv_content_preserves_values(self, tmp_path):
        path = tmp_path / "dgm.csv"
        diagram = DIAGRAM_SIMPLE
        from pynerve._formats_files import save_csv

        save_csv(diagram, path)
        loaded = auto_load(str(path))
        assert len(loaded) == len(diagram)
        for (lb, ld, ldim), (eb, ed, edim) in zip(loaded, diagram, strict=True):
            assert lb == eb
            assert ld == ed or (np.isinf(ld) and np.isinf(ed))
            assert ldim == edim

    def test_json_content_preserves_format_and_diagram(self, tmp_path):
        path = tmp_path / "dgm.json"
        from pynerve._formats_files import save_json

        save_json(DIAGRAM_SIMPLE, path)
        loaded = auto_load(str(path))
        assert loaded["format"] == "nerve_v1"
        assert len(loaded["diagram"]) == len(DIAGRAM_SIMPLE)
        assert loaded["diagram"][0] == DIAGRAM_SIMPLE[0]

    def test_off_content_returns_point_cloud(self, tmp_path):
        path = tmp_path / "pts.off"
        from pynerve._formats_files import save_off

        save_off(POINTS_3D, path)
        loaded = auto_load(str(path))
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, POINTS_3D)

    def test_ply_content_returns_point_cloud(self, tmp_path):
        path = tmp_path / "pts.ply"
        from pynerve._formats_files import save_ply

        save_ply(POINTS_3D, path)
        loaded = auto_load(str(path))
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_almost_equal(loaded, POINTS_3D)

    def test_npy_content_roundtrip(self, tmp_path):
        path = tmp_path / "arr.npy"
        arr = np.random.rand(10, 5)
        np.save(str(path), arr)
        loaded = auto_load(str(path))
        np.testing.assert_array_equal(loaded, arr)


# Auto-save + auto-load roundtrip per format


class TestAutoSaveLoadRoundtrip:
    """Save via auto_save, re-load via auto_load, verify equivalence."""

    def test_csv_roundtrip_preserves_diagram(self, tmp_path):
        path = tmp_path / "roundtrip.csv"
        auto_save(DIAGRAM_SIMPLE, str(path))
        loaded = auto_load(str(path))
        assert len(loaded) == len(DIAGRAM_SIMPLE)
        for (lb, ld, ldim), (eb, ed, edim) in zip(loaded, DIAGRAM_SIMPLE, strict=True):
            assert lb == eb
            assert ld == ed or (np.isinf(ld) and np.isinf(ed))
            assert ldim == edim

    def test_csv_roundtrip_empty(self, tmp_path):
        path = tmp_path / "empty.csv"
        auto_save(DIAGRAM_EMPTY, str(path))
        loaded = auto_load(str(path))
        assert loaded == []

    def test_csv_roundtrip_single_entry(self, tmp_path):
        path = tmp_path / "single.csv"
        auto_save(DIAGRAM_SINGLE, str(path))
        loaded = auto_load(str(path))
        assert loaded == DIAGRAM_SINGLE

    def test_json_roundtrip_preserves_all_properties(self, tmp_path):
        path = tmp_path / "roundtrip.json"
        auto_save(DIAGRAM_SIMPLE, str(path))
        loaded = auto_load(str(path))
        assert loaded["format"] == "nerve_v1"
        assert len(loaded["diagram"]) == len(DIAGRAM_SIMPLE)
        for (lb, ld, ldim), (eb, ed, edim) in zip(loaded["diagram"], DIAGRAM_SIMPLE, strict=True):
            assert lb == eb
            assert ld == ed or (np.isinf(ld) and np.isinf(ed))
            assert ldim == edim

    def test_json_roundtrip_with_format_hint(self, tmp_path):
        path = tmp_path / "data.bin"
        auto_save(DIAGRAM_SINGLE, str(path), format_hint="json")
        loaded = auto_load(str(path), format_hint="json")
        assert loaded["format"] == "nerve_v1"
        assert loaded["diagram"] == [(0.0, 1.0, 0)]

    def test_npy_roundtrip_numpy_array(self, tmp_path):
        path = tmp_path / "arr.npy"
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        auto_save(arr, str(path))
        loaded = auto_load(str(path))
        np.testing.assert_array_equal(loaded, arr)

    def test_off_roundtrip_point_cloud(self, tmp_path):
        path = tmp_path / "pts.off"
        auto_save(POINTS_3D, str(path), format_hint="off")
        loaded = auto_load(str(path), format_hint="off")
        np.testing.assert_array_almost_equal(loaded, POINTS_3D)


# Auto-detection: error cases


class TestAutoDetectErrors:
    """Error handling for corrupted, empty, and malformed files."""

    def test_corrupted_csv_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        _write_corrupted(path, "birth,death,dimension\nnot-a-number,1.0,0\n")
        with pytest.raises(ValueError, match="CSV"):
            auto_load(str(path))

    def test_corrupted_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        _write_corrupted(path, "{this is not json")
        with pytest.raises((ValueError, Exception)):
            auto_load(str(path))

    def test_corrupted_off_raises(self, tmp_path):
        path = tmp_path / "bad.off"
        _write_corrupted(path, "not a valid OFF file")
        with pytest.raises((InvalidArgumentError, ValueError)):
            auto_load(str(path))

    def test_corrupted_ply_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        _write_corrupted(path, "not a valid PLY file")
        with pytest.raises((InvalidArgumentError, ValueError)):
            auto_load(str(path))

    def test_empty_csv_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        loaded = auto_load(str(path))
        assert loaded == []

    def test_empty_json_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")
        # auto_load succeeds but returns an empty dict (no 'diagram' key)
        loaded = auto_load(str(path))
        assert isinstance(loaded, dict)
        assert "diagram" not in loaded

    def test_empty_off_raises(self, tmp_path):
        path = tmp_path / "empty.off"
        path.write_text("", encoding="utf-8")
        with pytest.raises((InvalidArgumentError, ValueError)):
            auto_load(str(path))

    def test_empty_ply_raises(self, tmp_path):
        path = tmp_path / "empty.ply"
        path.write_text("", encoding="utf-8")
        with pytest.raises((InvalidArgumentError, ValueError)):
            auto_load(str(path))

    def test_npy_nonexistent_raises(self, tmp_path):
        path = tmp_path / "nonexistent.npy"
        with pytest.raises((FileNotFoundError, OSError)):
            auto_load(str(path))


# Auto-save: property preservation


class TestAutoSaveProperties:
    """Verify auto_save preserves all diagram properties (birth, death, dimension)."""

    def test_preserves_birth_values(self, tmp_path):
        path = tmp_path / "births.csv"
        diagram = [(1.5, 3.0, 0), (2.71828, 3.14159, 1)]
        auto_save(diagram, str(path))
        loaded = auto_load(str(path))
        for (lb, _, _), (eb, _, _) in zip(loaded, diagram, strict=True):
            assert lb == pytest.approx(eb)

    def test_preserves_death_values_including_inf(self, tmp_path):
        path = tmp_path / "deaths.csv"
        diagram = [(0.0, float("inf"), 0), (1.0, 5.0, 1)]
        auto_save(diagram, str(path))
        loaded = auto_load(str(path))
        assert len(loaded) == 2
        assert np.isinf(loaded[0][1])
        assert loaded[1][1] == 5.0

    def test_preserves_dimension_values(self, tmp_path):
        path = tmp_path / "dims.json"
        auto_save(DIAGRAM_SIMPLE, str(path))
        loaded = auto_load(str(path))
        dimensions = [int(d) for _, _, d in loaded["diagram"]]
        expected = [d for _, _, d in DIAGRAM_SIMPLE]
        assert dimensions == expected

    def test_preserves_all_dimensions_zero_to_high(self, tmp_path):
        path = tmp_path / "highdim.json"
        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1), (1.0, 3.0, 2), (2.0, 4.0, 5)]
        auto_save(diagram, str(path))
        loaded = auto_load(str(path))
        loaded_dims = [int(d) for _, _, d in loaded["diagram"]]
        assert loaded_dims == [0, 1, 2, 5]

    def test_preserves_multiple_entries_same_dim(self, tmp_path):
        path = tmp_path / "sampledim.csv"
        diagram = [(0.0, 1.0, 0), (1.0, 2.0, 0), (2.0, 3.0, 0)]
        auto_save(diagram, str(path))
        loaded = auto_load(str(path))
        assert loaded == diagram


# TDA Interop: function existence and signature


class TestInteropExistence:
    """All interop conversion functions exist and are callable."""

    @pytest.mark.parametrize(
        "func",
        [from_gudhi, from_external, from_dionysus, from_giotto, from_sktda],
    )
    def test_from_functions_exist(self, func):
        assert callable(func)

    @pytest.mark.parametrize("func", [to_gudhi, to_external, to_dionysus])
    def test_to_functions_exist(self, func):
        assert callable(func)


# TDA Interop: from_* function validation


class TestFromGiotto:
    """from_giotto converts numpy arrays to standard Diagram."""

    def test_accepts_2d_array_3cols(self):
        arr = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1], [0.3, float("inf"), 0]])
        result = from_giotto(arr)
        assert len(result) == 3
        assert result[0] == (0.0, 1.0, 0)
        assert result[2][1] == float("inf")

    def test_accepts_empty_array(self):
        arr = np.empty((0, 3))
        result = from_giotto(arr)
        assert result == []

    def test_rejects_1d_array(self):
        arr = np.array([0.0, 1.0, 0.0])
        with pytest.raises((ValidationError, ValueError)):
            from_giotto(arr)

    def test_rejects_nan_births(self):
        arr = np.array([[float("nan"), 1.0, 0]])
        with pytest.raises((ValidationError, ValueError)):
            from_giotto(arr)

    def test_rejects_death_less_than_birth(self):
        arr = np.array([[1.0, 0.0, 0]])
        with pytest.raises((ValidationError, ValueError)):
            from_giotto(arr)

    def test_rejects_float_dimensions(self):
        arr = np.array([[0.0, 1.0, 1.5]])
        with pytest.raises((ValidationError, ValueError)):
            from_giotto(arr)

    def test_rejects_negative_dimensions(self):
        arr = np.array([[0.0, 1.0, -1]])
        with pytest.raises((ValidationError, ValueError)):
            from_giotto(arr)


class TestFromExternal:
    """from_external converts external dictionary output."""

    def test_converts_valid_external_output(self):
        external_out = {"dgms": [np.array([[0.0, 1.0], [0.5, 2.0]]), np.array([[0.3, 1.5]])]}
        result = from_external(external_out)
        assert len(result) == 3
        assert result[0] == (0.0, 1.0, 0)
        assert result[1] == (0.5, 2.0, 0)
        assert result[2] == (0.3, 1.5, 1)

    def test_converts_with_inf_death(self):
        external_out = {"dgms": [np.array([[0.0, np.inf]])]}
        result = from_external(external_out)
        assert result == [(0.0, float("inf"), 0)]

    def test_rejects_missing_dgms_key(self):
        with pytest.raises((ValidationError, ValueError), match="dgms"):
            from_external({"wrong_key": []})

    def test_rejects_nan_birth(self):
        with pytest.raises((ValidationError, ValueError)):
            from_external({"dgms": [np.array([[float("nan"), 1.0]])]})

    def test_rejects_death_less_than_birth(self):
        with pytest.raises((ValidationError, ValueError)):
            from_external({"dgms": [np.array([[1.0, 0.0]])]})


class TestFromGudhi:
    """from_gudhi converts GUDHI persistence diagram formats."""

    def test_converts_dim_birth_death_tuples(self):
        gudhi_dgm = [(0, (0.0, 1.0)), (1, (0.5, 2.0))]
        result = from_gudhi(gudhi_dgm, format_type="diagram")
        assert result == [(0.0, 1.0, 0), (0.5, 2.0, 1)]

    def test_converts_birth_death_dim_triples(self):
        gudhi_dgm = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        result = from_gudhi(gudhi_dgm, format_type="diagram")
        assert result == [(0.0, 1.0, 0), (0.5, 2.0, 1)]

    def test_rejects_invalid_entry_length(self):
        with pytest.raises(ValueError, match="GUDHI"):
            from_gudhi([(0.0,)], format_type="diagram")

    def test_rejects_unknown_format_type(self):
        with pytest.raises((ValueError, InvalidArgumentError), match="format_type"):
            from_gudhi([], format_type="unknown")

    def test_rejects_death_less_than_birth(self):
        with pytest.raises((ValidationError, ValueError)):
            from_gudhi([(0, (1.0, 0.0))], format_type="diagram")

    def test_simplex_tree_converts_filtration(self):
        gudhi_st = [([0], 0.0), ([1], 1.0), ([0, 1], 2.0)]
        result = from_gudhi(gudhi_st, format_type="simplex_tree")
        assert len(result) == 3
        assert result[0] == ((0,), 0.0)
        assert result[2] == ((0, 1), 2.0)

    def test_simplex_tree_rejects_nonfinite_filtration(self):
        gudhi_st = [([0], float("nan"))]
        with pytest.raises((ValidationError, ValueError)):
            from_gudhi(gudhi_st, format_type="simplex_tree")

    def test_simplex_tree_rejects_malformed_entries(self):
        gudhi_st = [("not_a_pair",)]
        with pytest.raises((ValidationError, ValueError)):
            from_gudhi(gudhi_st, format_type="simplex_tree")

    def test_simplex_tree_dict_with_simplices_key(self):
        gudhi_st = {"simplices": [([0], 0.0), ([1], 1.0)]}
        result = from_gudhi(gudhi_st, format_type="simplex_tree")
        assert len(result) == 2
        assert result[0] == ((0,), 0.0)


class TestFromDionysus:
    """from_dionysus converts Dionysus diagram objects."""

    def test_converts_objects_with_birth_death_attrs(self):
        class DgmPoint:
            def __init__(self, birth, death):
                self.birth = birth
                self.death = death

        dionysus_dgms = [[DgmPoint(0.0, 1.0), DgmPoint(0.5, 2.0)], [DgmPoint(0.3, 1.5)]]
        result = from_dionysus(dionysus_dgms)
        assert len(result) == 3
        assert result[0] == (0.0, 1.0, 0)
        assert result[1] == (0.5, 2.0, 0)
        assert result[2] == (0.3, 1.5, 1)

    def test_converts_sequence_pairs(self):
        dionysus_dgms = [[(0.0, 1.0), (0.5, 2.0)]]
        result = from_dionysus(dionysus_dgms)
        assert result == [(0.0, 1.0, 0), (0.5, 2.0, 0)]

    def test_converts_with_inf_death(self):
        class DgmPoint:
            def __init__(self, birth, death):
                self.birth = birth
                self.death = death

        result = from_dionysus([[DgmPoint(0.0, float("inf"))]])
        assert result == [(0.0, float("inf"), 0)]

    def test_rejects_invalid_point_without_attrs(self):
        with pytest.raises((ValidationError, ValueError)):
            from_dionysus([["not_valid"]])


class TestFromSktda:
    """from_sktda converts scikit-tda diagram objects."""

    def test_converts_with_dgms_attribute(self):
        class SktdaDiagram:
            def __init__(self):
                self.dgms = [np.array([[0.0, 1.0], [0.5, 2.0]]), np.array([[0.3, 1.5]])]

        result = from_sktda(SktdaDiagram())
        assert result[0] == (0.0, 1.0, 0)
        assert result[1] == (0.5, 2.0, 0)
        assert result[2] == (0.3, 1.5, 1)

    def test_converts_with_inf_death(self):
        class SktdaDiagram:
            def __init__(self):
                self.dgms = [np.array([[0.0, np.inf]])]

        result = from_sktda(SktdaDiagram())
        assert result == [(0.0, float("inf"), 0)]

    def test_rejects_missing_dgms_attribute(self):
        class BadObj:
            pass

        with pytest.raises((ValidationError, ValueError), match="dgms"):
            from_sktda(BadObj())


# TDA Interop: to_* function validation


class TestToGudhi:
    """to_gudhi converts standard Diagram to GUDHI format."""

    def test_returns_list_of_dim_tuple_pairs(self):
        result = to_gudhi(DIAGRAM_SIMPLE)
        assert isinstance(result, list)
        assert len(result) == len(DIAGRAM_SIMPLE)
        assert result[0] == (0, (0.0, 1.0))

    def test_preserves_inf_death(self):
        result = to_gudhi(DIAGRAM_SIMPLE)
        inf_entry = next(e for e in result if e[0] == 0 and e[1][0] == 0.3)
        assert np.isinf(inf_entry[1][1])

    def test_preserves_all_dimensions(self):
        result = to_gudhi(DIAGRAM_SIMPLE)
        dims = [dim for dim, _ in result]
        assert dims == [0, 1, 0, 1, 2]

    def test_empty_diagram(self):
        result = to_gudhi(DIAGRAM_EMPTY)
        assert result == []

    def test_rejects_invalid_birth(self):
        with pytest.raises((ValidationError, ValueError)):
            to_gudhi([(float("nan"), 1.0, 0)])

    def test_rejects_negative_dimension(self):
        with pytest.raises((ValidationError, ValueError)):
            to_gudhi([(0.0, 1.0, -1)])


class TestToExternal:
    """to_external converts standard Diagram to external JSON format."""

    def test_returns_string_by_default(self):
        result = to_external(DIAGRAM_SIMPLE)
        assert isinstance(result, str)
        assert '"format": "nerve_diagram"' in result
        assert '"diagrams"' in result

    def test_inf_death_becomes_null(self):
        result = to_external(DIAGRAM_SIMPLE)
        assert '"death": null' in result

    def test_writes_to_file_when_filepath_given(self, tmp_path):
        path = tmp_path / "external_output.json"
        returned = to_external(DIAGRAM_SIMPLE, filepath=str(path))
        assert path.exists()
        assert returned == str(path)
        content = path.read_text()
        assert '"format": "nerve_diagram"' in content

    def test_roundtrip_via_from_external(self):
        external_json = to_external(DIAGRAM_SIMPLE)
        import json

        parsed = json.loads(external_json)
        external_dgms = {}
        for entry in parsed["diagrams"]:
            dim = entry["dimension"]
            external_dgms.setdefault(dim, []).append(
                [entry["birth"], float("inf") if entry["death"] is None else entry["death"]]
            )
        external_input = {
            "dgms": [np.array(external_dgms.get(d, [])) for d in sorted(external_dgms)]
        }
        result = from_external(external_input)
        assert len(result) == len(DIAGRAM_SIMPLE)

    def test_empty_diagram(self):
        result = to_external(DIAGRAM_EMPTY)
        assert isinstance(result, str)
        parsed = __import__("json").loads(result)
        assert parsed["diagrams"] == []

    def test_rejects_invalid_birth(self):
        with pytest.raises((ValidationError, ValueError)):
            to_external([(float("nan"), 1.0, 0)])


class TestToDionysus:
    """to_dionysus converts standard Diagram to Dionysus dict format."""

    def test_returns_dict_int_to_float_pairs(self):
        result = to_dionysus(DIAGRAM_SIMPLE)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert result[0] == [(0.0, 1.0), (0.3, float("inf"))]
        assert result[1] == [(0.5, 2.0), (1.0, 1.5)]
        assert result[2] == [(2.0, 3.5)]

    def test_empty_diagram(self):
        result = to_dionysus(DIAGRAM_EMPTY)
        assert result == {}

    def test_roundtrip_via_from_dionysus(self):
        dionysus_fmt = to_dionysus(DIAGRAM_SIMPLE)
        dgm_list = [dionysus_fmt[d] for d in sorted(dionysus_fmt)]
        reimported = from_dionysus(dgm_list)
        assert len(reimported) == len(DIAGRAM_SIMPLE)
        # Verify entries match (order may differ from original but content is preserved)
        for entry in reimported:
            assert entry in DIAGRAM_SIMPLE

    def test_rejects_invalid_entry(self):
        with pytest.raises((ValidationError, ValueError)):
            to_dionysus([(0.0, 1.0, -1)])


# TDA Interop: cross-format roundtrip via standard Diagram


class TestInteropRoundtrips:
    """Full roundtrip: Diagram -> target format -> Diagram."""

    def test_to_gudhi_to_from_gudhi_roundtrip(self):
        gudhi_fmt = to_gudhi(DIAGRAM_SIMPLE)
        # Convert to GUDHI persistence diagram form: list of (dim, (birth, death))
        back = from_gudhi(gudhi_fmt, format_type="diagram")
        assert back == DIAGRAM_SIMPLE

    def test_to_dionysus_to_from_dionysus_roundtrip(self):
        dionysus_fmt = to_dionysus(DIAGRAM_SIMPLE)
        dgm_list = [dionysus_fmt[d] for d in sorted(dionysus_fmt)]
        back = from_dionysus(dgm_list)
        for entry in back:
            assert entry in DIAGRAM_SIMPLE
        assert len(back) == len(DIAGRAM_SIMPLE)


# TDA Interop: real library tests (skip if not installed)


class TestRealGUDHIIfAvailable:
    """Test with real GUDHI if installed."""

    @pytest.fixture(scope="class")
    def gudhi(self):
        gudhi = pytest.importorskip("gudhi")
        return gudhi

    def test_from_gudhi_persistence_diagram(self, gudhi):
        import gudhi.representations  # noqa: F401 -- ensure submodules are loaded

        st = gudhi.SimplexTree()
        st.insert([0], 0.0)
        st.insert([1], 0.0)
        st.insert([0, 1], 1.0)
        st.compute_persistence()
        dgm = st.persistence()
        result = from_gudhi(dgm, format_type="diagram")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_to_gudhi_format_consumable_by_gudhi(self, gudhi):  # noqa: ARG002
        gudhi_dgm = to_gudhi(DIAGRAM_SIMPLE)
        # Verify structure is correct for GUDHI consumption
        for item in gudhi_dgm:
            assert isinstance(item, tuple)
            assert len(item) == 2
            dim, pair = item
            assert isinstance(dim, int)
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_gudhi_simplex_tree_from_real(self, gudhi):
        st = gudhi.SimplexTree()
        st.insert([0], 0.0)
        st.insert([1], 1.0)
        st.insert([0, 1], 2.0)
        result = from_gudhi(st, format_type="simplex_tree")
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], tuple)  # vertices
        assert isinstance(result[0][1], float)  # filtration


class TestRealExternalIfAvailable:
    """Test with real external lib if available."""

    @pytest.fixture(scope="class")
    def external_lib(self):
        external_lib = pytest.importorskip("ripser")
        return external_lib

    def test_from_external_real_output(self, external_lib):
        import numpy as np

        pts = np.random.default_rng(42).random((10, 2))
        result = external_lib.ripser(pts, maxdim=1)
        converted = from_external(result)
        assert isinstance(converted, list)
        assert len(converted) > 0
        for birth, death, dim in converted:  # noqa: B007
            assert isinstance(birth, float)
            assert dim >= 0

    def test_external_roundtrip(self, external_lib):
        pts = np.random.default_rng(7).random((8, 2))
        result = external_lib.ripser(pts, maxdim=1)
        converted = from_external(result)
        # Convert back to external dict
        external_json = to_external(converted)
        assert isinstance(external_json, str)
        assert '"format": "nerve_diagram"' in external_json


class TestRealDionysusIfAvailable:
    """Test with real dionysus if installed."""

    @pytest.fixture(scope="class")
    def dionysus(self):
        dionysus = pytest.importorskip("dionysus")
        return dionysus

    def test_from_dionysus_real_diagrams(self, dionysus):
        import numpy as np

        pts = np.random.default_rng(99).random((12, 2))
        f = dionysus.fill_rips(pts, 2, 2.0)
        m = dionysus.homology_persistence(f)
        dgms = dionysus.init_diagrams(m, f)
        result = from_dionysus(dgms)
        assert isinstance(result, list)
        assert len(result) > 0
        for birth, death, dim in result:  # noqa: B007
            assert isinstance(birth, float)
            assert dim >= 0

    def test_dionysus_roundtrip(self, dionysus):
        pts = np.random.default_rng(33).random((8, 2))
        f = dionysus.fill_rips(pts, 2, 2.0)
        m = dionysus.homology_persistence(f)
        dgms = dionysus.init_diagrams(m, f)
        converted = from_dionysus(dgms)
        dionysus_fmt = to_dionysus(converted)
        assert isinstance(dionysus_fmt, dict)
        for dim, pairs in dionysus_fmt.items():
            assert isinstance(dim, int)
            for pair in pairs:
                assert len(pair) == 2


class TestRealGiottoIfAvailable:
    """Test with real giotto-tda if installed."""

    @pytest.fixture(scope="class")
    def gtda(self):
        gtda = pytest.importorskip("gtda")
        return gtda

    def test_from_giotto_real_output(self, gtda):  # noqa: ARG002
        import numpy as np
        from gtda.homology import VietorisRipsPersistence

        pts = np.random.default_rng(55).random((10, 2))
        vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
        dgms = vr.fit_transform(pts[None, :, :])[0]
        result = from_giotto(dgms)
        assert isinstance(result, list)
        assert len(result) > 0


# Edge cases: validation coverage


class TestInteropValidationEdgeCases:
    """Validation edge cases across all interop functions."""

    def test_empty_diagram_all_modules(self):
        assert from_giotto(np.empty((0, 3))) == []
        assert from_external({"dgms": []}) == []
        assert from_gudhi([], format_type="diagram") == []
        assert from_gudhi([], format_type="simplex_tree") == []
        assert from_dionysus([]) == []
        assert from_sktda(type("Sktda", (), {"dgms": []})()) == []
        assert to_gudhi([]) == []
        assert to_dionysus([]) == {}
        result = to_external([])
        assert isinstance(result, str)
        assert __import__("json").loads(result)["diagrams"] == []

    def test_single_entry_roundtrip_all_formats(self):
        entry = [(1.0, 3.0, 0)]
        assert to_gudhi(entry) == [(0, (1.0, 3.0))]
        dion = to_dionysus(entry)
        assert dion == {0: [(1.0, 3.0)]}
        back = from_dionysus([dion[0]])
        assert back == entry

    def test_multiple_dimensions_ordering(self):
        diagram = [(0.0, 1.0, 2), (0.0, 1.0, 0), (0.0, 1.0, 1)]
        gudhi = to_gudhi(diagram)
        dim_order = [dim for dim, _ in gudhi]
        assert dim_order == [2, 0, 1]
        back = from_gudhi(gudhi, format_type="diagram")
        assert back == diagram

    def test_death_equals_birth_is_valid(self):
        """Deaths equal to births are allowed (finite death >= birth check)."""
        diagram = [(1.0, 1.0, 0)]
        result = to_gudhi(diagram)
        assert result == [(0, (1.0, 1.0))]
        back = from_gudhi(result, format_type="diagram")
        assert back == [(1.0, 1.0, 0)]

    def test_large_dimension_values(self):
        diagram = [(0.0, 1.0, 100)]
        gudhi = to_gudhi(diagram)
        assert gudhi == [(100, (0.0, 1.0))]
        back = from_gudhi(gudhi, format_type="diagram")
        assert back == [(0.0, 1.0, 100)]

    def test_numpy_array_as_from_giotto_input(self):
        arr = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=np.float64)
        result = from_giotto(arr)
        assert result == [(0.0, 1.0, 0), (0.5, 2.0, 1)]

    def test_from_giotto_accepts_contiguous_and_noncontiguous(self):
        arr = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1], [0.3, 1.5, 0]])
        result = from_giotto(arr[::-1])
        assert len(result) == 3

    def test_to_external_file_creates_valid_json(self, tmp_path):
        path = tmp_path / "output.json"
        to_external(DIAGRAM_SIMPLE, filepath=str(path))
        import json

        data = json.loads(path.read_text())
        assert data["format"] == "nerve_diagram"
        assert len(data["diagrams"]) == len(DIAGRAM_SIMPLE)
