"""Tests for pipeline utility module."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve.exceptions import InvalidArgumentError
from pynerve.pipeline import (
    ConditionalPipeline,
    ParallelPipeline,
    Pipeline,
    analysis_pipeline,
    vr_pipeline,
)

# pipeline.py


def _square(x):
    return x * x


def _add_one(x):
    return x + 1


def _to_string(x):
    return str(x)


class TestPipelineConstruction:
    def test_empty_pipeline(self):
        p = Pipeline()
        assert len(p) == 0
        assert p.names() == []

    def test_single_named_step(self):
        p = Pipeline(("sq", _square))
        assert len(p) == 1
        assert p.names() == ["sq"]

    def test_single_bare_callable(self):
        p = Pipeline(_square)
        assert len(p) == 1
        assert p.names() == ["_step_0"]

    def test_multiple_named_steps(self):
        p = Pipeline(("sq", _square), ("plus", _add_one))
        assert len(p) == 2
        assert p.names() == ["sq", "plus"]

    def test_mixed_named_and_bare(self):
        p = Pipeline(("sq", _square), _add_one)
        assert len(p) == 2
        assert p.names() == ["sq", "_step_1"]

    def test_invalid_step_type_raises(self):
        with pytest.raises(InvalidArgumentError, match="step"):
            Pipeline(42)  # type: ignore[arg-type]

    def test_invalid_tuple_length_raises(self):
        with pytest.raises(InvalidArgumentError, match="tuple"):
            Pipeline(("a", "b", "c"))  # type: ignore[arg-type]


class TestPipelineCall:
    def test_empty_pipeline_noop(self):
        p = Pipeline()
        result = p(42)
        assert result == 42

    def test_single_step(self):
        p = Pipeline(_square)
        assert p(5) == 25

    def test_multi_step_composition(self):
        p = Pipeline(("sq", _square), ("add", _add_one))
        assert p(3) == 10  # 3*3 + 1

    def test_chained_transformations(self):
        p = Pipeline(("add", _add_one), ("sq", _square), ("str", _to_string))
        assert p(3) == "16"  # (3+1)^2 = 16 -> "16"

    def test_with_numpy_data(self):
        p = Pipeline(_square)
        arr = np.array([1, 2, 3])
        result = p(arr)
        assert np.array_equal(result, np.array([1, 4, 9]))

    def test_with_dict_data(self):
        def add_field(d):
            d["processed"] = True
            return d

        p = Pipeline(add_field)
        data = {"x": 1}
        result = p(data)
        assert result == {"x": 1, "processed": True}

    def test_pipeline_is_callable(self):
        p = Pipeline(_square)
        assert callable(p)


class TestPipelineGetItem:
    def test_by_index(self):
        p = Pipeline(("a", _square), ("b", _add_one))
        func = p[0]
        assert func(3) == 9

    def test_by_name(self):
        p = Pipeline(("sq", _square), ("add", _add_one))
        func = p["sq"]
        assert func(3) == 9

    def test_by_slice(self):
        p = Pipeline(("a", _square), ("b", _add_one), ("c", _to_string))
        sub = p[0:2]
        assert isinstance(sub, Pipeline)
        assert len(sub) == 2
        assert sub(3) == 10  # 3*3 + 1

    def test_by_slice_single(self):
        p = Pipeline(("a", _square), ("b", _add_one))
        sub = p[0:1]
        assert len(sub) == 1
        assert sub(4) == 16

    def test_missing_name_raises(self):
        p = Pipeline(("a", _square))
        with pytest.raises(KeyError):
            p["nonexistent"]


class TestPipelineAddStep:
    def test_add_step_returns_self(self):
        p = Pipeline()
        result = p.add_step("sq", _square)
        assert result is p

    def test_add_step_extends_pipeline(self):
        p = Pipeline()
        p.add_step("sq", _square)
        assert len(p) == 1
        assert p(3) == 9

    def test_duplicate_name_raises(self):
        p = Pipeline(("sq", _square))
        with pytest.raises(InvalidArgumentError, match="already exists"):
            p.add_step("sq", _add_one)

    def test_duplicate_name_replace(self):
        p = Pipeline(("sq", _square))
        p.add_step("sq", _add_one, replace=True)
        assert p(3) == 4  # _add_one, not _square

    def test_non_callable_raises(self):
        p = Pipeline()
        with pytest.raises(InvalidArgumentError, match="callable"):
            p.add_step("bad", 42)  # type: ignore[arg-type]

    def test_empty_name_raises(self):
        p = Pipeline()
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            p.add_step("", _square)


class TestPipelineInsertStep:
    def test_insert_at_beginning(self):
        p = Pipeline(("b", _add_one))
        p.insert_step(0, "a", _square)
        assert p.names() == ["a", "b"]
        assert p(3) == 10  # square then add_one

    def test_insert_in_middle(self):
        p = Pipeline(("a", _square), ("c", _to_string))
        p.insert_step(1, "b", _add_one)
        assert p.names() == ["a", "b", "c"]
        assert p(3) == "10"  # square, add_one, to_string

    def test_insert_at_end(self):
        p = Pipeline(("a", _square))
        p.insert_step(1, "b", _add_one)
        assert p.names() == ["a", "b"]

    def test_duplicate_name_raises(self):
        p = Pipeline(("a", _square))
        with pytest.raises(InvalidArgumentError, match="already exists"):
            p.insert_step(0, "a", _add_one)

    def test_non_integer_index_raises(self):
        p = Pipeline(("a", _square))
        with pytest.raises(InvalidArgumentError, match="integer"):
            p.insert_step("start", "b", _add_one)  # type: ignore[arg-type]


class TestPipelineRemoveStep:
    def test_remove_existing(self):
        p = Pipeline(("a", _square), ("b", _add_one))
        p.remove_step("a")
        assert p.names() == ["b"]
        assert p(3) == 4

    def test_remove_missing_raises(self):
        p = Pipeline(("a", _square))
        with pytest.raises(KeyError):
            p.remove_step("nonexistent")


class TestPipelinePopStep:
    def test_pop_returns_name_and_func(self):
        p = Pipeline(("sq", _square))
        name, func = p.pop_step("sq")
        assert name == "sq"
        assert func(3) == 9
        assert len(p) == 0

    def test_pop_missing_raises(self):
        p = Pipeline(("a", _square))
        with pytest.raises(KeyError):
            p.pop_step("x")


class TestPipelineCopy:
    def test_copy_independent(self):
        p = Pipeline(("sq", _square))
        c = p.copy()
        assert c.names() == p.names()
        assert c(3) == p(3)
        # Modify original, copy unaffected
        p.add_step("add", _add_one)
        assert len(c) == 1
        assert len(p) == 2

    def test_copy_preserves_behaviour(self):
        p = Pipeline(("sq", _square), ("add", _add_one))
        c = p.copy()
        assert c(3) == p(3) == 10


class TestPipelineToList:
    def test_to_list_structure(self):
        p = Pipeline(("a", _square), ("b", _add_one))
        items = p.to_list()
        assert len(items) == 2
        assert items[0] == ("a", _square)
        assert items[1] == ("b", _add_one)


class TestPipelineIter:
    def test_iter_yields_name_func_pairs(self):
        p = Pipeline(("a", _square), ("b", _add_one))
        items = list(p)
        assert items == [("a", _square), ("b", _add_one)]


class TestPipelineRepr:
    def test_repr_contains_step_names(self):
        p = Pipeline(("square", _square))
        r = repr(p)
        assert "square" in r

    def test_repr_empty_pipeline(self):
        p = Pipeline()
        r = repr(p)
        assert "Pipeline" in r


class TestConditionalPipeline:
    def test_simple_steps_chain(self):
        cp = ConditionalPipeline()
        cp.add_step("sq", _square)
        cp.add_step("add", _add_one)
        assert cp(3) == 10

    def test_conditional_true_branch(self):
        cp = ConditionalPipeline()
        cp.add_step("sq", _square)
        cp.add_conditional(
            "check",
            condition=lambda x: x > 5,
            if_true=lambda x: x * 10,
            if_false=lambda x: x - 1,
        )
        # 3*3 = 9 > 5 -> if_true: 9*10 = 90
        assert cp(3) == 90

    def test_conditional_false_branch(self):
        cp = ConditionalPipeline()
        cp.add_conditional(
            "check",
            condition=lambda x: x > 5,
            if_true=lambda x: x * 10,
            if_false=lambda x: x - 1,
        )
        # 3 <= 5 -> if_false: 3-1 = 2
        assert cp(3) == 2

    def test_conditional_no_false(self):
        cp = ConditionalPipeline()
        cp.add_conditional(
            "check",
            condition=lambda x: x > 5,
            if_true=lambda x: x * 10,
        )
        # 3 <= 5, no if_false -> data passes through unchanged
        assert cp(3) == 3

    def test_duplicate_name_raises(self):
        cp = ConditionalPipeline()
        cp.add_step("a", _square)
        with pytest.raises(InvalidArgumentError, match="already exists"):
            cp.add_step("a", _add_one)

    def test_non_callable_condition_raises(self):
        cp = ConditionalPipeline()
        with pytest.raises(TypeError, match="condition"):
            cp.add_conditional("x", condition=42, if_true=_square)  # type: ignore[arg-type]

    def test_non_callable_if_false_raises(self):
        cp = ConditionalPipeline()
        with pytest.raises(TypeError, match="if_false"):
            cp.add_conditional("x", condition=lambda _x: True, if_true=_square, if_false=42)  # type: ignore[arg-type]

    def test_remove_step(self):
        cp = ConditionalPipeline()
        cp.add_step("a", _square)
        cp.add_step("b", _add_one)
        cp.remove_step("a")
        assert cp.names() == ["b"]
        assert cp(3) == 4

    def test_names(self):
        cp = ConditionalPipeline()
        cp.add_step("first", _square)
        cp.add_step("second", _add_one)
        assert cp.names() == ["first", "second"]

    def test_repr(self):
        cp = ConditionalPipeline()
        cp.add_step("sq", _square)
        r = repr(cp)
        assert "ConditionalPipeline" in r


class TestParallelPipeline:
    def test_basic_parallel_execution(self):
        p1 = Pipeline(_square)
        p2 = Pipeline(_add_one)
        pp = ParallelPipeline(
            pipelines={"sq": p1, "add": p2},
            combine_fn=lambda results: results,
        )
        result = pp(3)
        assert result == {"sq": 9, "add": 4}

    def test_combine_fn_reduces(self):
        p1 = Pipeline(_square)
        p2 = Pipeline(_add_one)
        pp = ParallelPipeline(
            pipelines={"a": p1, "b": p2},
            combine_fn=lambda r: sum(r.values()),
        )
        result = pp(3)
        assert result == 13  # 9 + 4

    def test_empty_pipelines_raises(self):
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            ParallelPipeline(pipelines={}, combine_fn=lambda x: x)  # type: ignore[arg-type]

    def test_non_callable_combine_fn_raises(self):
        p = Pipeline(_square)
        with pytest.raises(InvalidArgumentError, match="callable"):
            ParallelPipeline(pipelines={"a": p}, combine_fn=42)  # type: ignore[arg-type]

    def test_non_callable_pipeline_raises(self):
        with pytest.raises(InvalidArgumentError, match="callable"):
            ParallelPipeline(pipelines={"a": 42}, combine_fn=lambda x: x)  # type: ignore[arg-type, dict-item]

    def test_empty_pipeline_name_raises(self):
        p = Pipeline(_square)
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            ParallelPipeline(pipelines={"": p}, combine_fn=lambda x: x)

    def test_repr(self):
        p = Pipeline(_square)
        pp = ParallelPipeline(pipelines={"sq": p}, combine_fn=lambda x: x)
        r = repr(pp)
        assert "ParallelPipeline" in r


class TestVRPipeline:
    def test_creates_pipeline(self):
        p = vr_pipeline(max_dim=1)
        assert isinstance(p, Pipeline)
        assert len(p) >= 1

    def test_with_min_persistence_adds_filter(self):
        p = vr_pipeline(max_dim=1, min_persistence=0.1)
        assert len(p) == 2


class TestAnalysisPipeline:
    def test_creates_pipeline(self):
        p = analysis_pipeline(compute_fn=_square)
        assert isinstance(p, Pipeline)
        assert len(p) >= 1

    def test_with_vector_representation(self):
        p = analysis_pipeline(compute_fn=_square, representations=["vector"])
        assert len(p) >= 2

    def test_unknown_representation_raises(self):
        with pytest.raises(InvalidArgumentError, match="unknown"):
            analysis_pipeline(compute_fn=_square, representations=["imaginary"])

    def test_diagram_only_is_single_step(self):
        p = analysis_pipeline(compute_fn=_square, representations=["diagram"])
        assert len(p) == 1

    def test_non_callable_compute_raises(self):
        with pytest.raises(TypeError, match="callable"):
            analysis_pipeline(compute_fn=42)  # type: ignore[arg-type]
