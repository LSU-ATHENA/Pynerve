"""Targeted tests for diff/_composite_loss, _pipeline_core, and diff helpers.

Exercises TopologyLoss (construct/forward with all components), Pipeline
(construct/call/getitem/iter/len/add/insert/remove/pop/copy), and diff
helper functions (_validate_non_negative_scalar, module imports).
"""

from __future__ import annotations


import pytest
import torch
from _test_helpers import make_diag_2d, make_diag_3d

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestTopologyLoss:
    """Covers diff/_composite_loss.py -- TopologyLoss."""

    def test_construct(self):
        from pynerve.diff._composite_loss import TopologyLoss
        loss = TopologyLoss(
            wasserstein_weight=1.0, betti_weight=0.1,
            complexity_weight=0.01, stability_weight=0.0
        )
        assert loss.wasserstein_weight == 1.0
        assert loss.betti_weight == 0.1

    def test_construct_negative_weight(self):
        from pynerve.diff._composite_loss import TopologyLoss
        with pytest.raises(ValueError, match="non-negative"):
            TopologyLoss(wasserstein_weight=-1.0)

    def test_forward(self):
        from pynerve.diff._composite_loss import TopologyLoss
        loss = TopologyLoss(wasserstein_weight=1.0, complexity_weight=0.01)
        pred = make_diag_2d(5)
        target = make_diag_2d(4)
        result = loss.forward(pred, target)
        assert "total" in result
        assert result["total"] >= 0

    def test_forward_with_betti(self):
        from pynerve.diff._composite_loss import TopologyLoss
        loss = TopologyLoss(wasserstein_weight=0.0, betti_weight=0.1)
        # BettiNumberLoss requires 3-column diagrams (birth, death, dim)
        pred = make_diag_3d(5)
        target = make_diag_3d(4)
        betti = torch.tensor([2.0, 1.0])
        result = loss.forward(pred, target, target_betti=betti)
        assert "betti" in result
        assert "total" in result

    def test_forward_all_components(self):
        from pynerve.diff._composite_loss import TopologyLoss
        loss = TopologyLoss(
            wasserstein_weight=1.0, betti_weight=0.1,
            complexity_weight=0.01, stability_weight=0.0
        )
        # BettiNumberLoss requires 3-column diagrams (birth, death, dim)
        pred = make_diag_3d(5)
        target = make_diag_3d(4)
        betti = torch.tensor([2.0, 1.0])
        result = loss.forward(pred, target, target_betti=betti)
        assert "wasserstein" in result
        assert "betti" in result
        assert "complexity" in result
        assert "total" in result

    def test_forward_zero_weights(self):
        from pynerve.diff._composite_loss import TopologyLoss
        loss = TopologyLoss(
            wasserstein_weight=0.0, betti_weight=0.0,
            complexity_weight=0.0, stability_weight=0.0
        )
        pred = make_diag_2d(3)
        target = make_diag_2d(3)
        result = loss.forward(pred, target)
        assert result["total"].item() == 0.0


class TestPipeline:
    """Covers _pipeline_core.py -- Pipeline."""

    def test_construct_empty(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline()
        assert len(p) == 0

    def test_construct_named(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("step1", lambda x: x + 1), ("step2", lambda x: x * 2))
        assert len(p) == 2

    def test_construct_bare_callable(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(lambda x: x + 1, lambda x: x * 2)
        assert len(p) == 2

    def test_construct_invalid_tuple(self):
        from pynerve._pipeline_core import Pipeline
        with pytest.raises(Exception, match="Invalid"):
            Pipeline(("only_name",))

    def test_construct_invalid_type(self):
        from pynerve._pipeline_core import Pipeline
        with pytest.raises(Exception, match="Invalid"):
            Pipeline(42)

    def test_call(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("add", lambda x: x + 1), ("mul", lambda x: x * 2))
        assert p(3) == 8  # (3+1)*2

    def test_getitem_int(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        assert callable(p[0])

    def test_getitem_str(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        assert callable(p["s2"])

    def test_getitem_slice(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2), ("s3", lambda x: x + 1))
        sub = p[1:]
        assert isinstance(sub, Pipeline)
        assert len(sub) == 2

    def test_iter(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        items = list(p)
        assert len(items) == 2
        assert items[0][0] == "s1"

    def test_add_step(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline()
        p.add_step("s1", lambda x: x + 1)
        assert len(p) == 1

    def test_add_step_duplicate(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x))
        with pytest.raises(Exception, match="already exists"):
            p.add_step("s1", lambda x: x * 2)

    def test_add_step_replace(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x + 1))
        p.add_step("s1", lambda x: x * 10, replace=True)
        assert p(5) == 50

    def test_add_step_invalid_name(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline()
        with pytest.raises(Exception, match="non-empty"):
            p.add_step("", lambda x: x)

    def test_add_step_not_callable(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline()
        with pytest.raises(Exception, match="callable"):
            p.add_step("s1", 42)

    def test_insert_step(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x + 1), ("s3", lambda x: x * 3))
        p.insert_step(1, "s2", lambda x: x * 2)
        assert p.names() == ["s1", "s2", "s3"]

    def test_insert_step_invalid_index(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x))
        with pytest.raises(Exception, match="integer"):
            p.insert_step("bad", "s2", lambda x: x)

    def test_insert_step_duplicate(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x))
        with pytest.raises(Exception, match="already exists"):
            p.insert_step(0, "s1", lambda x: x * 2)

    def test_remove_step(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        p.remove_step("s1")
        assert len(p) == 1
        assert p.names() == ["s2"]

    def test_remove_step_missing(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x))
        with pytest.raises(KeyError):
            p.remove_step("bad")

    def test_pop_step(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        name, func = p.pop_step("s1")
        assert name == "s1"
        assert callable(func)
        assert len(p) == 1

    def test_names(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        assert p.names() == ["s1", "s2"]

    def test_to_list(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x), ("s2", lambda x: x * 2))
        items = p.to_list()
        assert len(items) == 2
        assert items[0][0] == "s1"

    def test_copy(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x + 1))
        p2 = p.copy()
        assert len(p2) == 1
        p2.add_step("s2", lambda x: x * 2)
        assert len(p) == 1
        assert len(p2) == 2

    def test_repr(self):
        from pynerve._pipeline_core import Pipeline
        p = Pipeline(("s1", lambda x: x + 1))
        r = repr(p)
        assert "Pipeline" in r
        assert "s1" in r


class TestValidateRepresentations:
    """Covers _pipeline_core.py -- _validate_representations."""

    def test_valid(self):
        from pynerve._pipeline_core import _validate_representations
        result = _validate_representations(["landscape", "image"])
        assert result == ["landscape", "image"]

    def test_empty(self):
        from pynerve._pipeline_core import _validate_representations
        with pytest.raises(Exception, match="non-empty"):
            _validate_representations([])

    def test_not_sequence(self):
        from pynerve._pipeline_core import _validate_representations
        with pytest.raises(Exception, match="sequence"):
            _validate_representations("not_a_list")

    def test_empty_string(self):
        from pynerve._pipeline_core import _validate_representations
        with pytest.raises(Exception, match="non-empty strings"):
            _validate_representations(["valid", ""])

    def test_duplicates(self):
        from pynerve._pipeline_core import _validate_representations
        with pytest.raises(Exception, match="unique"):
            _validate_representations(["a", "a"])


class TestDiffHelpers:
    """Covers diff/_ph_representations, diff/_diagram_distances, and diff/_loss_helpers."""

    def test_import_persistence_landscape_fn(self):
        from pynerve.diff._ph_representations import compute_persistence_landscape
        assert compute_persistence_landscape is not None

    def test_import_persistence_image_fn(self):
        from pynerve.diff._ph_representations import persistence_image
        assert persistence_image is not None

    def test_import_persistence_loss(self):
        from pynerve.diff._diagram_distances import PersistenceLoss
        assert PersistenceLoss is not None

    def test_import_loss_helpers(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar
        result = _validate_non_negative_scalar("test", 0.5)
        assert result == 0.5

    def test_loss_helpers_negative(self):
        from pynerve.diff._loss_helpers import _validate_non_negative_scalar
        with pytest.raises(ValueError, match="non-negative"):
            _validate_non_negative_scalar("test", -1.0)
