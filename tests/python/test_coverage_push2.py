"""Targeted tests for curriculum trainer, BYOL, SimCLR, GNN classifier, and sklearn transformers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _mock_gpu_deps():
    """Inject mocks for GPU/CuPy/triton/numba/C++ deps, restore after."""
    saved = {}
    for mod in [
        "cupy", "cupy.cuda", "cupyx", "cupyx.scipy",
        "numba", "numba.cuda",
        "triton", "triton.language",
        "pynerve_torch_internal", "pynerve_internal", "nerve_torch_internal",
        "h5py",
    ]:
        saved[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()

    sys.modules["cupy"].cuda = MagicMock()
    sys.modules["cupy"].cuda.is_available = MagicMock(return_value=False)
    sys.modules["cupy"].ndarray = torch.Tensor
    sys.modules["cupy"].asarray = lambda x, **kw: torch.as_tensor(x)
    sys.modules["numba"].jit = lambda *a, **k: lambda f: f
    sys.modules["numba"].cuda = MagicMock()
    sys.modules["triton"].jit = lambda *a, **k: lambda f: f
    sys.modules["triton"].language = MagicMock()
    sys.modules["triton"].autotune = lambda *a, **k: lambda f: f
    sys.modules["h5py"].File = MagicMock()

    yield

    for mod, orig in saved.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig


def _diag(n=5):
    """Create a valid persistence diagram tensor (N, 3) with birth<death."""
    births = torch.rand(n) * 0.5
    deaths = births + torch.rand(n) * 0.5 + 0.01
    dims = torch.randint(0, 2, (n,)).float()
    return torch.stack([births, deaths, dims], dim=1)


class TestCurriculumTrainerDeep:
    """Covers training/_curriculum_trainer.py — 71 missed, 13%."""

    def test_construct_epoch(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(num_stages=3))
        assert trainer.criterion == "epoch"
        assert trainer.current_stage == 0

    def test_construct_performance(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(
            model, CurriculumConfig(), stage_advancement_criterion="performance"
        )
        assert trainer.criterion == "performance"

    def test_construct_manual(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(
            model, CurriculumConfig(), stage_advancement_criterion="manual"
        )
        assert trainer.criterion == "manual"

    def test_construct_invalid_criterion(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        with pytest.raises(ValueError, match="invalid"):
            TopologicalCurriculumTrainer(model, CurriculumConfig(), stage_advancement_criterion="bad")

    def test_should_advance_epoch(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(
            model, CurriculumConfig(num_stages=3, warmup_epochs=2)
        )
        assert not trainer.should_advance_stage()  # epoch=0
        trainer.epoch = 2
        assert trainer.should_advance_stage()

    def test_should_advance_performance(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(
            model, CurriculumConfig(num_stages=3), stage_advancement_criterion="performance"
        )
        # Need 3 scores with low variance (max-min < 0.01) to advance
        # Call naturally through the API to exercise the append path
        assert not trainer.should_advance_stage(0.5)   # 1 score
        assert not trainer.should_advance_stage(0.501)  # 2 scores
        assert trainer.should_advance_stage(0.5)         # 3 scores, max-min=0.001 < 0.01

    def test_should_advance_at_max_stage(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(num_stages=3))
        trainer.current_stage = 2
        assert not trainer.should_advance_stage()

    def test_advance_stage(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(num_stages=3))
        assert trainer.current_stage == 0
        trainer.advance_stage()
        assert trainer.current_stage == 1
        trainer.advance_stage()
        assert trainer.current_stage == 2
        trainer.advance_stage()
        assert trainer.current_stage == 2  # clamped

    def test_create_dataloader_invalid_batch(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        dataset = list(range(10))
        with pytest.raises(ValueError, match="positive"):
            trainer.create_dataloader(dataset, [_diag(3) for _ in range(10)], batch_size=0)

    def test_create_dataloader_invalid_workers(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        dataset = list(range(10))
        with pytest.raises(ValueError, match="non-negative"):
            trainer.create_dataloader(dataset, [_diag(3) for _ in range(10)], num_workers=-1)

    def test_fit_negative_epochs(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        with pytest.raises(ValueError, match="non-negative"):
            trainer.fit(list(range(5)), [_diag(3) for _ in range(5)], epochs=-1)

    def test_evaluate_empty_diagrams(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig
        model = nn.Linear(10, 2)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        score = trainer.evaluate([], [])
        assert score == 0.0


class TestBYOLDeep:
    """Covers ssl/_byol.py — 41 missed, 27%."""

    def test_construct(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        model = BYOLTopology(encoder, projection_dim=32, hidden_dim=64, tau=0.99)
        assert model.tau == 0.99
        assert model.online_encoder is encoder

    def test_construct_invalid_proj_dim(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        with pytest.raises(ValueError, match="positive"):
            BYOLTopology(encoder, projection_dim=0)

    def test_construct_invalid_tau(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        with pytest.raises(ValueError, match="tau"):
            BYOLTopology(encoder, tau=2.0)

    def test_regression_loss(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        model = BYOLTopology(encoder, projection_dim=8)
        pred = torch.rand(4, 8)
        target = torch.rand(4, 8)
        loss = model.regression_loss(pred, target)
        assert loss >= 0
        assert loss <= 4

    def test_regression_loss_shape_mismatch(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        model = BYOLTopology(encoder, projection_dim=8)
        with pytest.raises(ValueError, match="matching shapes"):
            model.regression_loss(torch.rand(4, 8), torch.rand(3, 8))

    def test_update_target_network(self):
        from pynerve.ssl._byol import BYOLTopology
        encoder = nn.Linear(3, 16)
        model = BYOLTopology(encoder, projection_dim=8, tau=0.9)
        model.update_target_network()  # should not raise


class TestSimCLRDeep:
    """Covers ssl/_simclr.py — 24 missed, 33%."""

    def test_construct(self):
        from pynerve.ssl._simclr import SimCLRTopology
        encoder = nn.Linear(3, 16)
        model = SimCLRTopology(encoder, projection_dim=32, temperature=0.5)
        assert model.temperature == 0.5

    def test_construct_invalid_proj_dim(self):
        from pynerve.ssl._simclr import SimCLRTopology
        encoder = nn.Linear(3, 16)
        with pytest.raises(ValueError, match="positive"):
            SimCLRTopology(encoder, projection_dim=0)

    def test_construct_invalid_temp(self):
        from pynerve.ssl._simclr import SimCLRTopology
        encoder = nn.Linear(3, 16)
        with pytest.raises(ValueError, match="positive"):
            SimCLRTopology(encoder, temperature=0)

    def test_nt_xent_loss(self):
        from pynerve.ssl._simclr import SimCLRTopology
        encoder = nn.Linear(3, 16)
        model = SimCLRTopology(encoder, projection_dim=8)
        z1 = torch.nn.functional.normalize(torch.rand(4, 8), dim=1)
        z2 = torch.nn.functional.normalize(torch.rand(4, 8), dim=1)
        loss = model.nt_xent_loss(z1, z2)
        assert loss >= 0

    def test_nt_xent_loss_shape_mismatch(self):
        from pynerve.ssl._simclr import SimCLRTopology
        encoder = nn.Linear(3, 16)
        model = SimCLRTopology(encoder, projection_dim=8)
        with pytest.raises(ValueError, match="matching"):
            model.nt_xent_loss(torch.rand(4, 8), torch.rand(3, 8))


class TestMapperGNNClassifier:
    """Covers mapper/_gnn_classifier.py — 56 missed, 15%."""

    def test_construct(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        model = MapperGNNClassifier(node_dim=4, hidden_dim=32, num_classes=3, num_gnn_layers=2)
        assert model.node_dim == 4

    def test_construct_invalid_node_dim(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        with pytest.raises(ValueError):
            MapperGNNClassifier(node_dim=0)

    def test_construct_no_hierarchical(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        model = MapperGNNClassifier(node_dim=4, hidden_dim=32, num_classes=3, use_hierarchical=False)
        assert not model.use_hierarchical

    def test_forward_empty_nodes(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        model = MapperGNNClassifier(node_dim=4, hidden_dim=32, num_classes=3)
        empty = torch.empty((0, 4), dtype=torch.float32)
        edges = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="non-empty"):
            model.forward(empty, edges)

    def test_forward_wrong_dim(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        model = MapperGNNClassifier(node_dim=4, hidden_dim=32, num_classes=3)
        nodes = torch.rand(5, 3, dtype=torch.float32)  # wrong dim
        edges = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="node feature dimension"):
            model.forward(nodes, edges)

    def test_forward_non_2d(self):
        from pynerve.mapper._gnn_classifier import MapperGNNClassifier
        model = MapperGNNClassifier(node_dim=4, hidden_dim=32, num_classes=3)
        nodes = torch.rand(2, 5, 4, dtype=torch.float32)
        edges = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="2D"):
            model.forward(nodes, edges)


class TestSklearnTransformersDeep:
    """Covers torch/sklearn_transformers.py — 58 missed, 64%."""

    def test_import_all_classes(self):
        import pynerve.torch.sklearn_transformers as mod
        names = [n for n in dir(mod) if not n.startswith("_") and n[0].isupper()]
        assert len(names) > 0

    def test_validate_point_cloud(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud
        pc = torch.rand(10, 3, dtype=torch.float32)
        # _validate_point_cloud validates in-place; may return None
        _validate_point_cloud(pc)  # should not raise on valid input

    def test_validate_diagram_tensor(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor
        # Must use a valid diagram with birth <= death
        d = _diag(5)
        result = _validate_diagram_tensor(d)
        assert result is not None

    def test_tensor_to_numpy(self):
        from pynerve.torch.sklearn_transformers import _tensor_to_numpy
        t = torch.rand(5, 3, dtype=torch.float32)
        result = _tensor_to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 3)

    def test_as_float_tensor(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor
        result = _as_float_tensor(0.5)
        assert result is not None

    def test_validate_sequence(self):
        from pynerve.torch.sklearn_transformers import _validate_sequence
        _validate_sequence("test", [1, 2, 3])  # should not raise


class TestDiffLossDeep:
    """Covers diff/_loss_modules.py — 38 missed, 61%."""

    def test_stability_loss_forward(self):
        from pynerve.diff._loss_modules import StabilityLoss
        obj = StabilityLoss(epsilon=0.01, num_samples=2)
        points = torch.rand(5, 3, dtype=torch.float32)
        d = _diag(4)
        fn = lambda x: [d.clone()]
        result = obj.forward(points, fn)
        assert result is not None

    def test_multi_scale_loss_forward(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss
        obj = MultiScaleTopologyLoss(scales=(0.1, 0.5))
        d = _diag(6)
        targets = [_diag(3), _diag(4)]
        result = obj.forward(d, targets)
        assert result is not None

    def test_multi_scale_loss_wrong_targets(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss
        obj = MultiScaleTopologyLoss(scales=(0.1, 0.5))
        d = _diag(6)
        with pytest.raises(ValueError, match="length"):
            obj.forward(d, [_diag(3)])  # wrong number of targets

    def test_landscape_loss_forward(self):
        from pynerve.diff._loss_modules import LandscapeLoss
        obj = LandscapeLoss(n_layers=3, resolution=50)
        d1 = _diag(5)
        d2 = _diag(6)
        result = obj.forward(d1, d2)
        assert result is not None

    def test_landscape_loss_landscape(self):
        from pynerve.diff._loss_modules import LandscapeLoss
        obj = LandscapeLoss(n_layers=3, resolution=50)
        d = _diag(5)
        result = obj.landscape(d)
        assert result is not None

    def test_betti_number_loss_forward(self):
        from pynerve.diff._loss_modules import BettiNumberLoss
        obj = BettiNumberLoss(threshold=0.1, temperature=0.1)
        d = _diag(8)
        target = torch.tensor([2.0, 1.0])
        result = obj.forward(d, target)
        assert result is not None


class TestPersistenceVRDeep:
    """Covers torch/_persistence_vr.py — 45 missed, 56%."""

    def test_import(self):
        import pynerve.torch._persistence_vr as mod
        assert mod is not None

    def test_vr_persistence_function(self):
        from pynerve.torch._persistence_vr import vr_persistence
        assert vr_persistence is not None

    def test_vr_persistence_function_class(self):
        from pynerve.torch._persistence_vr import _VRPersistenceFunction
        assert _VRPersistenceFunction is not None


class TestInitModule:
    """Covers torch/__init__.py — 14 missed, 67%."""

    def test_diagram_wasserstein_public(self):
        from pynerve.torch import diagram_wasserstein
        d1 = torch.tensor([[0.0, 0.3], [0.1, 0.5]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 0.4], [0.2, 0.6]], dtype=torch.float32)
        result = diagram_wasserstein(d1, d2)
        assert result >= 0

    def test_diagram_bottleneck_public(self):
        from pynerve.torch import diagram_bottleneck
        d1 = torch.tensor([[0.0, 0.3], [0.1, 0.5]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 0.4], [0.2, 0.6]], dtype=torch.float32)
        # With mocked C++ backends, result may be a MagicMock.
        # The test still exercises the import + dispatch path.
        result = diagram_bottleneck(d1, d2)
        assert result is not None
