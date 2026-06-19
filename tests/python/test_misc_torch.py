from __future__ import annotations

import builtins
import sys
from collections.abc import Callable, Generator
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_original_import = builtins.__import__


def _block_torch_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch'")
    return _original_import(name, *args, **kwargs)  # type: ignore[call-overload]


def _make_diagram(n_pairs: int = 3, seed: int = 42) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    births = torch.rand(n_pairs, generator=rng) * 0.8
    deaths = births + torch.rand(n_pairs, generator=rng) * 0.4 + 0.01
    dims = torch.randint(0, 3, (n_pairs, 1), generator=rng).float()
    return torch.cat([births.unsqueeze(1), deaths.unsqueeze(1), dims], dim=1)


class _DummyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, n: int = 16) -> None:
        self.data = torch.randn(n, 5)
        self.targets = torch.randint(0, 2, (n,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class _SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _subtract_is_positive(diagram: torch.Tensor) -> bool:
    return bool((diagram[:, 1] > diagram[:, 0]).all().item())


# ── mapper/__init__.py ────────────────────────────────────────────────


class TestMapperInit:
    def test_imports_with_torch(self) -> None:
        from pynerve.mapper import (
            AdaptiveCover,
            DifferentiableMapper,
            HierarchicalMapperPooling,
            LensFunction,
            MapperAutoencoder,
            MapperGNNClassifier,
            MapperGraphConv,
            MapperGraphEncoder,
            MapperNodeEncoder,
            SoftClusterAssignment,
            TopologyAwareReadout,
        )

        assert AdaptiveCover is not None
        assert LensFunction is not None

    def test_all_entries(self) -> None:
        from pynerve import mapper

        assert "AdaptiveCover" in mapper.__all__
        assert "LensFunction" in mapper.__all__

    def test_missing_torch_raises(self) -> None:
        for mod_key in list(sys.modules):
            if mod_key.startswith("pynerve.mapper"):
                del sys.modules[mod_key]
        with mock.patch.object(builtins, "__import__", side_effect=_block_torch_import):
            with pytest.raises(ImportError, match="torch"):
                __import__("pynerve.mapper")


# ── torch/_sklearn_compat.py ──────────────────────────────────────────


class TestSklearnCompat:
    def test_base_estimator_get_params(self) -> None:
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn is installed; fallback classes not used")

        from pynerve.torch._sklearn_compat import BaseEstimator

        est = BaseEstimator()
        assert est.get_params() == {}
        assert est.get_params(deep=False) == {}

    def test_base_estimator_set_params(self) -> None:
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn is installed; fallback classes not used")

        from pynerve.torch._sklearn_compat import BaseEstimator

        est = BaseEstimator()
        result = est.set_params(a=1, b=2)
        assert est.a == 1
        assert est.b == 2
        assert result is est

    def test_transformer_mixin_fit_transform(self) -> None:
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE

        if SKLEARN_AVAILABLE:
            pytest.skip("sklearn is installed; fallback classes not used")

        from pynerve.torch._sklearn_compat import TransformerMixin

        class _Minimal(TransformerMixin):
            def fit(self, X: object, y: object = None, **kwargs: object) -> _Minimal:
                self.fitted_ = True
                return self

            def transform(self, X: object) -> object:
                return [f"transformed_{len(X)}"]

        m = _Minimal()
        result = m.fit_transform([1, 2], y=[0, 0])
        assert result == ["transformed_2"]
        assert getattr(m, "fitted_", False)

    def test_require_non_empty_raises(self) -> None:
        from pynerve.torch._sklearn_compat import _require_non_empty

        with pytest.raises(ValueError, match="items"):
            _require_non_empty("items", [])

    def test_require_non_empty_passes(self) -> None:
        from pynerve.torch._sklearn_compat import _require_non_empty

        _require_non_empty("items", [1])

    def test_sklearn_available_flag(self) -> None:
        from pynerve.torch._sklearn_compat import SKLEARN_AVAILABLE

        assert isinstance(SKLEARN_AVAILABLE, bool)


# ── training/_adaptive.py ─────────────────────────────────────────────


class TestTopologyAdaptiveBatchSize:
    def test_init_valid(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(
            base_batch_size=64,
            min_batch_size=16,
            max_batch_size=256,
            complexity_measure="total_persistence",
        )
        assert calc.base_batch == 64
        assert calc.min_batch == 16
        assert calc.max_batch == 256

    def test_init_clamps_base_batch(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(base_batch_size=5, min_batch_size=10, max_batch_size=100)
        assert calc.base_batch == 10

        calc2 = TopologyAdaptiveBatchSize(
            base_batch_size=200, min_batch_size=10, max_batch_size=100
        )
        assert calc2.base_batch == 100

    @pytest.mark.parametrize("invalid", [0, -1, -5])
    def test_init_rejects_nonpositive_batch(self, invalid: int) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError):
            TopologyAdaptiveBatchSize(base_batch_size=invalid)

    def test_init_rejects_min_gt_max(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError):
            TopologyAdaptiveBatchSize(min_batch_size=64, max_batch_size=32)

    def test_init_rejects_bad_measure(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError):
            TopologyAdaptiveBatchSize(complexity_measure="bogus")

    def test_compute_empty_diagrams(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(base_batch_size=64)
        assert calc.compute_batch_size([]) == 64

    def test_compute_empty_diagram(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(complexity_measure="num_pairs")
        empty = torch.empty(0, 3, dtype=torch.float64)
        result = calc.compute_batch_size([empty])
        assert calc.min_batch <= result <= calc.max_batch

    def test_compute_num_features(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(complexity_measure="num_features")
        d = _make_diagram(20)
        result = calc.compute_batch_size([d])
        assert calc.min_batch <= result <= calc.max_batch

    def test_compute_total_persistence(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(complexity_measure="total_persistence")
        d = _make_diagram(10)
        result = calc.compute_batch_size([d])
        assert calc.min_batch <= result <= calc.max_batch

    def test_compute_high_complexity_returns_min(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(min_batch_size=8, complexity_measure="num_pairs")
        many_diagrams = [_make_diagram(200) for _ in range(10)]
        result = calc.compute_batch_size(many_diagrams)
        assert result == 8

    def test_compute_low_complexity_returns_max(self) -> None:
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        calc = TopologyAdaptiveBatchSize(max_batch_size=128, complexity_measure="num_features")
        threshold_diagrams = [
            torch.tensor([[0.0, 0.005, 0.0]], dtype=torch.float64) for _ in range(3)
        ]
        result = calc.compute_batch_size(threshold_diagrams)
        assert result == 128


# ── training/_betti.py ────────────────────────────────────────────────


class TestBettiBalancedSampler:
    def test_init_valid(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        d = _make_diagram(5)
        sampler = BettiBalancedSampler([d, d, d], max_betti=10, batch_size=2)
        assert sampler.batch_size == 2
        assert len(sampler) == 3

    def test_init_rejects_negative_max_betti(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        with pytest.raises(ValueError):
            BettiBalancedSampler([_make_diagram(3)], max_betti=-1)

    def test_init_rejects_nonpositive_batch_size(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        with pytest.raises(ValueError):
            BettiBalancedSampler([_make_diagram(3)], batch_size=0)

    def test_init_rejects_negative_dimensions(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        with pytest.raises(ValueError):
            BettiBalancedSampler([_make_diagram(3)], balance_dimensions=[-1])

    def test_init_default_dimensions(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        sampler = BettiBalancedSampler([_make_diagram(3)])
        assert sampler.balance_dimensions == [0, 1, 2]

    def test_compute_betti_empty_diagram(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        sampler = BettiBalancedSampler(
            [torch.empty(0, 3, dtype=torch.float64)], balance_dimensions=[0, 1]
        )
        assert sampler.betti_numbers[0] == {0: 0, 1: 0}

    def test_betti_groups_created(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        sampler = BettiBalancedSampler([_make_diagram(5), _make_diagram(5, seed=7)], max_betti=5)
        assert isinstance(sampler.betti_groups, dict)

    def test_iter_yields_all_indices(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(10)]
        sampler = BettiBalancedSampler(diagrams, batch_size=4, seed=123)
        yielded = list(sampler)
        assert sorted(yielded) == list(range(10))
        assert len(yielded) == 10

    def test_iter_reproducible(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(10)]
        s1 = list(BettiBalancedSampler(diagrams, seed=42))
        s2 = list(BettiBalancedSampler(diagrams, seed=42))
        assert s1 == s2

    def test_len(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(5) for _ in range(7)]
        sampler = BettiBalancedSampler(diagrams, batch_size=3)
        assert len(sampler) == 7

    def test_empty_diagrams(self) -> None:
        from pynerve.training._betti import BettiBalancedSampler

        sampler = BettiBalancedSampler([], batch_size=4)
        assert len(sampler) == 0
        assert list(sampler) == []


# ── training/_curriculum_trainer.py ───────────────────────────────────


class TestTopologicalCurriculumTrainer:
    def test_init_valid(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig()
        model = _SimpleModel()
        trainer = TopologicalCurriculumTrainer(model, cfg, stage_advancement_criterion="manual")
        assert trainer.current_stage == 0
        assert trainer.epoch == 0

    def test_init_rejects_invalid_criterion(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        with pytest.raises(ValueError):
            TopologicalCurriculumTrainer(
                _SimpleModel(), CurriculumConfig(), stage_advancement_criterion="bad"
            )

    def test_create_dataloader_rejects_nonpositive_batch(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        trainer = TopologicalCurriculumTrainer(_SimpleModel(), CurriculumConfig())
        with pytest.raises(ValueError):
            trainer.create_dataloader(_DummyDataset(), [_make_diagram(5)] * 16, batch_size=0)

    def test_create_dataloader_rejects_negative_workers(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        trainer = TopologicalCurriculumTrainer(_SimpleModel(), CurriculumConfig())
        with pytest.raises(ValueError):
            trainer.create_dataloader(_DummyDataset(), [_make_diagram(5)] * 16, num_workers=-1)

    def test_create_dataloader_returns_loader(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=3)
        trainer = TopologicalCurriculumTrainer(_SimpleModel(), cfg)
        loader = trainer.create_dataloader(
            _DummyDataset(20),
            [_make_diagram(5, seed=i) for i in range(20)],
            batch_size=4,
            num_workers=0,
        )
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_should_advance_stage_at_last(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=3)
        trainer = TopologicalCurriculumTrainer(
            _SimpleModel(), cfg, stage_advancement_criterion="manual"
        )
        trainer.current_stage = 2
        assert not trainer.should_advance_stage()

    def test_should_advance_epoch_criterion(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=5, warmup_epochs=2)
        trainer = TopologicalCurriculumTrainer(
            _SimpleModel(), cfg, stage_advancement_criterion="epoch"
        )
        assert not trainer.should_advance_stage()
        trainer.epoch = 2
        assert trainer.should_advance_stage()

    def test_should_advance_performance_no_scores(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            _SimpleModel(), cfg, stage_advancement_criterion="performance"
        )
        assert not trainer.should_advance_stage(0.5)
        assert trainer.validation_scores == [0.5]

    def test_should_advance_performance_stable(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            _SimpleModel(), cfg, stage_advancement_criterion="performance"
        )
        trainer.validation_scores = [0.505, 0.502, 0.500]
        assert trainer.should_advance_stage()

    def test_should_advance_performance_not_stable(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            _SimpleModel(), cfg, stage_advancement_criterion="performance"
        )
        trainer.validation_scores = [1.0, 0.5, 0.8]
        assert not trainer.should_advance_stage()

    def test_advance_stage(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=3)
        trainer = TopologicalCurriculumTrainer(_SimpleModel(), cfg)
        trainer.advance_stage()
        assert trainer.current_stage == 1
        trainer.advance_stage()
        trainer.advance_stage()
        trainer.advance_stage()
        assert trainer.current_stage == 2

    def test_train_epoch(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=3)
        model = _SimpleModel()
        trainer = TopologicalCurriculumTrainer(model, cfg)
        loader = trainer.create_dataloader(
            _DummyDataset(16),
            [_make_diagram(5, seed=i) for i in range(16)],
            batch_size=4,
            num_workers=0,
        )
        opt: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss()
        avg_loss = trainer.train_epoch(loader, opt, loss)
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0

    def test_fit_returns_history(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig(num_stages=3, warmup_epochs=1)
        model = _SimpleModel()
        trainer = TopologicalCurriculumTrainer(model, cfg, stage_advancement_criterion="epoch")
        ds = _DummyDataset(16)
        diagrams = [_make_diagram(5, seed=i) for i in range(16)]
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            history = trainer.fit(ds, diagrams, epochs=2, batch_size=4)
        assert len(history) == 2
        for record in history:
            assert "epoch" in record
            assert "train_loss" in record
            assert "stage" in record

    def test_fit_rejects_negative_epochs(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        with pytest.raises(ValueError):
            TopologicalCurriculumTrainer(_SimpleModel(), CurriculumConfig()).fit(
                _DummyDataset(), [], epochs=-1
            )

    def test_evaluate_returns_float(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig()
        trainer = TopologicalCurriculumTrainer(_SimpleModel(), cfg)
        diagrams = [_make_diagram(5, seed=i) for i in range(10)]
        score = trainer.evaluate(_DummyDataset(10), diagrams)
        assert isinstance(score, float)

    def test_evaluate_empty_diagrams(self) -> None:
        from pynerve.training.curriculum import CurriculumConfig
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer

        cfg = CurriculumConfig()
        trainer = TopologicalCurriculumTrainer(_SimpleModel(), cfg)
        score = trainer.evaluate(_DummyDataset(), [])
        assert score == 0.0


# ── training/_importance.py ───────────────────────────────────────────


class TestTopologyImportanceSampler:
    def test_init_valid(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(5)
        sampler = TopologyImportanceSampler(
            [d, d, d], batch_size=2, history_length=50, novelty_threshold=0.5
        )
        assert sampler.batch_size == 2
        assert len(sampler) == 3

    def test_init_rejects_nonpositive_batch(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        with pytest.raises(ValueError):
            TopologyImportanceSampler([_make_diagram(3)], batch_size=0)

    def test_init_rejects_nonpositive_history(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        with pytest.raises(ValueError):
            TopologyImportanceSampler([_make_diagram(3)], history_length=0)

    def test_init_rejects_nonpositive_threshold(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        with pytest.raises(ValueError):
            TopologyImportanceSampler([_make_diagram(3)], novelty_threshold=0.0)

    def test_init_rejects_nan_threshold(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        with pytest.raises(ValueError):
            TopologyImportanceSampler([_make_diagram(3)], novelty_threshold=float("nan"))

    def test_weights_sum_to_one(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(10)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=3)
        assert abs(sampler.weights.sum() - 1.0) < 1e-9

    def test_update_weights_empty(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        sampler = TopologyImportanceSampler([], batch_size=2)
        sampler.update_weights([])
        assert len(sampler.weights) == 0

    def test_update_weights_sums_to_one(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(5)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=2)
        sampler.update_weights([0, 1])
        assert abs(sampler.weights.sum() - 1.0) < 1e-9

    def test_iter_yields_all_indices(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(8)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=3, seed=42)
        yielded = list(sampler)
        assert sorted(yielded) == list(range(8))
        assert len(yielded) == 8

    def test_iter_reproducible(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(8)]
        s1 = list(TopologyImportanceSampler(diagrams, batch_size=3, seed=42))
        s2 = list(TopologyImportanceSampler(diagrams, batch_size=3, seed=42))
        assert s1 == s2

    def test_len(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5) for _ in range(9)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=4)
        assert len(sampler) == 9

    def test_embedding_shape(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(6)] * 3
        sampler = TopologyImportanceSampler(diagrams, batch_size=2)
        assert sampler.embeddings.shape == (3, 8)

    def test_empty_diagram_embedding(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        sampler = TopologyImportanceSampler([torch.empty(0, 3, dtype=torch.float64)], batch_size=1)
        assert sampler.embeddings.shape == (1, 8)
        assert (sampler.embeddings == 0).all()

    def test_novelty_no_history(self) -> None:
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(5, seed=i) for i in range(4)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=2)
        assert sampler._compute_novelty(0) == pytest.approx(1.0)


# ── regularization/__init__.py ────────────────────────────────────────


class TestRegularizationInit:
    def test_imports_with_torch(self) -> None:
        from pynerve.regularization import (
            AdaptivePersistentDropout,
            BettiConstraintLayer,
            CurricularPersistentDropout,
            HomotopyRegularizer,
            MorseRegularizer,
            PersistentBatchNorm,
            PersistentDropout,
            TopologicalSmoothness,
            TopologyPreservingDropout,
        )

        assert AdaptivePersistentDropout is not None
        assert TopologyPreservingDropout is not None

    def test_all_entries(self) -> None:
        from pynerve import regularization

        assert "AdaptivePersistentDropout" in regularization.__all__
        assert "BettiConstraintLayer" in regularization.__all__

    def test_missing_torch_raises(self) -> None:
        for mod_key in list(sys.modules):
            if mod_key.startswith("pynerve.regularization"):
                del sys.modules[mod_key]
        with mock.patch.object(builtins, "__import__", side_effect=_block_torch_import):
            with pytest.raises(ImportError, match="torch"):
                __import__("pynerve.regularization")


# ── _image_utils.py ───────────────────────────────────────────────────


class FakeDiagram:
    def __init__(self, data: np.ndarray) -> None:
        self.pairs = data
        self.pairs_array = data.copy()


class TestToDiagramArray:
    def test_from_ndarray(self) -> None:
        from pynerve._image_utils import _to_diagram_array

        d = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        result = _to_diagram_array(d)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_from_fake_diagram_pairs_array(self) -> None:
        from pynerve._image_utils import _to_diagram_array

        d = np.array([[0.0, 2.0, 1.0]], dtype=np.float64)
        fd = FakeDiagram(d)
        result = _to_diagram_array(fd)
        assert isinstance(result, np.ndarray)

    def test_from_fake_diagram_pairs(self) -> None:
        from pynerve._image_utils import _to_diagram_array

        class _PairsOnly:
            def __init__(self, data: np.ndarray) -> None:
                self.pairs = data

        d = np.array([[0.0, 1.5, 0.0]], dtype=np.float64)
        fd = _PairsOnly(d)
        result = _to_diagram_array(fd)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_from_list(self) -> None:
        from pynerve._image_utils import _to_diagram_array

        result = _to_diagram_array([[0.0, 1.0, 0.0], [0.5, 0.8, 1.0]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_empty_input(self) -> None:
        from pynerve._image_utils import _to_diagram_array

        result = _to_diagram_array(np.empty((0, 3), dtype=np.float64))
        assert result.shape == (0, 3)


class TestNormalizeImageResolution:
    def test_single_int(self) -> None:
        from pynerve._image_utils import _normalize_image_resolution

        assert _normalize_image_resolution(30) == (30, 30)

    def test_tuple(self) -> None:
        from pynerve._image_utils import _normalize_image_resolution

        assert _normalize_image_resolution((20, 40)) == (20, 40)

    def test_rejects_wrong_length_tuple(self) -> None:
        from pynerve._image_utils import _normalize_image_resolution

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            _normalize_image_resolution((1, 2, 3))

    def test_rejects_nonpositive(self) -> None:
        from pynerve._image_utils import _normalize_image_resolution

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            _normalize_image_resolution(0)

    def test_rejects_nonpositive_tuple(self) -> None:
        from pynerve._image_utils import _normalize_image_resolution

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            _normalize_image_resolution((0, 20))


class TestFiniteRange:
    def test_explicit(self) -> None:
        from pynerve._image_utils import _finite_range

        assert _finite_range(np.array([1.0, 2.0]), (0.0, 10.0)) == (0.0, 10.0)

    def test_from_values(self) -> None:
        from pynerve._image_utils import _finite_range

        assert _finite_range(np.array([1.0, 5.0]), None) == (1.0, 5.0)

    def test_empty_values_default(self) -> None:
        from pynerve._image_utils import _finite_range

        assert _finite_range(np.array([]), None) == (0.0, 1.0)

    def test_equal_bounds_bumped(self) -> None:
        from pynerve._image_utils import _finite_range

        low, high = _finite_range(np.array([3.0, 3.0]), None)
        assert high == low + 1.0

    def test_rejects_nan(self) -> None:
        from pynerve._image_utils import _finite_range

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            _finite_range(np.array([1.0]), (float("nan"), 2.0))


class TestPersistenceImage:
    def test_basic_image(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, 1.0, 0.0], [0.2, 0.8, 1.0]], dtype=np.float64)
        img = persistence_image(d, resolution=20, sigma=0.1)
        assert img.shape == (20, 20)
        assert img.dtype == np.float64

    def test_custom_resolution(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        img = persistence_image(d, resolution=(30, 40), sigma=0.1)
        assert img.shape == (30, 40)

    def test_empty_diagram(self) -> None:
        from pynerve._image_utils import persistence_image

        img = persistence_image(np.empty((0, 3), dtype=np.float64), resolution=10)
        assert img.shape == (10, 10)
        assert (img == 0).all()

    def test_all_infinite_deaths_produces_zeros(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, np.inf, 0.0]], dtype=np.float64)
        img = persistence_image(d, resolution=10)
        assert img.shape == (10, 10)
        assert (img == 0).all()

    def test_uniform_weight(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]], dtype=np.float64)
        img = persistence_image(d, resolution=20, sigma=0.1, weight="uniform")
        assert img.shape == (20, 20)

    def test_invalid_weight_raises(self) -> None:
        from pynerve._image_utils import persistence_image

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            persistence_image(np.empty((0, 3)), weight="bogus")

    def test_invalid_sigma_raises(self) -> None:
        from pynerve._image_utils import persistence_image

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            persistence_image(np.empty((0, 3)), sigma=-0.1)

    def test_nan_sigma_raises(self) -> None:
        from pynerve._image_utils import persistence_image

        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            persistence_image(np.empty((0, 3)), sigma=float("nan"))

    def test_explicit_ranges(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        img = persistence_image(
            d, resolution=20, sigma=0.1, birth_range=(0.0, 2.0), persistence_range=(0.0, 2.0)
        )
        assert img.shape == (20, 20)

    def test_fake_diagram_object(self) -> None:
        from pynerve._image_utils import persistence_image

        fd = FakeDiagram(np.array([[0.0, 1.0, 0.0], [0.2, 0.8, 1.0]], dtype=np.float64))
        img = persistence_image(fd, resolution=10, sigma=0.1)
        assert img.shape == (10, 10)

    def test_image_positive(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.0, 2.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float64)
        img = persistence_image(d, resolution=20, sigma=0.5)
        assert (img >= 0).all()

    def test_single_point(self) -> None:
        from pynerve._image_utils import persistence_image

        d = np.array([[0.5, 0.8, 0.0]], dtype=np.float64)
        img = persistence_image(d, resolution=10, sigma=0.2)
        assert img.shape == (10, 10)
