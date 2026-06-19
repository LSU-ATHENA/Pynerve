"""Tests for pynerve training modules."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _make_diagram(
    n_pairs: int = 5,
    max_val: float = 1.0,
    dims: bool = True,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    rng = torch.Generator(device=device).manual_seed(seed)
    births = torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.8
    deaths = births + torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.4 + 0.01
    pairs = torch.stack([births, deaths], dim=1)
    if dims:
        dims_t = torch.randint(0, 3, (n_pairs, 1), generator=rng, device=device).float()
        return torch.cat([pairs, dims_t], dim=1)
    return pairs


class TestTopologyAdaptiveBatchSize:
    def test_init_defaults(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize()
        assert abs_.base_batch == 32
        assert abs_.min_batch == 8
        assert abs_.max_batch == 128
        assert abs_.complexity_measure == "num_features"

    def test_init_custom(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(
            base_batch_size=64,
            min_batch_size=16,
            max_batch_size=256,
            complexity_measure="total_persistence",
        )
        assert abs_.base_batch == 64
        assert abs_.min_batch == 16
        assert abs_.max_batch == 256
        assert abs_.complexity_measure == "total_persistence"

    def test_init_negative_sizes(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError, match="positive"):
            TopologyAdaptiveBatchSize(min_batch_size=0)
        with pytest.raises(ValueError, match="positive"):
            TopologyAdaptiveBatchSize(base_batch_size=-1)
        with pytest.raises(ValueError, match="positive"):
            TopologyAdaptiveBatchSize(max_batch_size=0)

    def test_init_min_exceeds_max(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError, match="min_batch_size cannot exceed"):
            TopologyAdaptiveBatchSize(min_batch_size=200, max_batch_size=100)

    def test_init_bad_measure(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        with pytest.raises(ValueError, match="unsupported complexity_measure"):
            TopologyAdaptiveBatchSize(complexity_measure="bogus")

    def test_init_base_clamped(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(base_batch_size=5, min_batch_size=10, max_batch_size=100)
        assert abs_.base_batch == 10
        abs_ = TopologyAdaptiveBatchSize(base_batch_size=200, min_batch_size=10, max_batch_size=100)
        assert abs_.base_batch == 100

    def test_compute_empty_diagrams(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(base_batch_size=32)
        assert abs_.compute_batch_size([]) == 32

    def test_compute_with_empty_diagram(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="num_features")
        empty = torch.empty((0, 3), dtype=torch.float64)
        result = abs_.compute_batch_size([empty])
        assert result == abs_.max_batch

    def test_compute_num_features(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="num_features")
        d = _make_diagram(3, dims=True)
        result = abs_.compute_batch_size([d])
        assert 8 <= result <= 128

    def test_compute_total_persistence(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="total_persistence")
        d = _make_diagram(5, dims=True)
        result = abs_.compute_batch_size([d])
        assert 8 <= result <= 128

    def test_compute_num_pairs(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="num_pairs")
        d = _make_diagram(10, dims=True)
        result = abs_.compute_batch_size([d])
        assert 8 <= result <= 128

    def test_compute_high_complexity_returns_min(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="num_pairs", min_batch_size=4)
        diagrams = [_make_diagram(100, dims=True) for _ in range(5)]
        result = abs_.compute_batch_size(diagrams)
        assert result == 4

    def test_compute_moderate_complexity(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(
            complexity_measure="num_pairs", base_batch_size=32, min_batch_size=4
        )
        diagrams = [_make_diagram(30, dims=True) for _ in range(3)]
        result = abs_.compute_batch_size(diagrams)
        assert result == 16

    def test_compute_low_complexity_returns_max(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize

        abs_ = TopologyAdaptiveBatchSize(complexity_measure="num_pairs", max_batch_size=200)
        empty = torch.empty((0, 3), dtype=torch.float64)
        result = abs_.compute_batch_size([empty, empty])
        assert result == 200


class TestBettiBalancedSampler:
    def test_init_defaults(self):
        from pynerve.training._betti import BettiBalancedSampler

        d = _make_diagram(5, dims=True)
        sampler = BettiBalancedSampler([d, d])
        assert sampler.max_betti == 10
        assert sampler.batch_size == 32
        assert sampler.balance_dimensions == [0, 1, 2]
        assert len(sampler) == 2

    def test_init_validation(self):
        from pynerve.training._betti import BettiBalancedSampler

        d = _make_diagram(5, dims=True)
        with pytest.raises(ValueError, match="max_betti"):
            BettiBalancedSampler([d], max_betti=-1)
        with pytest.raises(ValueError, match="batch_size"):
            BettiBalancedSampler([d], batch_size=0)
        with pytest.raises(ValueError, match="balance_dimensions"):
            BettiBalancedSampler([d], balance_dimensions=[-1])

    def test_len(self):
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(7)]
        sampler = BettiBalancedSampler(diagrams)
        assert len(sampler) == 7

    def test_iter_empty(self):
        from pynerve.training._betti import BettiBalancedSampler

        sampler = BettiBalancedSampler([])
        assert list(sampler) == []

    def test_iter_yields_all_indices(self):
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(20)]
        sampler = BettiBalancedSampler(diagrams, batch_size=4, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(20))

    def test_compute_betti_numbers_valid(self):
        from pynerve.training._betti import BettiBalancedSampler

        d = _make_diagram(5, dims=True)
        sampler = BettiBalancedSampler([d])
        bettis = sampler.betti_numbers
        assert len(bettis) == 1
        assert all(isinstance(v, int) for v in bettis[0].values())

    def test_compute_betti_numbers_with_empty(self):
        from pynerve.training._betti import BettiBalancedSampler

        empty = torch.empty((0, 3), dtype=torch.float64)
        d = _make_diagram(5, dims=True)
        sampler = BettiBalancedSampler([empty, d])
        bettis = sampler.betti_numbers
        assert bettis[0] == {0: 0, 1: 0, 2: 0}

    def test_betti_groups(self):
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(np.random.randint(1, 4), dims=True, seed=i) for i in range(10)]
        sampler = BettiBalancedSampler(diagrams, max_betti=5, batch_size=3, seed=42)
        groups = sampler.betti_groups
        all_indices = [idx for indices in groups.values() for idx in indices]
        assert sorted(all_indices) == list(range(10))

    def test_reproducibility(self):
        from pynerve.training._betti import BettiBalancedSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(15)]
        s1 = BettiBalancedSampler(diagrams, batch_size=4, seed=42)
        s2 = BettiBalancedSampler(diagrams, batch_size=4, seed=42)
        assert list(s1) == list(s2)

    def test_custom_balance_dimensions(self):
        from pynerve.training._betti import BettiBalancedSampler

        d = _make_diagram(5, dims=True)
        sampler = BettiBalancedSampler([d], balance_dimensions=[0, 1])
        assert sampler.balance_dimensions == [0, 1]
        assert len(sampler.betti_numbers[0]) == 2


class TestTopologicalCurriculumTrainer:
    def test_init_defaults(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        assert trainer.current_stage == 0
        assert trainer.epoch == 0
        assert trainer.criterion == "epoch"

    def test_init_validation(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        with pytest.raises(ValueError, match="stage_advancement_criterion"):
            TopologicalCurriculumTrainer(
                model, CurriculumConfig(), stage_advancement_criterion="bogus"
            )

    def test_advance_stage(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(num_stages=4))
        assert trainer.current_stage == 0
        trainer.advance_stage()
        assert trainer.current_stage == 1
        trainer.advance_stage()
        trainer.advance_stage()
        trainer.advance_stage()
        assert trainer.current_stage == 3

    def test_should_advance_epoch_criterion(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        cfg = CurriculumConfig(num_stages=5, warmup_epochs=2)
        trainer = TopologicalCurriculumTrainer(model, cfg, stage_advancement_criterion="epoch")
        assert not trainer.should_advance_stage()
        trainer.epoch = 1
        assert not trainer.should_advance_stage()
        trainer.epoch = 2
        assert trainer.should_advance_stage()

    def test_should_advance_performance_criterion(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            model, cfg, stage_advancement_criterion="performance"
        )
        assert not trainer.should_advance_stage(validation_score=0.5)
        assert not trainer.should_advance_stage(validation_score=0.505)
        assert trainer.should_advance_stage(validation_score=0.502)

    def test_should_advance_performance_stable(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            model, cfg, stage_advancement_criterion="performance"
        )
        trainer.validation_scores = [0.8, 0.805, 0.803]
        assert trainer.should_advance_stage()

    def test_should_advance_performance_unstable(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        cfg = CurriculumConfig(num_stages=5)
        trainer = TopologicalCurriculumTrainer(
            model, cfg, stage_advancement_criterion="performance"
        )
        trainer.validation_scores = [0.5, 0.8, 0.2]
        assert not trainer.should_advance_stage()

    def test_should_advance_manual(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(
            model, CurriculumConfig(num_stages=5), stage_advancement_criterion="manual"
        )
        trainer.epoch = 10
        assert not trainer.should_advance_stage()
        assert not trainer.should_advance_stage(validation_score=0.0)

    def test_should_advance_at_last_stage(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        cfg = CurriculumConfig(num_stages=3, warmup_epochs=1)
        trainer = TopologicalCurriculumTrainer(model, cfg, stage_advancement_criterion="epoch")
        trainer.current_stage = 2
        trainer.epoch = 5
        assert not trainer.should_advance_stage()

    def test_create_dataloader_validation(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(3, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        ds = torch.utils.data.TensorDataset(torch.randn(10, 3), torch.randn(10))
        diagrams = [_make_diagram(3, dims=True) for _ in range(10)]
        with pytest.raises(ValueError, match="batch_size"):
            trainer.create_dataloader(ds, diagrams, batch_size=0)
        with pytest.raises(ValueError, match="num_workers"):
            trainer.create_dataloader(ds, diagrams, batch_size=4, num_workers=-1)

    def test_create_dataloader(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(3, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(), seed=42)
        ds = torch.utils.data.TensorDataset(torch.randn(15, 3), torch.randn(15))
        diagrams = [_make_diagram(np.random.randint(2, 5), dims=True) for _ in range(15)]
        loader = trainer.create_dataloader(ds, diagrams, batch_size=4, num_workers=0)
        batches = list(loader)
        assert len(batches) > 0

    def test_evaluate(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig, ComplexityMeasure

        model = torch.nn.Linear(3, 1)
        cfg = CurriculumConfig(complexity_measure=ComplexityMeasure.NUM_FEATURES)
        trainer = TopologicalCurriculumTrainer(model, cfg)
        ds = torch.utils.data.TensorDataset(torch.randn(5, 3))
        diagrams = [_make_diagram(np.random.randint(2, 5), dims=True) for _ in range(5)]
        score = trainer.evaluate(ds, diagrams)
        assert score >= 0.0

    def test_evaluate_empty_diagrams(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(3, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig())
        ds = torch.utils.data.TensorDataset(torch.randn(3, 3))
        score = trainer.evaluate(ds, [])
        assert score == 0.0

    def test_train_epoch(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(4, 1)
        cfg = CurriculumConfig(num_stages=2)
        trainer = TopologicalCurriculumTrainer(model, cfg, seed=42)
        ds = torch.utils.data.TensorDataset(torch.randn(10, 4), torch.randn(10, 1))
        diagrams = [_make_diagram(np.random.randint(2, 5), dims=True) for _ in range(10)]
        loader = trainer.create_dataloader(ds, diagrams, batch_size=3, num_workers=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = trainer.train_epoch(loader, optimizer, torch.nn.MSELoss())
        assert np.isfinite(loss)

    def test_fit_validation(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(), seed=42)
        ds = torch.utils.data.TensorDataset(torch.randn(10, 5), torch.randn(10, 1))
        diagrams = [_make_diagram(3, dims=True) for _ in range(10)]
        with pytest.raises(ValueError, match="epochs"):
            trainer.fit(ds, diagrams, epochs=-1)

    def test_fit(self, monkeypatch):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        model = torch.nn.Linear(3, 1)
        cfg = CurriculumConfig(num_stages=2, warmup_epochs=1)
        trainer = TopologicalCurriculumTrainer(model, cfg, seed=42)
        train_ds = torch.utils.data.TensorDataset(torch.randn(10, 3), torch.randn(10, 1))
        train_diags = [_make_diagram(np.random.randint(2, 5), dims=True) for _ in range(10)]
        history = trainer.fit(
            train_ds,
            train_diags,
            epochs=2,
            batch_size=3,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=torch.nn.MSELoss(),
        )
        assert len(history) == 2
        for record in history:
            assert "epoch" in record
            assert "train_loss" in record
            assert "stage" in record
            assert np.isfinite(record["train_loss"])

    def test_fit_default_optimizer_loss(self, monkeypatch):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        from pynerve.training.curriculum import CurriculumConfig

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        model = torch.nn.Linear(2, 2)
        cfg = CurriculumConfig(num_stages=2, warmup_epochs=2)
        trainer = TopologicalCurriculumTrainer(model, cfg, seed=42)
        train_ds = torch.utils.data.TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10,)))
        train_diags = [_make_diagram(np.random.randint(2, 4), dims=True) for _ in range(10)]
        history = trainer.fit(train_ds, train_diags, epochs=1, batch_size=3)
        assert len(history) == 1


class TestTopologyImportanceSampler:
    def test_init_defaults(self):
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(5, dims=True)
        sampler = TopologyImportanceSampler([d, d])
        assert sampler.batch_size == 32
        assert sampler.history_length == 100
        assert sampler.novelty_threshold == 0.5
        assert len(sampler) == 2

    def test_init_validation_batch_size(self):
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(3, dims=True)
        with pytest.raises(ValueError, match="batch_size and history_length"):
            TopologyImportanceSampler([d], batch_size=0)
        with pytest.raises(ValueError, match="batch_size and history_length"):
            TopologyImportanceSampler([d], history_length=0)

    def test_init_validation_novelty_threshold(self):
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(3, dims=True)
        with pytest.raises(Exception, match="novelty_threshold"):
            TopologyImportanceSampler([d], novelty_threshold=0.0)
        with pytest.raises(Exception, match="novelty_threshold"):
            TopologyImportanceSampler([d], novelty_threshold=-1.0)

    def test_len(self):
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(11)]
        sampler = TopologyImportanceSampler(diagrams)
        assert len(sampler) == 11

    def test_iter_empty(self):
        from pynerve.training._importance import TopologyImportanceSampler

        sampler = TopologyImportanceSampler([])
        assert list(sampler) == []

    def test_iter_yields_all_indices(self):
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(15)]
        sampler = TopologyImportanceSampler(diagrams, batch_size=4, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(15))

    def test_embeddings_shape(self):
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True, seed=i) for i in range(5)]
        sampler = TopologyImportanceSampler(diagrams)
        assert sampler.embeddings.shape == (5, 8)

    def test_embeddings_empty_diagram(self):
        from pynerve.training._importance import TopologyImportanceSampler

        empty = torch.empty((0, 3), dtype=torch.float64)
        sampler = TopologyImportanceSampler([empty])
        assert sampler.embeddings.shape == (1, 8)
        assert np.allclose(sampler.embeddings[0], 0.0)

    def test_compute_novelty_no_seen(self):
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(5, dims=True)
        sampler = TopologyImportanceSampler([d, d])
        assert sampler._compute_novelty(0) == 1.0

    def test_compute_novelty_with_seen(self):
        from pynerve.training._importance import TopologyImportanceSampler

        d = _make_diagram(5, dims=True)
        sampler = TopologyImportanceSampler([d, d])
        sampler.seen_samples = [0]
        novelty = sampler._compute_novelty(1)
        assert 0.0 <= novelty <= 1.0

    def test_update_weights_empty(self):
        from pynerve.training._importance import TopologyImportanceSampler

        sampler = TopologyImportanceSampler([])
        sampler.update_weights([0])
        assert len(sampler.weights) == 0

    def test_update_weights_normalizes(self):
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 5), dims=True, seed=i) for i in range(5)]
        sampler = TopologyImportanceSampler(diagrams, history_length=10, seed=42)
        original_sum = sampler.weights.sum()
        assert abs(original_sum - 1.0) < 1e-9
        batch = list(sampler)[:3]
        sampler.update_weights(batch)
        assert abs(sampler.weights.sum() - 1.0) < 1e-9

    def test_reproducibility(self):
        from pynerve.training._importance import TopologyImportanceSampler

        diagrams = [_make_diagram(np.random.randint(2, 5), dims=True, seed=i) for i in range(10)]
        s1 = TopologyImportanceSampler(diagrams, batch_size=3, seed=42)
        s2 = TopologyImportanceSampler(diagrams, batch_size=3, seed=42)
        assert list(s1) == list(s2)


class _MockWriter:
    def __init__(self):
        self.images = []
        self.scalars = []

    def add_image(self, tag, img, step):
        self.images.append((tag, img, step))

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class TestDiagramVisualizationCallback:
    def test_init_defaults(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback()
        assert cb.log_every == 10
        assert cb.max_diagrams == 4
        assert cb.step == 0
        assert cb.writer is None

    def test_init_validation(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        with pytest.raises(Exception):
            DiagramVisualizationCallback(log_every=0)
        with pytest.raises(Exception):
            DiagramVisualizationCallback(max_diagrams=-1)
        with pytest.raises(Exception):
            DiagramVisualizationCallback(log_every=1.5)

    def test_on_batch_end_skips_when_not_interval(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        writer = _MockWriter()
        cb = DiagramVisualizationCallback(log_every=5, writer=writer)
        d = _make_diagram(3, dims=False)
        cb.on_batch_end(d, batch_idx=3)
        assert len(writer.images) == 0

    def test_on_batch_end_writer_none(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback(log_every=1)
        d = _make_diagram(3, dims=False)
        cb.on_batch_end(d, batch_idx=0)

    def test_on_batch_end_logs_at_interval(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        writer = _MockWriter()
        cb = DiagramVisualizationCallback(log_every=2, writer=writer)
        d = _make_diagram(3, dims=False)
        cb.on_batch_end(d, batch_idx=0)
        cb.on_batch_end(d, batch_idx=2)
        assert len(writer.images) >= 1
        for tag, img, step in writer.images:
            assert tag == "diagram"
            assert isinstance(img, torch.Tensor)
            assert img.dim() == 3

    def test_on_epoch_end_skips_when_not_interval(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        writer = _MockWriter()
        cb = DiagramVisualizationCallback(log_every=5, writer=writer)
        d = _make_diagram(3, dims=False)
        cb.on_epoch_end(3, d)
        assert len(writer.scalars) == 0

    def test_on_epoch_end_writer_none(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback(log_every=1)
        d = _make_diagram(3, dims=False)
        cb.on_epoch_end(0, d)

    def test_on_epoch_end_logs(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        writer = _MockWriter()
        cb = DiagramVisualizationCallback(log_every=1, writer=writer)
        d = _make_diagram(4, dims=False)
        cb.on_epoch_end(0, d)
        assert len(writer.scalars) == 2
        tags = {entry[0] for entry in writer.scalars}
        assert "diagram/total_persistence" in tags
        assert "diagram/num_features" in tags

    def test_on_batch_end_validates_batch_idx(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback(log_every=1)
        d = _make_diagram(3, dims=False)
        with pytest.raises(Exception):
            cb.on_batch_end(d, batch_idx=-1)

    def test_on_epoch_end_validates_epoch(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback(log_every=1)
        d = _make_diagram(3, dims=False)
        with pytest.raises(Exception):
            cb.on_epoch_end(-1, d)

    def test_on_epoch_end_with_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        writer = _MockWriter()
        cb = DiagramVisualizationCallback(log_every=1, writer=writer)
        raw = _make_diagram(4, dims=True)
        pd = PersistenceDiagram(raw)
        cb.on_epoch_end(0, pd)
        assert len(writer.scalars) == 2

    def test_diagram_to_image(self):
        from pynerve.torch._training_callbacks import DiagramVisualizationCallback

        cb = DiagramVisualizationCallback(log_every=1)
        d = _make_diagram(3, dims=False)
        img = cb._diagram_to_image(d)
        assert isinstance(img, torch.Tensor)
        assert img.dim() == 3


class TestTopologicalEarlyStopping:
    def test_init_defaults(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        es = TopologicalEarlyStopping()
        assert es.patience == 10
        assert es.target is None
        assert es.mode == "approach"
        assert es.min_delta == 0.01
        assert es.counter == 0
        assert es.history == []

    def test_init_validation(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        with pytest.raises(ValueError, match="patience"):
            TopologicalEarlyStopping(patience=0)
        with pytest.raises(ValueError, match="mode"):
            TopologicalEarlyStopping(mode="bogus")
        with pytest.raises(Exception):
            TopologicalEarlyStopping(min_delta=float("nan"))
        with pytest.raises(Exception):
            TopologicalEarlyStopping(min_delta=-1.0)
        with pytest.raises(Exception):
            TopologicalEarlyStopping(target_complexity=float("inf"))

    def test_approach_mode_stops(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(
            mode="approach", patience=3, target_complexity=3.0, min_delta=0.5
        )
        assert not es(d)
        assert not es(d)
        assert es(d)
        assert es.counter >= es.patience

    def test_stabilize_mode(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(mode="stabilize", patience=2, min_delta=10.0)
        assert not es(d)
        assert not es(d)
        assert es(d)

    def test_stabilize_mode_not_enough_history(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(mode="stabilize", patience=1, min_delta=1e-6)
        assert not es(d)

    def test_increase_mode(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(mode="increase", patience=2, min_delta=10.0)
        assert not es(d)
        assert not es(d)
        assert es(d)

    def test_decrease_mode(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(mode="decrease", patience=2, min_delta=10.0)
        assert not es(d)
        assert not es(d)
        assert es(d)

    def test_reset(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(
            mode="approach", patience=2, target_complexity=3.0, min_delta=0.5
        )
        es(d)
        assert len(es.history) == 1
        es.reset()
        assert es.history == []
        assert es.counter == 0

    def test_approach_mode_without_target(self):
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        d = _make_diagram(3, dims=False)
        es = TopologicalEarlyStopping(
            mode="approach", patience=2, target_complexity=None, min_delta=0.01
        )
        assert not es(d)
        assert not es(d)

    def test_with_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._training_callbacks import TopologicalEarlyStopping

        raw = _make_diagram(4, dims=True)
        pd = PersistenceDiagram(raw)
        es = TopologicalEarlyStopping(
            mode="approach", patience=1, target_complexity=100.0, min_delta=100.0
        )
        result = es(pd)
        assert isinstance(result, bool)


class TestDiagramMetric:
    def test_init_defaults(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric()
        assert dm.name == "diagram"
        assert dm.dim is None
        assert dm.track_stats == ["total", "mean", "max", "count", "entropy"]
        assert dm.values == {"total": [], "mean": [], "max": [], "count": [], "entropy": []}

    def test_init_validation_track_stats_type(self):
        from pynerve.torch._training_metrics import DiagramMetric

        with pytest.raises(TypeError, match="track_stats"):
            DiagramMetric(track_stats="total")
        with pytest.raises(TypeError, match="track_stats"):
            DiagramMetric(track_stats=42)

    def test_init_validation_unknown_stat(self):
        from pynerve.torch._training_metrics import DiagramMetric

        with pytest.raises(ValueError, match="unsupported track_stats"):
            DiagramMetric(track_stats=["bogus"])

    def test_init_validation_dim(self):
        from pynerve.torch._training_metrics import DiagramMetric

        with pytest.raises(Exception):
            DiagramMetric(dim=1.5)

    def test_update_and_compute(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric()
        d = _make_diagram(3, dims=False)
        dm.update(d)
        dm.update(d)
        result = dm.compute()
        assert "diagram_total_mean" in result
        assert "diagram_total_std" in result
        assert "diagram_mean_mean" in result
        assert "diagram_count_mean" in result
        assert "diagram_entropy_mean" in result
        for key in result:
            assert np.isfinite(result[key])

    def test_compute_empty_values(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric(track_stats=["total"])
        result = dm.compute()
        assert result == {}

    def test_reset(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric(track_stats=["total"])
        d = _make_diagram(3, dims=False)
        dm.update(d)
        assert len(dm.values["total"]) == 1
        dm.reset()
        assert dm.values["total"] == []

    def test_with_dim(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric(dim=0, track_stats=["total", "count"])
        d = _make_diagram(5, dims=True)
        dm.update(d)
        result = dm.compute()
        assert "diagram_total_mean" in result

    def test_with_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._training_metrics import DiagramMetric

        raw = _make_diagram(4, dims=True)
        pd = PersistenceDiagram(raw)
        dm = DiagramMetric(track_stats=["total", "count"])
        dm.update(pd)
        result = dm.compute()
        assert "diagram_total_mean" in result

    def test_selective_track_stats(self):
        from pynerve.torch._training_metrics import DiagramMetric

        dm = DiagramMetric(track_stats=["max", "mean"])
        d = _make_diagram(3, dims=False)
        dm.update(d)
        result = dm.compute()
        assert "diagram_max_mean" in result
        assert "diagram_mean_mean" in result
        assert "diagram_total_mean" not in result


class TestTopologicalComplexityMetric:
    def test_init_defaults(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        tcm = TopologicalComplexityMetric()
        assert tcm.target == 10.0
        assert tcm.history == []

    def test_init_validation_nan(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        with pytest.raises(ValueError, match="finite"):
            TopologicalComplexityMetric(target_complexity=float("nan"))

    def test_init_validation_negative(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        with pytest.raises(ValueError, match="non-negative"):
            TopologicalComplexityMetric(target_complexity=-1.0)

    def test_update_and_compute(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        tcm = TopologicalComplexityMetric(target_complexity=10.0)
        d = _make_diagram(4, dims=False)
        tcm.update(d)
        result = tcm.compute()
        assert "complexity" in result
        assert "target_distance" in result
        assert "mean_complexity" in result
        assert result["target_distance"] >= 0

    def test_compute_empty_history(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        tcm = TopologicalComplexityMetric(target_complexity=5.0)
        result = tcm.compute()
        assert result["complexity"] == 0.0
        assert result["target_distance"] == 5.0

    def test_reset(self):
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        tcm = TopologicalComplexityMetric()
        d = _make_diagram(3, dims=False)
        tcm.update(d)
        assert len(tcm.history) == 1
        tcm.reset()
        assert tcm.history == []

    def test_with_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._training_metrics import TopologicalComplexityMetric

        raw = _make_diagram(4, dims=True)
        pd = PersistenceDiagram(raw)
        tcm = TopologicalComplexityMetric(target_complexity=5.0)
        tcm.update(pd)
        result = tcm.compute()
        assert "complexity" in result
