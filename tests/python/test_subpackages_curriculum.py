"""Tests for pynerve curriculum subpackage."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _cuda_functional() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.current_device()
        return True
    except Exception:
        return False


# synthetic test helpers


def _make_diagram(
    n_pairs: int = 5,
    max_val: float = 1.0,
    dims: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Construct a valid diagram tensor with birth < death."""
    rng = torch.Generator(device=device).manual_seed(seed)
    births = torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.8
    deaths = births + torch.rand(n_pairs, generator=rng, device=device) * max_val * 0.4 + 0.01
    pairs = torch.stack([births, deaths], dim=1)
    if dims:
        dims_t = torch.randint(0, 3, (n_pairs, 1), generator=rng, device=device).float()
        return torch.cat([pairs, dims_t], dim=1)
    return pairs


def _make_points(
    batch: int = 2, n_points: int = 10, dim: int = 3, device: str = "cpu", seed: int = 42
) -> torch.Tensor:
    rng = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch, n_points, dim, generator=rng, device=device)


# training / curriculum


class TestComplexityMeasure:
    def test_enum_values(self):
        from pynerve.training.curriculum import ComplexityMeasure

        assert ComplexityMeasure.TOTAL_PERSISTENCE.value == "total_persistence"
        assert ComplexityMeasure.NUM_FEATURES.value == "num_features"


class TestCurriculumConfig:
    def test_defaults(self):
        from pynerve.training.curriculum import CurriculumConfig

        cfg = CurriculumConfig()
        assert cfg.num_stages == 5
        assert cfg.schedule == "linear"
        assert 0 < cfg.stage_ratio <= 1

    def test_validation(self):
        from pynerve.training.curriculum import CurriculumConfig

        with pytest.raises(ValueError):
            CurriculumConfig(num_stages=0)
        with pytest.raises(ValueError):
            CurriculumConfig(schedule="invalid")


class TestTopologicalComplexityCalculator:
    def test_total_persistence(self):
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.01)
        d = torch.tensor([[0.0, 0.5, 0.0], [0.1, 0.8, 1.0]])
        val = calc.compute_complexity(d, ComplexityMeasure.TOTAL_PERSISTENCE)
        assert val > 0

    def test_empty_diagram_zero(self):
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator()
        val = calc.compute_complexity(torch.empty((0, 3)), ComplexityMeasure.TOTAL_PERSISTENCE)
        assert val == 0.0

    def test_all_measures_finite(self):
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator()
        d = _make_diagram(10, dims=True)
        for measure in ComplexityMeasure:
            val = calc.compute_complexity(d, measure)
            assert np.isfinite(val)

    def test_batch_complexity(self):
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator()
        diagrams = [_make_diagram(n, dims=True, seed=i) for i, n in enumerate([3, 5, 7])]
        vals = calc.compute_batch_complexity(diagrams, ComplexityMeasure.NUM_FEATURES)
        assert len(vals) == 3
        assert all(v >= 0 for v in vals)


class TestTopologicalCurriculumSampler:
    def test_iter_yields_indices(self):
        from pynerve.training.curriculum import (
            CurriculumConfig,
            TopologicalCurriculumSampler,
        )

        dataset = torch.utils.data.TensorDataset(torch.randn(20, 5))
        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(20)]
        cfg = CurriculumConfig(num_stages=3, schedule="linear", stage_ratio=0.3)
        sampler = TopologicalCurriculumSampler(dataset, diagrams, cfg, current_stage=0, seed=42)
        indices = list(sampler)
        assert len(indices) > 0
        assert all(0 <= i < 20 for i in indices)

    def test_set_stage_increases_length(self):
        from pynerve.training.curriculum import (
            CurriculumConfig,
            TopologicalCurriculumSampler,
        )

        dataset = torch.utils.data.TensorDataset(torch.randn(30, 5))
        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(30)]
        cfg = CurriculumConfig(num_stages=3, schedule="linear")
        sampler = TopologicalCurriculumSampler(dataset, diagrams, cfg, current_stage=0, seed=42)
        len0 = len(list(sampler))
        sampler.set_stage(2)
        len2 = len(list(sampler))
        assert len2 >= len0


class TestTopologicalCurriculumTrainer:
    def test_create_dataloader(self):
        if not _cuda_functional():
            pytest.skip("CUDA not functional (compute capability mismatch)")

        from pynerve.training.curriculum import CurriculumConfig, TopologicalCurriculumTrainer

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(), seed=42)
        dataset = torch.utils.data.TensorDataset(torch.randn(20, 5), torch.randn(20))
        diagrams = [_make_diagram(np.random.randint(2, 6), dims=True) for _ in range(20)]
        loader = trainer.create_dataloader(dataset, diagrams, batch_size=4, num_workers=0)
        batches = list(loader)
        assert len(batches) > 0

    def test_advance_stage(self):
        from pynerve.training.curriculum import CurriculumConfig, TopologicalCurriculumTrainer

        model = torch.nn.Linear(5, 1)
        trainer = TopologicalCurriculumTrainer(model, CurriculumConfig(num_stages=3))
        assert trainer.current_stage == 0
        trainer.advance_stage()
        assert trainer.current_stage == 1
        trainer.advance_stage()
        trainer.advance_stage()  # clamped
        assert trainer.current_stage == 2

    def test_fit_short(self):
        if not _cuda_functional():
            pytest.skip("CUDA not functional (compute capability mismatch)")

        from pynerve.training.curriculum import CurriculumConfig, TopologicalCurriculumTrainer

        model = torch.nn.Linear(3, 1)
        cfg = CurriculumConfig(num_stages=2, warmup_epochs=1)
        trainer = TopologicalCurriculumTrainer(model, cfg, seed=42)
        train_ds = torch.utils.data.TensorDataset(torch.randn(30, 3), torch.randn(30, 1))
        train_diags = [_make_diagram(np.random.randint(2, 5), dims=True) for _ in range(30)]
        history = trainer.fit(
            train_ds,
            train_diags,
            epochs=2,
            batch_size=4,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=torch.nn.MSELoss(),
        )
        assert len(history) == 2


class TestBettiCurriculum:
    def test_get_active_dimensions_increases(self):
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=5)
        assert bc.get_active_dimensions(0) == [0]
        assert bc.get_active_dimensions(5) == [0, 1]
        assert bc.get_active_dimensions(10) == [0, 1, 2]
        assert bc.get_active_dimensions(20) == [0, 1, 2, 3]

    def test_filter_diagram_by_dim(self):
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=5)
        d = torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.1, 0.8, 1.0],
                [0.2, 0.6, 2.0],
                [0.3, 0.7, 3.0],
            ]
        )
        filtered = bc.filter_diagram_by_dim(d, 0)
        assert filtered.shape[0] == 1
        assert filtered[0, 2].item() == 0.0


class TestFiltrationCurriculum:
    def test_radius_increases_with_stage(self):
        from pynerve.training.curriculum import FiltrationCurriculum

        fc = FiltrationCurriculum(max_radius=2.0, num_stages=5)
        radii = [fc.get_stage_radius(s) for s in range(5)]
        assert all(r > 0 for r in radii)
        assert radii[0] <= radii[-1]

    def test_threshold_decreases_with_stage(self):
        from pynerve.training.curriculum import FiltrationCurriculum

        fc = FiltrationCurriculum(max_radius=2.0, num_stages=5)
        thresholds = [fc.get_stage_threshold(s) for s in range(5)]
        assert thresholds[-1] <= thresholds[0]
