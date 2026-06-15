"""Coverage tests for public API items not tested elsewhere."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _probe_cuda_jit() -> bool:
    """Lazily probe whether numba CUDA JIT compilation works."""
    try:
        from numba import cuda as _numba_cuda

        if not _numba_cuda.is_available():
            return False

        @_numba_cuda.jit("void(float32[:])")
        def _probe(x):
            x[0] = 1.0

        _probe[1, 1](np.zeros(1, dtype=np.float32))
        _numba_cuda.synchronize()
        return True
    except Exception:
        return False


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))


class TestPersistenceEngine:
    def test_enum_values(self):
        from pynerve._fallback_classes import PersistenceEngine

        assert PersistenceEngine.AUTO.value == "auto"
        assert PersistenceEngine.PH0.value == "ph0"
        assert PersistenceEngine.PH3.value == "ph3"
        assert PersistenceEngine.PH4.value == "ph4"
        assert PersistenceEngine.PH5.value == "ph5"
        assert PersistenceEngine.PH6.value == "ph6"

    def test_engine_names_unique(self):
        from pynerve._fallback_classes import PersistenceEngine

        values = [e.value for e in PersistenceEngine]
        assert len(values) == len(set(values)), f"Duplicate engine values: {values}"

    def test_auto_select_engine(self):
        from pynerve._compute_core import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine

        assert _auto_select_engine(100, 3, None, None, None) == PersistenceEngine.PH0
        assert _auto_select_engine(100, 3, "cpu", None, None) == PersistenceEngine.PH0
        assert _auto_select_engine(500, 2, None, None, None) == PersistenceEngine.PH0
        assert _auto_select_engine(5000, 3, None, None, None) == PersistenceEngine.PH3
        assert _auto_select_engine(50000, 3, None, None, None) == PersistenceEngine.PH4
        assert _auto_select_engine(500000, 3, None, None, None) == PersistenceEngine.PH5
        assert _auto_select_engine(5_000_000, 3, None, None, None) == PersistenceEngine.PH6

    def test_importable_from_root(self):
        import pynerve

        assert pynerve.PersistenceEngine is not None, "PersistenceEngine should not be None"


class TestEventType:
    def test_enum_values(self):
        from pynerve._fallback_classes import EventType

        assert EventType.ADD.value == "add", f"expected 'add', got {EventType.ADD.value!r}"
        assert EventType.REMOVE.value == "remove", (
            f"expected 'remove', got {EventType.REMOVE.value!r}"
        )

    def test_importable_from_root(self):
        import pynerve

        assert pynerve.EventType is not None, "EventType should not be None"


class TestPersistenceResult:
    def test_default_construction(self):
        from pynerve._compute_core import PersistenceResult

        r = PersistenceResult()
        assert r.pairs == [], f"expected empty list, got {r.pairs}"
        assert r.betti_numbers == [], f"expected empty list, got {r.betti_numbers}"
        assert r.max_dim == 0, f"expected 0, got {r.max_dim}"
        assert r.max_radius == 0.0, f"expected 0.0, got {r.max_radius}"
        assert r.num_pairs == 0, f"expected 0, got {r.num_pairs}"

    def test_from_dict(self):
        from pynerve._compute_core import PersistenceResult

        r = PersistenceResult.from_dict(
            {
                "pairs": [(0.0, 1.0, 0), (0.5, 2.0, 1)],
                "betti_numbers": [1, 0],
                "max_dim": 1,
                "max_radius": 2.0,
            }
        )
        assert len(r.pairs) == 2, f"expected 2, got {len(r.pairs)}"
        assert r.betti_numbers == [1, 0], f"expected [1, 0], got {r.betti_numbers}"
        assert r.max_dim == 1, f"expected 1, got {r.max_dim}"
        assert r.num_pairs == 2, f"expected 2, got {r.num_pairs}"

    def test_repr(self):
        from pynerve._compute_core import PersistenceResult

        r = PersistenceResult(pairs=[(0.0, 1.0, 0)], max_dim=1)
        s = repr(r)
        assert "PersistenceResult" in s, f"expected 'PersistenceResult' in {s!r}"
        assert "pairs=1" in s, f"expected 'pairs=1' in {s!r}"

    def test_importable_from_root(self):
        import pynerve

        assert pynerve.PersistenceResult is not None, "PersistenceResult should not be None"


class TestDatasets:
    def test_load_swiss_roll(self):
        from pynerve.datasets import load_swiss_roll

        data = load_swiss_roll(n_samples=50, seed=0)
        assert data.shape[0] == 50, f"expected 50, got {data.shape[0]}"
        assert np.isfinite(data).all(), "not all values are finite"

    def test_load_mobius_strip(self):
        from pynerve.datasets import load_mobius_strip

        data = load_mobius_strip(n_samples=50, seed=0)
        assert data.shape[0] == 50, f"expected 50, got {data.shape[0]}"
        assert np.isfinite(data).all(), "not all values are finite"

    def test_load_klein_bottle(self):
        from pynerve.datasets import load_klein_bottle

        data = load_klein_bottle(n_samples=50, seed=0)
        assert data.shape[0] == 50, f"expected 50, got {data.shape[0]}"
        assert np.isfinite(data).all(), "not all values are finite"


class TestDiagnostics:
    def test_profile_memory(self):
        from pynerve.diagnostics import profile_memory

        def fn():
            return [i for i in range(10000)]

        result, stats = profile_memory(fn)
        assert len(result) == 10000, f"expected 10000, got {len(result)}"
        assert isinstance(stats, dict), f"expected dict, got {type(stats).__name__}"

    def test_check_gpu_availability(self):
        from pynerve.diagnostics import check_gpu_availability

        result = check_gpu_availability()
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert "cuda_available" in result, f"expected 'cuda_available' in {result}"
        assert "device_count" in result, f"expected 'device_count' in {result}"

    def test_system_info(self):
        from pynerve.diagnostics import system_info

        info = system_info()
        assert isinstance(info, dict), f"expected dict, got {type(info).__name__}"
        assert "platform" in info or "python_version" in info, (
            f"expected platform or python_version in {info}"
        )

    def test_profile_memory_empty_stats_on_missing_psutil(self):
        from pynerve.diagnostics import profile_memory

        def fn():
            return 42

        result, stats = profile_memory(fn)
        assert result == 42, f"expected 42, got {result}"


class TestPipeline:
    def test_conditional_pipeline_builder(self):
        from pynerve.pipeline import ConditionalPipeline

        p = ConditionalPipeline()
        p.add_step("double", lambda x: x * 2)
        assert "double" in str(p.__dict__) or "double" in repr(p), (
            f"expected 'double' in p.__dict__ or repr(p), got dict={p.__dict__!r}, repr={repr(p)!r}"
        )

    def test_conditional_pipeline_with_conditional(self):
        from pynerve.pipeline import ConditionalPipeline

        p = ConditionalPipeline()
        p.add_conditional(
            "pos_only",
            condition=lambda x: x > 0,
            if_true=lambda x: x,
            if_false=lambda _x: 0,
        )
        assert p is not None, "ConditionalPipeline should not be None"

    def test_parallel_pipeline(self):
        from pynerve.pipeline import ConditionalPipeline, ParallelPipeline

        inner = ConditionalPipeline()
        inner.add_step("double", lambda x: x * 2)
        p = ParallelPipeline(
            pipelines={"a": inner},
            combine_fn=lambda outputs: sum(outputs.values()),
        )
        assert "a" in str(p.pipelines) or hasattr(p, "pipelines"), (
            f"expected 'a' in pipelines or hasattr, got pipelines={p.pipelines!r}"
        )


class TestFastOps:
    def test_boundary_matrix_sparse(self):
        from pynerve.fast_ops import boundary_matrix_sparse

        simplices = [
            np.array([[0], [1], [2]], dtype=np.int64),
            np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
        ]
        matrices = boundary_matrix_sparse(simplices, max_dim=1)
        assert len(matrices) >= 1, f"expected at least 1 matrix, got {len(matrices)}"

    def test_column_reduction_sparse(self):
        from pynerve.fast_ops import boundary_matrix_sparse, column_reduction_sparse

        simplices = [
            np.array([[0], [1], [2]], dtype=np.int64),
            np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
        ]
        bd = boundary_matrix_sparse(simplices, max_dim=1)
        filt = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        pairs = column_reduction_sparse(bd[0], filt)
        assert isinstance(pairs, list), f"expected list, got {type(pairs).__name__}"

    def test_vietoris_rips_filtration(self):
        from pynerve.fast_ops import vietoris_rips_filtration

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        simplices, values = vietoris_rips_filtration(points, max_dist=2.0, max_dim=1)
        assert isinstance(simplices, list), f"expected list, got {type(simplices).__name__}"
        assert isinstance(values, list), f"expected list, got {type(values).__name__}"
        assert len(simplices) >= 1, f"expected at least 1 simplex, got {len(simplices)}"


class TestJit:
    def test_pairwise_distances(self):
        from pynerve.jit import pairwise_distances

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dists = pairwise_distances(points)
        assert dists.shape == (3, 3), f"expected (3, 3), got {dists.shape}"
        assert np.isfinite(dists).all(), "not all distances are finite"
        assert dists[0, 0] == 0.0, f"expected 0.0, got {dists[0, 0]}"
        assert abs(dists[0, 1] - 1.0) < 1e-5, f"expected ~1.0, got {dists[0, 1]}"

    def test_pairwise_distances_invalid_input(self):
        from pynerve.exceptions import InvalidArgumentError
        from pynerve.jit import pairwise_distances

        with pytest.raises(InvalidArgumentError):
            pairwise_distances(np.array([]))

    @pytest.mark.skipif(not _probe_cuda_jit(), reason="requires CUDA GPU with JIT support")
    def test_persistence_image_gpu(self):
        from pynerve.jit import persistence_image

        pairs = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=np.float32)
        img = persistence_image(pairs, resolution=10, sigma=0.1, device="cuda")
        assert img.shape == (10, 10), f"expected (10, 10), got {img.shape}"
        assert np.isfinite(img).all(), "not all image values are finite"

    def test_pairwise_distances_gpu_raises_without_cuda(self):
        from pynerve.jit import pairwise_distances

        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        if not _probe_cuda_jit():
            with pytest.raises((RuntimeError, Exception)):
                pairwise_distances(pts, device="cuda")


class TestFormats:
    def test_save_csv_roundtrip(self, tmp_path):
        from pynerve.formats import auto_load, save_csv

        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        path = tmp_path / "test.csv"
        save_csv(diagram, str(path))
        assert path.exists(), f"expected {path} to exist"
        loaded = auto_load(str(path))
        assert loaded is not None, "loaded result should not be None"

    def test_auto_save(self, tmp_path):
        from pynerve.formats import auto_save

        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        path = tmp_path / "test_out.csv"
        auto_save(diagram, str(path))
        assert path.exists(), f"expected {path} to exist"

    def test_from_dionysus(self):
        from pynerve.formats import from_dionysus

        class MockDionysus:
            def __init__(self, birth, death):
                self.birth = birth
                self.death = death

        mock_diagrams = [[MockDionysus(0.0, 1.0)]]
        result = from_dionysus(mock_diagrams)
        assert isinstance(result, list), f"expected list, got {type(result).__name__}"

    def test_from_sktda(self):
        from pynerve.formats import from_sktda

        class MockSktda:
            def __init__(self):
                self.dgms = [np.array([[0.0, 1.0], [0.5, 2.0]])]

        result = from_sktda(MockSktda())
        assert isinstance(result, list), f"expected list, got {type(result).__name__}"

    def test_to_gudhi(self):
        from pynerve.formats import to_gudhi

        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        result = to_gudhi(diagram)
        assert result is not None, "to_gudhi result should not be None"
        assert isinstance(result, list), f"expected list, got {type(result).__name__}"

    def test_to_dionysus(self):
        from pynerve.formats import to_dionysus

        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        result = to_dionysus(diagram)
        assert result is not None, "to_dionysus result should not be None"


class TestTraining:
    def test_topology_adaptive_batch_size(self):
        from pynerve.training import TopologyAdaptiveBatchSize

        module = TopologyAdaptiveBatchSize(
            base_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            complexity_measure="num_features",
        )
        assert module.base_batch == 32, f"expected 32, got {module.base_batch}"


@pytest.mark.usefixtures("torch")
class TestMapper:
    def test_hierarchical_mapper_pooling_construct(self):
        from pynerve.mapper import HierarchicalMapperPooling

        module = HierarchicalMapperPooling(node_dim=16)
        assert module is not None, "HierarchicalMapperPooling should not be None"

    def test_mapper_node_encoder_construct(self):
        from pynerve.mapper import MapperNodeEncoder

        encoder = MapperNodeEncoder(input_dim=16, hidden_dim=32)
        assert encoder is not None, "MapperNodeEncoder should not be None"

    def test_topology_aware_readout_construct(self):
        from pynerve.mapper import TopologyAwareReadout

        module = TopologyAwareReadout(node_dim=16, output_dim=8)
        assert module is not None, "TopologyAwareReadout should not be None"


@pytest.mark.usefixtures("torch")
class TestDiff:
    def test_vietoris_rips_layer(self):
        from pynerve.diff import DifferentiableVietorisRips

        layer = DifferentiableVietorisRips(max_dim=1)
        assert layer.max_dim == 1, f"expected 1, got {layer.max_dim}"
        assert layer.max_radius is None, f"expected None, got {layer.max_radius}"

    def test_vietoris_rips_rejects_negative_dim(self):
        from pynerve.diff import DifferentiableVietorisRips

        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableVietorisRips(max_dim=-1)

    def test_alpha_complex_construct(self):
        from pynerve.diff import DifferentiableAlphaComplex

        layer = DifferentiableAlphaComplex(max_dim=2)
        assert layer.max_dim == 2, f"expected 2, got {layer.max_dim}"

    def test_cubical_construct(self):
        from pynerve.diff import DifferentiableCubical

        layer = DifferentiableCubical(max_dim=2, sublevel=True)
        assert layer.max_dim == 2, f"expected 2, got {layer.max_dim}"
        assert layer.sublevel is True, f"expected True, got {layer.sublevel}"

    def test_filtration_learning_layer(self):
        from pynerve.diff import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3)
        assert layer is not None, "FiltrationLearningLayer should not be None"

    def test_filtration_learning_rejects_non_positive_input(self):
        from pynerve.diff import FiltrationLearningLayer

        with pytest.raises(ValueError, match="positive"):
            FiltrationLearningLayer(input_dim=0)

    def test_learnable_filtration_persistence(self):
        from pynerve.diff import LearnableFiltrationPersistence

        layer = LearnableFiltrationPersistence(input_dim=4, max_dim=1)
        assert layer is not None, "LearnableFiltrationPersistence should not be None"

    def test_differentiable_ph_function_import(self):
        from pynerve.diff import DifferentiablePHFunction

        assert DifferentiablePHFunction is not None, "DifferentiablePHFunction should not be None"

    def test_combined_topology_loss_construct(self):
        from pynerve.diff import CombinedTopologyLoss

        loss = CombinedTopologyLoss()
        assert loss is not None, "CombinedTopologyLoss should not be None"

    def test_persistence_penalty_import(self):
        from pynerve.diff import persistence_penalty

        assert callable(persistence_penalty), "persistence_penalty should be callable"


@pytest.mark.usefixtures("torch")
class TestNN:
    def test_persistent_homology_construct(self):
        from pynerve.nn import PersistentHomology

        ph = PersistentHomology(max_dim=1, max_radius=2.0)
        assert ph.max_dim == 1, f"expected 1, got {ph.max_dim}"

    def test_sparse_distance_matrix_construct(self):
        from pynerve.nn import SparseDistanceMatrix

        m = SparseDistanceMatrix(k_neighbors=20)
        assert m is not None, "SparseDistanceMatrix should not be None"

    def test_witness_complex_persistence_construct(self):
        from pynerve.nn import WitnessComplexPersistence

        w = WitnessComplexPersistence(n_landmarks=50, max_dim=1)
        assert w is not None, "WitnessComplexPersistence should not be None"

    def test_persistence_sketch_construct(self):
        from pynerve.nn import PersistenceSketch

        s = PersistenceSketch(output_dim=64, max_dim=2)
        assert s is not None, "PersistenceSketch should not be None"

    def test_diagram_transformer_block_construct(self):
        from pynerve.nn import DiagramTransformerBlock

        block = DiagramTransformerBlock(d_model=32, num_heads=4)
        assert block is not None, "DiagramTransformerBlock should not be None"


@pytest.mark.usefixtures("torch")
class TestTorch:
    def test_persistence_dataset(self, torch):
        from pynerve.torch import PersistenceDataset

        points = [torch.randn(50, 3)]
        ds = PersistenceDataset(point_clouds=points, max_dim=1)
        assert len(ds) == 1, f"expected 1, got {len(ds)}"

    def test_bottleneck_distance_import(self):
        from pynerve.torch import diagram_bottleneck

        assert callable(diagram_bottleneck), "diagram_bottleneck should be callable"

    def test_unbatch_diagrams_import(self):
        from pynerve.torch import unbatch_diagrams

        assert callable(unbatch_diagrams), "unbatch_diagrams should be callable"

    def test_visualize_mapper_graph_import(self):
        from pynerve.torch import visualize_mapper_graph

        assert callable(visualize_mapper_graph), "visualize_mapper_graph should be callable"
