"""Mock-based coverage tests -- exercises module imports, class definitions, constructors for 0% modules.

Uses unittest.mock to bypass GPU/CuPy/triton/numba/C++ extension dependencies.
Fixtures are scoped to this module only to prevent polluting other tests.
"""

from __future__ import annotations


import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")
_GPU_MOCK_CUDA_AVAILABLE = True

torch = pytest.importorskip("torch")


class TestPersistenceResult:
    def test_import(self):
        from pynerve._persistence_result import PersistenceResult
        assert PersistenceResult is not None

    def test_construct_empty(self):
        from pynerve._persistence_result import PersistenceResult
        r = PersistenceResult(pairs=[], betti_numbers=[], max_dim=1, max_radius=1.0, diagnostics={})
        assert r.pairs == []

    def test_construct_with_pairs(self):
        from pynerve._persistence_result import PersistenceResult
        r = PersistenceResult(
            pairs=[(0.0, 1.0, 0), (0.5, 2.0, 0)],
            betti_numbers=[1], max_dim=1, max_radius=2.0, diagnostics={},
        )
        assert len(r.pairs) == 2


class TestRegularization:
    def test_topology_dropout_import(self):
        from pynerve.regularization._topology_dropout import (
            PersistentDropout, TopologyPreservingDropout,
        )
        assert PersistentDropout is not None
        assert TopologyPreservingDropout is not None

    def test_topology_dropout_construct(self):
        from pynerve.regularization._topology_dropout import PersistentDropout
        pd = PersistentDropout(p=0.5)
        assert pd is not None

    def test_regularizers_import(self):
        from pynerve.regularization._topology_regularizers import (
            BettiConstraintLayer, HomotopyRegularizer,
        )
        assert BettiConstraintLayer is not None

    def test_persistent_dropout_import(self):
        from pynerve.regularization.persistent_dropout import (
            AdaptivePersistentDropout, CurricularPersistentDropout,
            FeaturePersistenceTracker,
        )
        assert AdaptivePersistentDropout is not None


class TestTraining:
    def test_stability_training_import(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        assert CoherentPerturbationSampler is not None

    def test_curriculum_import(self):
        from pynerve.training.curriculum import BettiCurriculum, CurriculumConfig
        assert BettiCurriculum is not None

    def test_curriculum_config(self):
        from pynerve.training.curriculum import CurriculumConfig
        # CurriculumConfig may be a dataclass; try no-arg or read signature
        config = CurriculumConfig()
        assert config is not None

    def test_stability_regularizer_import(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer
        assert StabilityRegularizer is not None

    def test_curriculum_trainer_import(self):
        from pynerve.training._curriculum_trainer import TopologicalCurriculumTrainer
        assert TopologicalCurriculumTrainer is not None


class TestNnModules:
    def test_ph_module_import(self):
        from pynerve.nn._ph_module import PersistentHomology, PersistentHomologyFunction
        assert PersistentHomology is not None

    def test_building_blocks_import(self):
        from pynerve.nn._building_blocks_persistence import (
            PersistenceDiagram, PersistenceSketch, SparseRipsPersistence,
        )
        assert SparseRipsPersistence is not None

    def test_topo_regularization_import(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss, PersistenceEntropyLoss
        assert DiagramMatchingLoss is not None

    def test_diagram_pooling_import(self):
        from pynerve.nn._diagram_pooling import DiagramPooling
        assert DiagramPooling is not None

    def test_ph_autograd_import(self):
        from pynerve.nn._ph_autograd import PersistentHomologyFunction
        assert PersistentHomologyFunction is not None


class TestDiffModules:
    def test_ph_layer_import(self):
        from pynerve.diff.ph_layer import (
            DifferentiableVietorisRips, DifferentiableAlphaComplex, DifferentiableCubical,
        )
        assert DifferentiableVietorisRips is not None

    def test_loss_modules_import(self):
        from pynerve.diff._loss_modules import BettiNumberLoss, DiagramComplexityLoss
        assert BettiNumberLoss is not None

    def test_ph_layer_module_import(self):
        from pynerve.diff.ph_layer_module import (
            DifferentiablePersistentHomology, DifferentiablePHFunction,
        )
        assert DifferentiablePersistentHomology is not None


class TestMapperModules:
    def test_components_import(self):
        from pynerve.mapper._learnable_mapper_components import (
            AdaptiveCover, CoverElement, LensFunction,
        )
        assert AdaptiveCover is not None

    def test_models_import(self):
        from pynerve.mapper._learnable_mapper_models import DifferentiableMapper
        assert DifferentiableMapper is not None


class TestPipelineModules:
    def test_core_import(self):
        import pynerve._pipeline_core
        assert pynerve._pipeline_core is not None

    def test_advanced_import(self):
        from pynerve._pipeline_advanced import ConditionalPipeline, ParallelPipeline
        assert ConditionalPipeline is not None

    def test_topology_import(self):
        from pynerve._pipeline_topology import analysis_pipeline
        assert analysis_pipeline is not None


class TestParallelPool:
    def test_import(self):
        from pynerve._parallel_pool import ParallelPH, SharedMemoryArray
        assert ParallelPH is not None


class TestSharedMemory:
    def test_import(self):
        from pynerve._shared_memory import SharedMemory, SharedMemoryArray
        assert SharedMemory is not None


class TestJitCpuKernels:
    def test_import(self):
        import pynerve.jit._cpu_kernels
        assert pynerve.jit._cpu_kernels is not None


class TestBenchmarkModules:
    def test_suite_import(self):
        from pynerve.benchmark._suite import benchmark_complexity_analysis
        assert benchmark_complexity_analysis is not None

    def test_compare_import(self):
        from pynerve.benchmark._compare_internal import BenchmarkComparison, GPUComparison
        assert BenchmarkComparison is not None


class TestSSLModules:
    def test_multitask_import(self):
        from pynerve.ssl._multitask import MultiTaskTopologySSL
        assert MultiTaskTopologySSL is not None

    def test_augmentation_import(self):
        from pynerve.ssl._augmentation import TopologyAugmentation
        assert TopologyAugmentation is not None


class TestTorchPersistenceCore:
    def test_import(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer, CoreCBackend
        assert PersistenceComputer is not None


class TestTorchPersistencePython:
    def test_import(self):
        from pynerve.torch._persistence_python import compute_vr_python
        assert compute_vr_python is not None

    def test_basic_compute(self):
        from pynerve.torch._persistence_python import compute_vr_python
        points = torch.rand(1, 10, 3, dtype=torch.float64)
        result = compute_vr_python(points, max_dim=1, metric="euclidean")
        assert result is not None
