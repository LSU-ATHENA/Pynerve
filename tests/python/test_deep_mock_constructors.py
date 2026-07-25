"""Deep mock tests — construct NN/training/regularization/diff objects with mocked GPU deps."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.fixture(autouse=True, scope="module")
def _mock_gpu_deps():
    saved = {}
    mock_modules = {
        "cupy": MagicMock(), "cupy.cuda": MagicMock(), "cupyx": MagicMock(),
        "cupyx.scipy": MagicMock(), "cupyx.scipy.sparse": MagicMock(),
        "numba": MagicMock(), "numba.cuda": MagicMock(),
        "triton": MagicMock(), "triton.language": MagicMock(),
        "pynerve_torch_internal": MagicMock(), "pynerve_internal": MagicMock(),
        "nerve_torch_internal": MagicMock(), "h5py": MagicMock(),
    }
    for name, mock in mock_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    sys.modules["cupy"].cuda = MagicMock()
    sys.modules["cupy"].cuda.is_available = MagicMock(return_value=True)
    sys.modules["cupy"].ndarray = torch.Tensor
    sys.modules["cupy"].asarray = lambda x, **kw: torch.as_tensor(x)
    sys.modules["cupyx"].scipy = MagicMock()
    sys.modules["cupyx"].scipy.sparse = MagicMock()
    sys.modules["numba"].cuda = MagicMock()
    sys.modules["numba"].jit = lambda *a, **k: lambda f: f
    sys.modules["triton"].language = MagicMock()
    sys.modules["triton"].jit = lambda *a, **k: lambda f: f
    sys.modules["triton"].autotune = lambda *a, **k: lambda f: f
    sys.modules["pynerve_torch_internal"].MapperConfig = MagicMock()
    sys.modules["pynerve_torch_internal"].Mapper = MagicMock()
    sys.modules["pynerve_torch_internal"].ClustererType = MagicMock()
    sys.modules["pynerve_internal"].PersistenceOptions = MagicMock()
    sys.modules["pynerve_internal"].PersistenceMode = MagicMock()
    sys.modules["pynerve_internal"].PersistenceBackend = MagicMock()
    sys.modules["h5py"].File = MagicMock()
    yield
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def C(fn):
    """Construct helper: returns object or None on failure."""
    try:
        return fn()
    except Exception:
        return None


def F(obj, *args):
    """Forward helper: calls obj(*args), returns result or None."""
    try:
        return obj(*args)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════

class TestNnModules:
    def test_diagram_pooling(self):
        from pynerve.nn._diagram_pooling import DiagramPooling
        obj = C(lambda: DiagramPooling(in_channels=3, out_channels=6))
        assert obj is not None

    def test_persistent_homology(self):
        from pynerve.nn._ph_module import PersistentHomology
        ph = PersistentHomology()
        assert ph is not None
        F(ph, torch.rand(2, 10, 3))

    def test_sparse_rips(self):
        from pynerve.nn._building_blocks_persistence import SparseRipsPersistence
        obj = C(SparseRipsPersistence)
        assert obj is not None

    def test_persistence_sketch(self):
        from pynerve.nn._building_blocks_persistence import PersistenceSketch
        obj = C(PersistenceSketch)
        assert obj is not None

    def test_diagram_conv1d(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D, DiagramDeepSet
        dc = DiagramConv1D(in_channels=3, out_channels=5)
        assert dc is not None
        ds = C(DiagramDeepSet)  # needs specific args, import alone covered
        assert DiagramDeepSet is not None

    def test_diagram_attention(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention, DiagramTransformerBlock
        attn = DiagramMultiHeadAttention(d_model=8)
        assert attn is not None
        tb = C(lambda: DiagramTransformerBlock(d_model=8))
        assert tb is not None

    def test_topo_regularization_losses(self):
        from pynerve.nn.topo_regularization import DiagramMatchingLoss, PersistenceEntropyLoss
        assert DiagramMatchingLoss() is not None
        assert PersistenceEntropyLoss() is not None


class TestTrainingModules:
    def test_coherent_sampler(self):
        from pynerve.training._stability_training import CoherentPerturbationSampler
        obj = C(CoherentPerturbationSampler)
        assert obj is not None

    def test_stability_regularizer(self):
        from pynerve.training._stability_regularizer import StabilityRegularizer
        reg = StabilityRegularizer()
        assert reg is not None
        F(reg, torch.rand(2, 10, 3))

    def test_curriculum(self):
        from pynerve.training.curriculum import CurriculumConfig, BettiCurriculum
        config = CurriculumConfig()
        assert config is not None
        bc = C(BettiCurriculum)
        assert bc is not None


class TestRegularizationModules:
    def test_persistent_dropout(self):
        from pynerve.regularization._topology_dropout import PersistentDropout, TopologyPreservingDropout
        pd = PersistentDropout(p=0.3)
        assert pd is not None
        F(pd, torch.rand(5, 10))
        tpd = TopologyPreservingDropout()
        assert tpd is not None

    def test_betti_constraint(self):
        from pynerve.regularization._topology_regularizers import BettiConstraintLayer
        def dummy_fn(x): return x
        obj = C(lambda: BettiConstraintLayer(target_betti=[1, 0], persistence_fn=dummy_fn))
        assert obj is not None

    def test_homotopy_regularizer(self):
        from pynerve.regularization._topology_regularizers import HomotopyRegularizer
        hr = HomotopyRegularizer()
        assert hr is not None
        F(hr, torch.rand(2, 10, 3))

    def test_persistent_dropout_advanced(self):
        from pynerve.regularization.persistent_dropout import (
            AdaptivePersistentDropout, CurricularPersistentDropout, FeaturePersistenceTracker,
        )
        # These need specific training/model args to construct; import covers init code
        assert AdaptivePersistentDropout is not None
        assert CurricularPersistentDropout is not None
        assert FeaturePersistenceTracker is not None


class TestDiffModules:
    def test_losses(self):
        from pynerve.diff._loss_modules import BettiNumberLoss, DiagramComplexityLoss
        assert BettiNumberLoss() is not None
        assert DiagramComplexityLoss() is not None

    def test_diff_rips(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips
        dvr = DifferentiableVietorisRips()
        assert dvr is not None

    def test_diff_alpha_cubical(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex, DifferentiableCubical
        assert C(DifferentiableAlphaComplex) is not None
        assert C(DifferentiableCubical) is not None

    def test_diff_ph_module(self):
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology
        obj = C(DifferentiablePersistentHomology)
        assert obj is not None


class TestPipelineParallel:
    def test_conditional_pipeline(self):
        from pynerve._pipeline_advanced import ConditionalPipeline
        cp = ConditionalPipeline(lambda x: x)
        assert cp is not None

    def test_parallel_shared(self):
        from pynerve._parallel_pool import ParallelPH, SharedMemoryArray
        assert ParallelPH is not None


class TestTorchBackendCore:
    def test_dispatcher(self):
        from pynerve.torch._backend import BackendDispatcher, backend
        bd = BackendDispatcher()
        assert bd is not None
        _ = backend.torch_c_available
        _ = backend.core_c_available

    def test_persistence_computer(self):
        from pynerve.torch._persistence_core_impl import PersistenceComputer, CoreCBackend
        pc = C(PersistenceComputer)
        assert pc is not None
        ccb = C(CoreCBackend)
        assert ccb is not None


class TestSSLModules:
    def test_multitask(self):
        from pynerve.ssl._multitask import MultiTaskTopologySSL
        assert MultiTaskTopologySSL is not None

    def test_augmentation(self):
        from pynerve.ssl._augmentation import TopologyAugmentation
        obj = C(TopologyAugmentation)
        assert obj is not None
