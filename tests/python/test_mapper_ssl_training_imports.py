"""Import-only tests for mapper, ssl, and training module class existence.

Exercises DifferentiableMapper, MapperAutoencoder, LensFunction, AdaptiveCover,
SoftClusterAssignment, MapperGraphEncoder, MapperGraphConv, TopologyAwareReadout,
MapperPoolingLayer, TopologyAugmentation, PersistencePredictionTask,
TopologyCompletionModel, BettiNumberPrediction, FiltrationOrderingTask,
TopologyDenoising, TopologyAdaptiveBatchSize, TopologyImportanceSampler,
BettiBalancedSampler, StabilityRegularizer, and MultiScaleTopologySampler.
"""

from __future__ import annotations


import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestMapperModules:
    """Covers mapper/learnable_mapper and related modules."""

    def test_import_learnable_mapper(self):
        from pynerve.mapper.learnable_mapper import DifferentiableMapper
        assert DifferentiableMapper is not None

    def test_import_mapper_autoencoder(self):
        from pynerve.mapper.learnable_mapper import MapperAutoencoder
        assert MapperAutoencoder is not None

    def test_import_lens_function(self):
        from pynerve.mapper._learnable_mapper_components import LensFunction
        assert LensFunction is not None

    def test_import_adaptive_cover(self):
        from pynerve.mapper._learnable_mapper_components import AdaptiveCover
        assert AdaptiveCover is not None

    def test_import_soft_cluster(self):
        from pynerve.mapper._learnable_mapper_components import SoftClusterAssignment
        assert SoftClusterAssignment is not None

    def test_import_mapper_graph_encoder(self):
        from pynerve.mapper._learnable_mapper_graph import MapperGraphEncoder
        assert MapperGraphEncoder is not None

    def test_import_gnn_conv(self):
        from pynerve.mapper._gnn_conv import MapperGraphConv
        assert MapperGraphConv is not None

    def test_import_gnn_readout(self):
        from pynerve.mapper._gnn_readout import TopologyAwareReadout
        assert TopologyAwareReadout is not None

    def test_import_gnn_pooling(self):
        from pynerve.mapper._gnn_pooling import MapperPoolingLayer
        assert MapperPoolingLayer is not None


class TestSSLModules:
    """Covers ssl modules — augmentation, persistence prediction, completion, etc."""

    def test_import_augmentation(self):
        from pynerve.ssl._augmentation import TopologyAugmentation
        assert TopologyAugmentation is not None

    def test_import_persistence_prediction(self):
        from pynerve.ssl._persistence_prediction import PersistencePredictionTask
        assert PersistencePredictionTask is not None

    def test_import_completion(self):
        from pynerve.ssl._completion import TopologyCompletionModel
        assert TopologyCompletionModel is not None

    def test_import_betti(self):
        from pynerve.ssl._betti import BettiNumberPrediction
        assert BettiNumberPrediction is not None

    def test_import_filtration(self):
        from pynerve.ssl._filtration import FiltrationOrderingTask
        assert FiltrationOrderingTask is not None

    def test_import_denoising(self):
        from pynerve.ssl._denoising import TopologyDenoising
        assert TopologyDenoising is not None


class TestTrainingModules:
    """Covers training/_adaptive, _importance, _betti, stability_reg, topology_sampler."""

    def test_import_adaptive_batch_size(self):
        from pynerve.training._adaptive import TopologyAdaptiveBatchSize
        assert TopologyAdaptiveBatchSize is not None

    def test_import_importance_sampler(self):
        from pynerve.training._importance import TopologyImportanceSampler
        assert TopologyImportanceSampler is not None

    def test_import_betti_balanced_sampler(self):
        from pynerve.training._betti import BettiBalancedSampler
        assert BettiBalancedSampler is not None

    def test_import_stability_reg(self):
        from pynerve.training.stability_reg import StabilityRegularizer
        assert StabilityRegularizer is not None

    def test_import_multiscale_sampler(self):
        from pynerve.training.topology_sampler import MultiScaleTopologySampler
        assert MultiScaleTopologySampler is not None
