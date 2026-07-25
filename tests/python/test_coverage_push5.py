"""Targeted tests for nn/_ph_module, torch/data, fast_ops, and nn/building_blocks.

Exercises PersistentHomology module, PersistenceDataset, PointCloudDataset,
collate_diagrams, collate_point_clouds, create_dataloader, and fast_ops
NumPy functions (pairwise_distances, nearest_neighbors, vr_edges,
simplex_boundary, persistence_image, betti_curve, persistence_landscape,
connected_components, minimum_spanning_tree, enumerate_simplices, etc.).
"""

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


class TestPersistentHomologyModule:
    """Covers nn/_ph_module.py — PersistentHomology."""

    def test_construct(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology(max_dim=1, max_radius=5.0)
        assert ph.max_dim == 1
        assert ph.max_radius == 5.0

    def test_construct_defaults(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        assert ph.max_dim == 1
        assert ph.reduction == "clearing"
        assert ph.memory_mode == "standard"

    def test_construct_invalid_max_dim(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="non-negative"):
            PersistentHomology(max_dim=-1)

    def test_construct_invalid_max_radius(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="positive"):
            PersistentHomology(max_radius=0.0)

    def test_construct_invalid_metric(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="euclidean"):
            PersistentHomology(metric="manhattan")

    def test_construct_invalid_reduction(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="reduction"):
            PersistentHomology(reduction="bad")

    def test_construct_invalid_memory_mode(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="memory_mode"):
            PersistentHomology(memory_mode="bad")

    def test_construct_invalid_max_memory(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        with pytest.raises(Exception, match="positive"):
            PersistentHomology(memory_mode="extreme", max_memory_gb=0.0)

    def test_construct_cohomology(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology(reduction="cohomology")
        assert ph.reduction == "cohomology"

    def test_construct_streaming(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology(memory_mode="streaming")
        assert ph.memory_mode == "streaming"

    def test_extra_repr(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology(max_dim=2, max_radius=3.0)
        repr_str = ph.extra_repr()
        assert "max_dim=2" in repr_str
        assert "max_radius=3.000" in repr_str

    def test_to_cpu(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        ph.cpu()
        assert ph.device.type == "cpu"

    def test_float(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        ph.float()
        assert ph.dtype == torch.float32

    def test_double(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        ph.double()
        assert ph.dtype == torch.float64

    def test_half(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        ph.half()
        assert ph.dtype == torch.float16

    def test_train_eval(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        ph.train(True)
        assert ph.training is True
        ph.eval()
        assert ph.training is False

    def test_forward_not_3d(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        with pytest.raises(Exception, match="3D"):
            ph(torch.rand(5, 3))

    def test_forward_empty(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        with pytest.raises(Exception, match="non-empty"):
            ph(torch.empty(0, 5, 3))

    def test_forward_non_finite(self):
        from pynerve.nn.persistent_homology import PersistentHomology
        ph = PersistentHomology()
        points = torch.rand(1, 5, 3)
        points[0, 0, 0] = float("nan")
        with pytest.raises(Exception, match="finite"):
            ph(points)


class TestFastOps:
    """Covers fast_ops.py — NumPy-based topology operations."""

    def test_pairwise_distances(self):
        from pynerve.fast_ops import pairwise_distances
        points = np.random.rand(10, 3).astype(np.float64)
        result = pairwise_distances(points)
        assert result.shape == (10, 10)
        assert np.allclose(np.diag(result), 0.0)

    def test_pairwise_distances_broadcast(self):
        from pynerve.fast_ops import pairwise_distances_broadcast
        points = np.random.rand(8, 2).astype(np.float64)
        result = pairwise_distances_broadcast(points)
        assert result.shape == (8, 8)

    def test_nearest_neighbors(self):
        from pynerve.fast_ops import nearest_neighbors
        points = np.random.rand(10, 3).astype(np.float64)
        distances, indices = nearest_neighbors(points, k=3)
        assert distances.shape == (10, 3)
        assert indices.shape == (10, 3)

    def test_nearest_neighbors_invalid_k(self):
        from pynerve.fast_ops import nearest_neighbors
        points = np.random.rand(5, 3).astype(np.float64)
        with pytest.raises(ValueError, match="positive"):
            nearest_neighbors(points, k=0)

    def test_sparse_distance_matrix(self):
        from pynerve.fast_ops import sparse_distance_matrix
        points = np.random.rand(10, 3).astype(np.float64)
        result = sparse_distance_matrix(points, max_dist=0.5)
        assert result.shape == (10, 10)

    def test_sparse_distance_matrix_csr(self):
        from pynerve.fast_ops import sparse_distance_matrix
        points = np.random.rand(10, 3).astype(np.float64)
        result = sparse_distance_matrix(points, max_dist=0.5, output_type="csr")
        assert result.shape[0] == 10

    def test_sparse_distance_matrix_invalid_max_dist(self):
        from pynerve.fast_ops import sparse_distance_matrix
        points = np.random.rand(5, 3).astype(np.float64)
        with pytest.raises(ValueError, match="negative|finite"):
            sparse_distance_matrix(points, max_dist=-1.0)

    def test_vr_edges(self):
        from pynerve.fast_ops import vr_edges
        points = np.random.rand(10, 3).astype(np.float64)
        edges = vr_edges(points, max_dist=0.5)
        assert edges.shape[1] == 2

    def test_vr_edges_with_dists(self):
        from pynerve.fast_ops import vr_edges
        points = np.random.rand(10, 3).astype(np.float64)
        edges, dists = vr_edges(points, max_dist=0.5, return_dists=True)
        assert edges.shape[1] == 2
        assert dists.shape[0] == edges.shape[0]

    def test_vr_edges_invalid_max_dist(self):
        from pynerve.fast_ops import vr_edges
        points = np.random.rand(5, 3).astype(np.float64)
        with pytest.raises(ValueError, match="negative"):
            vr_edges(points, max_dist=-1.0)

    def test_simplex_boundary(self):
        from pynerve.fast_ops import simplex_boundary
        simplex = np.array([0, 1, 2])
        result = simplex_boundary(simplex)
        assert result.shape == (3, 2)

    def test_simplex_boundary_empty(self):
        from pynerve.fast_ops import simplex_boundary
        with pytest.raises(ValueError, match="empty"):
            simplex_boundary(np.array([], dtype=int))

    def test_simplex_boundary_duplicate(self):
        from pynerve.fast_ops import simplex_boundary
        with pytest.raises(ValueError, match="unique"):
            simplex_boundary(np.array([0, 1, 1]))

    def test_enumerate_simplices(self):
        from pynerve.fast_ops import enumerate_simplices
        points = np.random.rand(5, 3).astype(np.float64)
        result = enumerate_simplices(points, max_dist=0.5, max_dim=1)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_enumerate_simplices_invalid_max_dim(self):
        from pynerve.fast_ops import enumerate_simplices
        points = np.random.rand(5, 3).astype(np.float64)
        with pytest.raises(ValueError, match="negative"):
            enumerate_simplices(points, max_dist=0.5, max_dim=-1)

    def test_sort_filtration(self):
        from pynerve.fast_ops import sort_filtration
        simplices = np.array([[0, 1], [1, 2], [0, 2]])
        filt = np.array([0.3, 0.1, 0.2])
        idx, sorted_s, sorted_f = sort_filtration(simplices, filt)
        assert sorted_f[0] <= sorted_f[1] <= sorted_f[2]

    def test_vietoris_rips_filtration(self):
        from pynerve.fast_ops import vietoris_rips_filtration
        points = np.random.rand(6, 3).astype(np.float64)
        simplices, filt = vietoris_rips_filtration(points, max_dist=0.5, max_dim=1)
        assert isinstance(simplices, list)
        assert isinstance(filt, list)

    def test_persistence_image(self):
        from pynerve.fast_ops import persistence_image
        pairs = np.array([[0.0, 0.5], [0.1, 0.8], [0.2, 0.3]])
        result = persistence_image(pairs, resolution=32, sigma=0.1)
        assert result.shape == (32, 32)

    def test_persistence_image_linear_weight(self):
        from pynerve.fast_ops import persistence_image
        pairs = np.array([[0.0, 0.5], [0.1, 0.8]])
        result = persistence_image(pairs, resolution=16, sigma=0.1, weight_fn="linear")
        assert result.shape == (16, 16)

    def test_persistence_image_constant_weight(self):
        from pynerve.fast_ops import persistence_image
        pairs = np.array([[0.0, 0.5], [0.1, 0.8]])
        result = persistence_image(pairs, resolution=16, sigma=0.1, weight_fn="constant")
        assert result.shape == (16, 16)

    def test_persistence_image_invalid_resolution(self):
        from pynerve.fast_ops import persistence_image
        pairs = np.array([[0.0, 0.5]])
        with pytest.raises(ValueError, match="positive"):
            persistence_image(pairs, resolution=0)

    def test_persistence_image_invalid_sigma(self):
        from pynerve.fast_ops import persistence_image
        pairs = np.array([[0.0, 0.5]])
        with pytest.raises(ValueError, match="sigma|finite|positive"):
            persistence_image(pairs, sigma=-1.0)

    def test_betti_curve(self):
        from pynerve.fast_ops import betti_curve
        pairs = np.array([[0.0, 0.5, 0], [0.1, 0.8, 0], [0.2, 0.3, 1]])
        result = betti_curve(pairs, max_dim=1, resolution=50)
        assert result.shape == (2, 50)

    def test_betti_curve_max_time(self):
        from pynerve.fast_ops import betti_curve
        pairs = np.array([[0.0, 0.5, 0], [0.1, 0.8, 0]])
        result = betti_curve(pairs, max_dim=0, resolution=20, max_time=1.0)
        assert result.shape == (1, 20)

    def test_betti_curve_invalid_resolution(self):
        from pynerve.fast_ops import betti_curve
        pairs = np.array([[0.0, 0.5, 0]])
        with pytest.raises(ValueError, match="positive"):
            betti_curve(pairs, resolution=0)

    def test_persistence_landscape(self):
        from pynerve.fast_ops import persistence_landscape
        pairs = np.array([[0.0, 0.5], [0.1, 0.8], [0.2, 0.3]])
        result = persistence_landscape(pairs, n_layers=3, resolution=50)
        assert result.shape == (3, 50)

    def test_persistence_landscape_invalid_layers(self):
        from pynerve.fast_ops import persistence_landscape
        pairs = np.array([[0.0, 0.5]])
        with pytest.raises(ValueError, match="positive"):
            persistence_landscape(pairs, n_layers=0)

    def test_connected_components(self):
        from pynerve.fast_ops import connected_components
        edges = np.array([[0, 1], [1, 2], [3, 4]])
        labels = connected_components(edges, n_vertices=5)
        assert labels.shape == (5,)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]

    def test_connected_components_invalid_n(self):
        from pynerve.fast_ops import connected_components
        with pytest.raises(ValueError, match="negative"):
            connected_components(np.array([[0, 1]]), n_vertices=-1)

    def test_minimum_spanning_tree(self):
        from pynerve.fast_ops import minimum_spanning_tree
        points = np.random.rand(5, 3).astype(np.float64)
        mst = minimum_spanning_tree(points)
        assert mst.shape[1] == 2

    def test_minimum_spanning_tree_with_edges(self):
        from pynerve.fast_ops import minimum_spanning_tree
        points = np.random.rand(5, 3).astype(np.float64)
        edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]])
        mst = minimum_spanning_tree(points, edges=edges)
        assert mst.shape[1] == 2

    def test_boundary_matrix_sparse(self):
        from pynerve.fast_ops import boundary_matrix_sparse
        simplices_0 = np.array([[0], [1], [2]])
        simplices_1 = np.array([[0, 1], [1, 2], [0, 2]])
        result = boundary_matrix_sparse([simplices_0, simplices_1], max_dim=1)
        assert len(result) == 1
        assert result[0].shape[0] == 3  # 0-simplices
        assert result[0].shape[1] == 3  # 1-simplices


class TestTorchData:
    """Covers torch/data.py — datasets, collation, dataloader."""

    def test_collate_diagrams_empty(self):
        from pynerve.torch.data import collate_diagrams
        result = collate_diagrams([])
        assert result is not None

    def test_collate_point_clouds_empty(self):
        from pynerve.torch.data import collate_point_clouds
        result = collate_point_clouds([])
        assert result is not None

    def test_collate_point_clouds_basic(self):
        from pynerve.torch.data import collate_point_clouds
        pc1 = torch.rand(3, 2, dtype=torch.float32)
        pc2 = torch.rand(5, 2, dtype=torch.float32)
        result = collate_point_clouds([pc1, pc2])
        assert result.shape == (2, 5, 2)  # padded to max points

    def test_collate_point_clouds_with_labels(self):
        from pynerve.torch.data import collate_point_clouds
        pc1 = torch.rand(3, 2, dtype=torch.float32)
        pc2 = torch.rand(5, 2, dtype=torch.float32)
        result = collate_point_clouds([(pc1, torch.tensor(0)), (pc2, torch.tensor(1))])
        assert isinstance(result, tuple)
        assert result[0].shape == (2, 5, 2)
        assert result[1].shape == (2,)

    def test_collate_point_clouds_invalid_pad_value(self):
        from pynerve.torch.data import collate_point_clouds
        pc1 = torch.rand(3, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            collate_point_clouds([pc1], pad_value=float("inf"))

    def test_point_cloud_dataset(self):
        from pynerve.torch.data import PointCloudDataset
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(3)]
        ds = PointCloudDataset(pcs)
        assert len(ds) == 3
        assert ds[0].shape == (5, 3)

    def test_point_cloud_dataset_with_labels(self):
        from pynerve.torch.data import PointCloudDataset
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(3)]
        labels = torch.tensor([0, 1, 2])
        ds = PointCloudDataset(pcs, labels=labels)
        item = ds[0]
        assert isinstance(item, tuple)
        assert item[0].shape == (5, 3)
        assert item[1].item() == 0

    def test_point_cloud_dataset_label_mismatch(self):
        from pynerve.torch.data import PointCloudDataset
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(3)]
        labels = torch.tensor([0, 1])  # wrong length
        with pytest.raises(ValueError, match="length"):
            PointCloudDataset(pcs, labels=labels)

    def test_persistence_dataset_construct(self):
        from pynerve.torch.data import PersistenceDataset
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(3)]
        ds = PersistenceDataset(pcs, max_dim=1, cache=False)
        assert len(ds) == 3

    def test_persistence_dataset_label_mismatch(self):
        from pynerve.torch.data import PersistenceDataset
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(3)]
        labels = torch.tensor([0, 1])  # wrong length
        with pytest.raises(ValueError, match="length"):
            PersistenceDataset(pcs, labels=labels)

    def test_create_dataloader_point_cloud(self):
        from pynerve.torch.data import PointCloudDataset, create_dataloader
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(4)]
        ds = PointCloudDataset(pcs)
        dl = create_dataloader(ds, batch_size=2, shuffle=False)
        assert dl is not None

    def test_create_dataloader_invalid_batch(self):
        from pynerve.torch.data import PointCloudDataset, create_dataloader
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(4)]
        ds = PointCloudDataset(pcs)
        with pytest.raises(ValueError, match="positive"):
            create_dataloader(ds, batch_size=0)

    def test_create_dataloader_invalid_workers(self):
        from pynerve.torch.data import PointCloudDataset, create_dataloader
        pcs = [torch.rand(5, 3, dtype=torch.float32) for _ in range(4)]
        ds = PointCloudDataset(pcs)
        with pytest.raises(ValueError, match="non-negative"):
            create_dataloader(ds, num_workers=-1)

    def test_validate_single_point_cloud_not_tensor(self):
        from pynerve.torch.data import _validate_single_point_cloud
        with pytest.raises(TypeError, match="tensor"):
            _validate_single_point_cloud([1, 2, 3], dim=3, device=torch.device("cpu"), dtype=torch.float32)

    def test_validate_single_point_cloud_not_2d(self):
        from pynerve.torch.data import _validate_single_point_cloud
        with pytest.raises(ValueError, match="2D"):
            _validate_single_point_cloud(torch.rand(5), dim=3, device=torch.device("cpu"), dtype=torch.float32)

    def test_validate_single_point_cloud_empty(self):
        from pynerve.torch.data import _validate_single_point_cloud
        with pytest.raises(ValueError, match="at least one point"):
            _validate_single_point_cloud(torch.empty(0, 3), dim=3, device=torch.device("cpu"), dtype=torch.float32)

    def test_validate_single_point_cloud_not_floating(self):
        from pynerve.torch.data import _validate_single_point_cloud
        with pytest.raises(TypeError, match="floating"):
            _validate_single_point_cloud(torch.zeros(3, 3, dtype=torch.int32), dim=3, device=torch.device("cpu"), dtype=torch.float32)


class TestBuildingBlocks:
    """Covers nn/building_blocks.py — import and basic construction."""

    def test_import(self):
        import pynerve.nn.building_blocks as mod
        assert mod is not None

    def test_sparse_distance_matrix_class(self):
        from pynerve.nn._building_blocks_distance import SparseDistanceMatrix
        assert SparseDistanceMatrix is not None

    def test_persistence_sketch_class(self):
        from pynerve.nn._building_blocks_persistence import PersistenceSketch
        assert PersistenceSketch is not None

    def test_sparse_rips_persistence_class(self):
        from pynerve.nn._building_blocks_persistence import SparseRipsPersistence
        assert SparseRipsPersistence is not None

    def test_witness_complex_persistence_class(self):
        from pynerve.nn._building_blocks_persistence import WitnessComplexPersistence
        assert WitnessComplexPersistence is not None

    def test_persistence_diagram_class(self):
        from pynerve.nn._building_blocks_diagram import PersistenceDiagram
        assert PersistenceDiagram is not None
