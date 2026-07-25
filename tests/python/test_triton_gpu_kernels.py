"""GPU kernel tests for triton modules — requires CUDA + functional triton.

These tests exercise the @triton.jit kernel paths that are only accessible
when _use_triton() returns True (CUDA tensor + triton available + GPU).

If triton is installed but kernels fail to compile (e.g., incompatible GPU
architecture), the fixture skips all tests gracefully.

Run on HPC with:
    pytest test_triton_gpu_kernels.py -v --timeout=120
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def require_gpu_and_triton():
    """Skip if no CUDA GPU or triton not installed/functional."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    from pynerve.triton import _check_triton, _use_triton
    if not _check_triton():
        pytest.skip("triton not installed")
    # Verify triton is functional with a simple CUDA tensor probe
    cuda_t = torch.randn(3, 2, device="cuda", dtype=torch.float32)
    if not _use_triton(cuda_t):
        pytest.skip("triton installed but _use_triton() returned False")


class TestMapperGpuKernels:
    """Tests that exercise _density_kernel, _eccentricity_kernel,
    _kmeans_assign_kernel, _build_cover_kernel, and _nerve_edges_kernel."""

    @pytest.fixture(autouse=True)
    def _check(self, require_gpu_and_triton):
        pass

    def test_density_filter_gpu(self):
        from pynerve.triton._mapper import density_filter

        pts = torch.randn(100, 3, device="cuda", dtype=torch.float32)
        result = density_filter(pts, k_neighbors=10)
        assert result.device.type == "cuda"
        assert result.shape == (100,)
        assert (result >= 0).all()

    def test_eccentricity_filter_gpu(self):
        from pynerve.triton._mapper import eccentricity_filter

        pts = torch.randn(50, 2, device="cuda", dtype=torch.float32)
        result = eccentricity_filter(pts)
        assert result.device.type == "cuda"
        assert result.shape == (50,)
        assert (result >= 0).all()

    def test_kmeans_assign_gpu(self):
        from pynerve.triton._mapper import kmeans_assign

        pts = torch.randn(200, 3, device="cuda", dtype=torch.float32)
        centroids = torch.randn(5, 3, device="cuda", dtype=torch.float32)
        labels = kmeans_assign(pts, centroids)
        assert labels.device.type == "cuda"
        assert labels.dtype == torch.int32
        assert labels.shape == (200,)

    def test_kmeans_cluster_gpu(self):
        from pynerve.triton._mapper import kmeans_cluster

        pts = torch.cat([
            torch.randn(30, 2, device="cuda", dtype=torch.float32) + 0,
            torch.randn(30, 2, device="cuda", dtype=torch.float32) + 5,
            torch.randn(30, 2, device="cuda", dtype=torch.float32) + 10,
        ])
        labels = kmeans_cluster(pts, k=3, max_iter=10, seed=42)
        assert labels.shape == (90,)
        assert labels.device.type == "cuda"

    def test_build_cover_gpu(self):
        from pynerve.triton._mapper import build_cover

        filter_vals = torch.rand(500, 2, device="cuda", dtype=torch.float32)
        sizes, indices = build_cover(filter_vals, resolution=5, overlap=0.3)
        assert sizes.device.type == "cuda"
        assert indices.device.type == "cuda"
        assert (sizes >= 0).all()

    def test_compute_nerve_edges_gpu(self):
        from pynerve.triton._mapper import compute_nerve_edges

        # 10 nodes, some overlapping covers
        cover_sets = torch.randint(0, 5, (20,), device="cuda", dtype=torch.int32)
        starts = torch.arange(0, 20, 2, device="cuda", dtype=torch.int32)
        sizes = torch.full((10,), 2, device="cuda", dtype=torch.int32)
        edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=50)
        assert edges.device.type == "cuda"

    def test_roundtrip_consistency_gpu_cpu(self):
        """GPU results should match CPU fallback results numerically."""
        from pynerve.triton._mapper import kmeans_assign

        pts_gpu = torch.randn(50, 3, device="cuda", dtype=torch.float32)
        centroids_gpu = torch.randn(4, 3, device="cuda", dtype=torch.float32)
        labels_gpu = kmeans_assign(pts_gpu, centroids_gpu)

        pts_cpu = pts_gpu.cpu()
        centroids_cpu = centroids_gpu.cpu()
        labels_cpu = kmeans_assign(pts_cpu, centroids_cpu)

        assert (labels_gpu.cpu() == labels_cpu).all()


class TestNnOpsGpuKernels:
    """Tests that exercise _diagram_conv1d_kernel,
    _diagram_conv1d_relu_kernel, and _diagram_conv1d_sigmoid_kernel."""

    @pytest.fixture(autouse=True)
    def _check(self, require_gpu_and_triton):
        pass

    def test_diagram_conv1d_none_gpu(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 50, device="cuda", dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, device="cuda", dtype=torch.float32)
        bias = torch.zeros(4, device="cuda", dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="none")
        assert out.device.type == "cuda"
        assert out.shape == (2, 4, 48)

    def test_diagram_conv1d_relu_gpu(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 50, device="cuda", dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, device="cuda", dtype=torch.float32)
        bias = torch.zeros(4, device="cuda", dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="relu")
        assert out.device.type == "cuda"
        assert (out >= 0).all()

    def test_diagram_conv1d_sigmoid_gpu(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 50, device="cuda", dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, device="cuda", dtype=torch.float32)
        bias = torch.zeros(4, device="cuda", dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="sigmoid")
        assert out.device.type == "cuda"
        assert (out >= 0).all() and (out <= 1).all()

    def test_roundtrip_consistency_gpu_cpu(self):
        """GPU results should match CPU fallback for relu activation."""
        from pynerve.triton._nn_ops import diagram_conv1d

        torch.manual_seed(42)
        features_gpu = torch.randn(2, 3, 20, device="cuda", dtype=torch.float32)
        kernel_gpu = torch.randn(2, 3, 3, device="cuda", dtype=torch.float32)
        bias_gpu = torch.zeros(2, device="cuda", dtype=torch.float32)

        out_gpu = diagram_conv1d(features_gpu, kernel_gpu, bias_gpu, activation="relu")
        out_cpu = diagram_conv1d(
            features_gpu.cpu(), kernel_gpu.cpu(), bias_gpu.cpu(), activation="relu"
        )
        assert torch.allclose(out_gpu.cpu(), out_cpu, atol=1e-5)


class TestPersistenceGpuKernels:
    """Tests that exercise _pixel_kernel and _pair_kernel."""

    @pytest.fixture(autouse=True)
    def _check(self, require_gpu_and_triton):
        pass

    def test_persistence_image_pixel_strategy_gpu(self):
        """Many pairs triggers pixel strategy."""
        from pynerve.triton._persistence import persistence_image_from_diagram

        n = 200
        births = torch.rand(n, device="cuda", dtype=torch.float32) * 3.0
        deaths = births + torch.rand(n, device="cuda", dtype=torch.float32) * 2.0 + 0.1
        img = persistence_image_from_diagram(births, deaths, resolution=8, sigma=1.0)
        assert img.device.type == "cuda"
        assert img.shape == (8, 8)

    def test_persistence_image_pair_strategy_gpu(self):
        """Few pairs triggers pair strategy."""
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0, 1.0, 2.0], device="cuda", dtype=torch.float32)
        deaths = torch.tensor([3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=16, sigma=1.0)
        assert img.device.type == "cuda"
        assert img.shape == (16, 16)

    def test_roundtrip_consistency_gpu_cpu(self):
        """GPU and CPU produce comparable images."""
        from pynerve.triton._persistence import persistence_image_from_diagram

        births_cpu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        deaths_cpu = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)

        births_gpu = births_cpu.cuda()
        deaths_gpu = deaths_cpu.cuda()

        img_cpu = persistence_image_from_diagram(births_cpu, deaths_cpu, resolution=16, sigma=1.0)
        img_gpu = persistence_image_from_diagram(births_gpu, deaths_gpu, resolution=16, sigma=1.0)

        # Images should be similar (not identical due to float32 differences)
        diff = (img_gpu.cpu() - img_cpu).abs().max().item()
        assert diff < 1e-4, f"GPU/CPU image mismatch: max diff = {diff}"
