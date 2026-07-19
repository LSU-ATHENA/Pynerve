"""Functional tests for C++ backed memory, spectral, and sheaf submodules.

Covers nerve_extras.memory, nerve_extras.spectral, and nerve_extras.sheaf.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Add the build directory so nerve_extras can be imported
_build_path = Path(__file__).resolve().parent.parent.parent / "build" / "python"
if _build_path.is_dir() and str(_build_path) not in sys.path:
    sys.path.insert(0, str(_build_path))

try:
    import nerve_extras
except ImportError:
    pytest.skip("nerve_extras C++ module not available", allow_module_level=True)


# nerve_extras.memory


@pytest.mark.nerve_extras
class TestSizeClass:
    """Tests for the SizeClass enum."""

    @pytest.mark.parametrize("enum_name, expected_value", [
        ("TINY16", 16),
        ("TINY32", 32),
        ("TINY64", 64),
        ("TINY128", 128),
        ("TINY256", 256),
        ("SMALL512", 512),
        ("SMALL1024", 1024),
        ("SMALL2048", 2048),
        ("SMALL4096", 4096),
        ("MEDIUM8192", 8192),
        ("MEDIUM16384", 16384),
        ("MEDIUM32768", 32768),
        ("LARGE65536", 65536),
        ("LARGE131072", 131072),
        ("HUGE262144", 262144),
    ])
    def test_enum_values(self, enum_name: str, expected_value: int) -> None:
        cls = nerve_extras.memory.SizeClass
        val = getattr(cls, enum_name)
        assert int(val) == expected_value

    def test_enum_members(self) -> None:
        members = [x for x in dir(nerve_extras.memory.SizeClass) if not x.startswith("_")]
        assert "SMALL512" in members
        assert "HUGE262144" in members


@pytest.mark.nerve_extras
class TestGlobalPagePool:
    """Tests for GlobalPagePool singleton."""

    def test_singleton(self) -> None:
        p1 = nerve_extras.memory.GlobalPagePool.instance()
        p2 = nerve_extras.memory.GlobalPagePool.instance()
        assert p1 is p2

    def test_page_size(self) -> None:
        pool = nerve_extras.memory.GlobalPagePool.instance()
        assert pool.page_size > 0

    def test_pages_allocated(self) -> None:
        pool = nerve_extras.memory.GlobalPagePool.instance()
        # Pages are initially allocated by the pool itself
        assert pool.pages_allocated() >= 0

    def test_allocate_deallocate_page(self) -> None:
        pool = nerve_extras.memory.GlobalPagePool.instance()
        before = pool.pages_allocated()
        page = pool.allocate_page()
        assert pool.pages_allocated() == before + 1
        pool.deallocate_page(page)
        assert pool.pages_allocated() == before

    def test_hugetlb_pages(self) -> None:
        pool = nerve_extras.memory.GlobalPagePool.instance()
        # hugetlb may or may not be available on the system, but should not crash
        count = pool.hugetlb_pages_allocated()
        assert count >= 0
        assert isinstance(count, int)


@pytest.mark.nerve_extras
class TestRawArrayPool:
    """Tests for RawArrayPool."""

    def test_constructor_default(self) -> None:
        pool = nerve_extras.memory.RawArrayPool()
        assert pool.total_allocated == 0

    def test_constructor_custom_size(self) -> None:
        pool = nerve_extras.memory.RawArrayPool(1024 * 1024)
        assert pool.total_allocated == 0

    def test_allocate_and_deallocate(self) -> None:
        pool = nerve_extras.memory.RawArrayPool(4096)
        before = pool.total_allocated
        ptr = pool.allocate(128)
        assert pool.total_allocated == before + 128
        pool.deallocate(ptr, 128)
        assert pool.total_allocated == before

    def test_peak_utilization(self) -> None:
        pool = nerve_extras.memory.RawArrayPool(1024)
        pool.allocate(256)
        peak1 = pool.peak_utilization
        pool.allocate(128)
        peak2 = pool.peak_utilization
        assert peak2 >= peak1

    def test_reset(self) -> None:
        pool = nerve_extras.memory.RawArrayPool(1024)
        pool.allocate(256)
        assert pool.total_allocated > 0
        pool.reset()
        assert pool.total_allocated == 0

    def test_allocate_large_block(self) -> None:
        pool = nerve_extras.memory.RawArrayPool(1024 * 1024)
        ptr = pool.allocate(64 * 1024)
        assert ptr is not None
        pool.deallocate(ptr, 64 * 1024)


@pytest.mark.nerve_extras
class TestNumaAwareAllocator:
    """Tests for NumaAwareAllocator."""

    def test_constructor_default(self) -> None:
        alloc = nerve_extras.memory.NumaAwareAllocator()
        assert alloc.preferred_node == -1

    def test_constructor_with_node(self) -> None:
        alloc = nerve_extras.memory.NumaAwareAllocator(0)
        # Node 0 should always exist on any NUMA-capable system
        assert alloc.preferred_node == 0

    def test_set_preferred_node(self) -> None:
        alloc = nerve_extras.memory.NumaAwareAllocator()
        alloc.preferred_node = 0
        assert alloc.preferred_node == 0
        alloc.preferred_node = -1
        assert alloc.preferred_node == -1

    def test_allocate_deallocate(self) -> None:
        alloc = nerve_extras.memory.NumaAwareAllocator()
        ptr = alloc.allocate(256)
        assert ptr is not None
        alloc.deallocate(ptr, 256)

    def test_allocate_with_alignment(self) -> None:
        alloc = nerve_extras.memory.NumaAwareAllocator()
        ptr = alloc.allocate(256, 256)
        assert ptr is not None
        alloc.deallocate(ptr, 256)


@pytest.mark.nerve_extras
class TestSizeClassAllocator:
    """Tests for SizeClassAllocator."""

    def test_constructor(self) -> None:
        alloc = nerve_extras.memory.SizeClassAllocator()
        assert alloc is not None

    def test_allocate_deallocate_small(self) -> None:
        alloc = nerve_extras.memory.SizeClassAllocator()
        ptr = alloc.allocate(64)
        assert ptr is not None
        alloc.deallocate(ptr, 64)

    def test_allocate_deallocate_medium(self) -> None:
        alloc = nerve_extras.memory.SizeClassAllocator()
        ptr = alloc.allocate(4096)
        assert ptr is not None
        alloc.deallocate(ptr, 4096)

    def test_get_size_class(self) -> None:
        alloc = nerve_extras.memory.SizeClassAllocator()
        cls = alloc.get_size_class(100)  # 100 bytes -> rounded up to 128
        assert cls == 128

    def test_total_allocated(self) -> None:
        alloc = nerve_extras.memory.SizeClassAllocator()
        before = alloc.total_allocated()
        ptr = alloc.allocate(128)
        assert alloc.total_allocated() >= before + 128
        alloc.deallocate(ptr, 128)


@pytest.mark.nerve_extras
class TestMemoryGlobalTracking:
    """Tests for memory global tracking free functions."""

    def test_get_global_allocation_count(self) -> None:
        count = nerve_extras.memory.get_global_allocation_count()
        assert count >= 0

    def test_get_global_deallocation_count(self) -> None:
        count = nerve_extras.memory.get_global_deallocation_count()
        assert count >= 0

    def test_get_global_current_bytes(self) -> None:
        bytes_ = nerve_extras.memory.get_global_current_bytes()
        assert bytes_ >= 0

    def test_get_global_peak_bytes(self) -> None:
        peak = nerve_extras.memory.get_global_peak_bytes()
        assert peak >= 0

    def test_track_alloc_event(self) -> None:
        nerve_extras.memory.reset_global_memory_stats()
        before = nerve_extras.memory.get_global_allocation_count()
        nerve_extras.memory.track_alloc_event(1024)
        after = nerve_extras.memory.get_global_allocation_count()
        assert after == before + 1

    def test_track_free_event(self) -> None:
        nerve_extras.memory.reset_global_memory_stats()
        before = nerve_extras.memory.get_global_deallocation_count()
        nerve_extras.memory.track_free_event(512)
        after = nerve_extras.memory.get_global_deallocation_count()
        assert after == before + 1

    def test_reset_global_memory_stats(self) -> None:
        nerve_extras.memory.track_alloc_event(1024)
        nerve_extras.memory.reset_global_memory_stats()
        assert nerve_extras.memory.get_global_allocation_count() == 0
        assert nerve_extras.memory.get_global_current_bytes() == 0

    def test_get_slab_allocator_diagnostic_count(self) -> None:
        count = nerve_extras.memory.get_slab_allocator_diagnostic_count()
        assert count >= 0

    def test_estimate_memory_overhead(self) -> None:
        overhead = nerve_extras.memory.estimate_memory_overhead(100, 64, 256)
        assert overhead > 0
        # 100 objects of 64 bytes with 256 slab capacity = 1 slab
        # overhead = 1 * 256 * (64 + sizeof(void*))
        assert overhead > 64 * 100  # should be more than raw data


# nerve_extras.spectral


@pytest.mark.nerve_extras
class TestLaplacianConfig:
    """Tests for LaplacianConfig."""

    def test_defaults(self) -> None:
        cfg = nerve_extras.spectral.LaplacianConfig()
        assert cfg.enable_gpu is False
        assert cfg.threshold >= 0
        assert hasattr(cfg, "prefer_tiled_kernels")
        assert hasattr(cfg, "max_gpu_memory_mb")

    def test_field_assignment(self) -> None:
        cfg = nerve_extras.spectral.LaplacianConfig()
        cfg.enable_gpu = True
        cfg.threshold = 500
        assert cfg.enable_gpu is True
        assert cfg.threshold == 500

    @pytest.mark.parametrize("field", ["enable_gpu", "threshold", "prefer_tiled_kernels", "max_gpu_memory_mb"])
    def test_has_fields(self, field: str) -> None:
        cfg = nerve_extras.spectral.LaplacianConfig()
        assert hasattr(cfg, field)


@pytest.mark.nerve_extras
class TestSpectralConfig:
    """Tests for SpectralConfig."""

    def test_defaults(self) -> None:
        cfg = nerve_extras.spectral.SpectralConfig()
        assert cfg.num_eigenpairs > 0
        assert cfg.convergence_tolerance > 0
        assert cfg.max_iterations > 0
        assert hasattr(cfg, "spectral_shift")

    def test_field_assignment(self) -> None:
        cfg = nerve_extras.spectral.SpectralConfig()
        cfg.num_eigenpairs = 10
        cfg.convergence_tolerance = 1e-10
        assert cfg.num_eigenpairs == 10
        assert cfg.convergence_tolerance == 1e-10

    @pytest.mark.parametrize("field", ["num_eigenpairs", "convergence_tolerance", "max_iterations", "spectral_shift"])
    def test_has_fields(self, field: str) -> None:
        cfg = nerve_extras.spectral.SpectralConfig()
        assert hasattr(cfg, field)


@pytest.mark.nerve_extras
class TestLaplacian:
    """Tests for Laplacian (empty/default constructed)."""

    def test_constructor(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        assert lap is not None

    def test_size(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        assert lap.size() == 0

    def test_max_dimension(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        assert lap.max_dimension() >= 0

    def test_get_laplacian_empty_dim_out_of_range(self) -> None:
        """On an empty Laplacian, requesting a dimension raises IndexError."""
        lap = nerve_extras.spectral.Laplacian()
        with pytest.raises((IndexError, RuntimeError)):
            lap.get_laplacian(0)

    def test_spectrum_empty_dim_out_of_range(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        with pytest.raises((IndexError, RuntimeError)):
            lap.spectrum(0)

    def test_compute_spectral_gap_empty_dim_out_of_range(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        with pytest.raises((IndexError, RuntimeError)):
            lap.compute_spectral_gap(0)

    def test_compute_cheeger_constants_empty(self) -> None:
        """compute_cheeger_constants() takes no dimension arg, may return empty list."""
        lap = nerve_extras.spectral.Laplacian()
        try:
            constants = lap.compute_cheeger_constants()
            assert isinstance(constants, list)
        except (IndexError, RuntimeError):
            pass  # acceptable on empty Laplacian

    def test_compute_morse_index_empty_dim_out_of_range(self) -> None:
        lap = nerve_extras.spectral.Laplacian()
        with pytest.raises((IndexError, RuntimeError)):
            lap.compute_morse_index(0)


@pytest.mark.nerve_extras
class TestDiracOperator:
    """Tests for DiracOperator (empty/default constructed)."""

    def test_constructor(self) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        assert dirac is not None

    def test_get_dirac(self) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        matrix = dirac.get_dirac()
        assert isinstance(matrix, list)

    def test_get_dirac_squared(self) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        matrix = dirac.get_dirac_squared()
        assert isinstance(matrix, list)

    def test_eigenvalues(self) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        evals = dirac.eigenvalues()
        assert isinstance(evals, list)

    def test_eigenvectors(self) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        evecs = dirac.eigenvectors()
        assert isinstance(evecs, list)

    @pytest.mark.parametrize("method", [
        "get_spinor_laplacian",
        "get_chirality_operator",
        "compute_atiyah_singer_index",
        "compute_analytical_index",
        "compute_topological_index",
    ])
    def test_operator_methods(self, method: str) -> None:
        dirac = nerve_extras.spectral.DiracOperator()
        result = getattr(dirac, method)()
        assert result is not None


# nerve_extras.sheaf


@pytest.mark.nerve_extras
class TestSheafConfig:
    """Tests for SheafConfig."""

    def test_defaults(self) -> None:
        cfg = nerve_extras.sheaf.SheafConfig()
        # num_stalks and stalk_dimension default to 0
        assert cfg.num_stalks == 0
        assert cfg.stalk_dimension == 0
        assert hasattr(cfg, "use_parallel")
        assert hasattr(cfg, "use_simd")
        assert hasattr(cfg, "num_threads")
        assert hasattr(cfg, "gpu_batch_size")
        assert hasattr(cfg, "use_memory_pool")
        assert hasattr(cfg, "memory_pool_size")

    def test_field_assignment(self) -> None:
        cfg = nerve_extras.sheaf.SheafConfig()
        cfg.num_stalks = 10
        cfg.stalk_dimension = 3
        cfg.use_parallel = True
        cfg.num_threads = 4
        assert cfg.num_stalks == 10
        assert cfg.stalk_dimension == 3
        assert cfg.use_parallel is True
        assert cfg.num_threads == 4


@pytest.mark.nerve_extras
class TestSheafResult:
    """Tests for SheafResult."""

    def test_defaults(self) -> None:
        sr = nerve_extras.sheaf.SheafResult()
        assert sr.computation_time_ms == 0.0
        assert sr.success is False
        assert isinstance(sr.cohomology, list)

    def test_field_assignment(self) -> None:
        sr = nerve_extras.sheaf.SheafResult()
        sr.computation_time_ms = 1.5
        sr.success = True
        assert sr.computation_time_ms == 1.5
        assert sr.success is True


@pytest.mark.nerve_extras
class TestSheafHardwareInfo:
    """Tests for SheafHardwareInfo."""

    def test_detect(self) -> None:
        hw = nerve_extras.sheaf.detect_sheaf_hardware()
        assert hw is not None
        assert hw.num_cores > 0
        assert isinstance(hw.has_gpu, bool)
        assert isinstance(hw.has_avx512, bool)
        assert isinstance(hw.has_avx2, bool)
        # repr should work
        assert "SheafHardwareInfo" in repr(hw)


@pytest.mark.nerve_extras
class TestPoint:
    """Tests for the Point struct."""

    def test_defaults(self) -> None:
        pt = nerve_extras.sheaf.Point()
        assert pt.x == 0.0
        assert pt.y == 0.0
        assert pt.z == 0.0

    def test_field_assignment(self) -> None:
        pt = nerve_extras.sheaf.Point()
        pt.x = 1.0
        pt.y = 2.0
        pt.z = 3.0
        assert pt.x == 1.0
        assert pt.y == 2.0
        assert pt.z == 3.0

    def test_repr(self) -> None:
        pt = nerve_extras.sheaf.Point()
        pt.x = 1.5
        pt.y = 2.5
        pt.z = 3.5
        r = repr(pt)
        assert "1.5" in r
        assert "2.5" in r
        assert "3.5" in r


@pytest.mark.nerve_extras
class TestSheafEngine:
    """Tests for SheafEngine."""

    def test_constructor_default(self) -> None:
        cfg = nerve_extras.sheaf.SheafConfig()
        engine = nerve_extras.sheaf.SheafEngine(cfg)
        assert engine is not None

    def test_build_sheaf_with_point_objects(self) -> None:
        """build_sheaf should run without error."""
        cfg = nerve_extras.sheaf.SheafConfig()
        cfg.num_stalks = 3
        cfg.stalk_dimension = 2
        engine = nerve_extras.sheaf.SheafEngine(cfg)

        p1 = nerve_extras.sheaf.Point()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        p2 = nerve_extras.sheaf.Point()
        p2.x, p2.y, p2.z = 1.0, 0.0, 0.0
        p3 = nerve_extras.sheaf.Point()
        p3.x, p3.y, p3.z = 0.0, 1.0, 0.0
        positions = [p1, p2, p3]
        dimensions = [2, 2, 2]
        # build_sheaf returns None (configures the engine for morphism operations)
        engine.build_sheaf(positions, dimensions)


# nerve_extras.sheaf.parallel


@pytest.mark.nerve_extras
class TestStalkData:
    """Tests for StalkData."""

    def test_default_constructor(self) -> None:
        sd = nerve_extras.sheaf.parallel.StalkData()
        assert sd.id == 0
        assert sd.dimension == 0
        assert isinstance(sd.data, list)

    def test_custom_constructor(self) -> None:
        sd = nerve_extras.sheaf.parallel.StalkData(5, 3)
        assert sd.id == 5
        assert sd.dimension == 3


@pytest.mark.nerve_extras
class TestParallelSheafConfig:
    """Tests for ParallelSheafConfig."""

    def test_defaults(self) -> None:
        cfg = nerve_extras.sheaf.parallel.ParallelSheafConfig()
        # num_stalks and stalk_dimension default to 0
        assert cfg.num_stalks == 0
        assert cfg.stalk_dimension == 0
        assert hasattr(cfg, "use_simd")
        assert hasattr(cfg, "num_threads")

    def test_assignment(self) -> None:
        cfg = nerve_extras.sheaf.parallel.ParallelSheafConfig()
        cfg.num_stalks = 10
        cfg.num_threads = 4
        assert cfg.num_stalks == 10
        assert cfg.num_threads == 4


@pytest.mark.nerve_extras
class TestParallelSheafBuilder:
    """Tests for ParallelSheafBuilder."""

    def test_constructor(self) -> None:
        cfg = nerve_extras.sheaf.parallel.ParallelSheafConfig()
        builder = nerve_extras.sheaf.parallel.ParallelSheafBuilder(cfg)
        assert builder is not None

    def test_build_returns_stalks(self) -> None:
        cfg = nerve_extras.sheaf.parallel.ParallelSheafConfig()
        cfg.num_stalks = 4
        cfg.stalk_dimension = 2
        builder = nerve_extras.sheaf.parallel.ParallelSheafBuilder(cfg)
        builder.build()
        stalks = builder.get_stalks()
        assert isinstance(stalks, list)
        assert len(stalks) == 4
        for s in stalks:
            assert isinstance(s, nerve_extras.sheaf.parallel.StalkData)
            assert s.dimension == 2


@pytest.mark.nerve_extras
class TestSIMDStalkOperations:
    """Tests for SIMDStalkOperations static methods."""

    def test_add_stalks(self) -> None:
        a = nerve_extras.sheaf.parallel.StalkData(0, 3)
        a.data = [1.0, 2.0, 3.0]
        b = nerve_extras.sheaf.parallel.StalkData(1, 3)
        b.data = [4.0, 5.0, 6.0]
        result = nerve_extras.sheaf.parallel.StalkData(2, 3)

        nerve_extras.sheaf.parallel.SIMDStalkOperations.add_stalks(a, b, result)
        assert len(result.data) == 3
        assert math.isclose(result.data[0], 5.0, rel_tol=1e-6)
        assert math.isclose(result.data[1], 7.0, rel_tol=1e-6)
        assert math.isclose(result.data[2], 9.0, rel_tol=1e-6)

    def test_scale_stalk(self) -> None:
        a = nerve_extras.sheaf.parallel.StalkData(0, 3)
        a.data = [1.0, 2.0, 3.0]
        result = nerve_extras.sheaf.parallel.StalkData(1, 3)

        nerve_extras.sheaf.parallel.SIMDStalkOperations.scale_stalk(a, 2.0, result)
        assert math.isclose(result.data[0], 2.0, rel_tol=1e-6)
        assert math.isclose(result.data[1], 4.0, rel_tol=1e-6)
        assert math.isclose(result.data[2], 6.0, rel_tol=1e-6)

    def test_dot_product(self) -> None:
        a = nerve_extras.sheaf.parallel.StalkData(0, 3)
        a.data = [1.0, 2.0, 3.0]
        b = nerve_extras.sheaf.parallel.StalkData(1, 3)
        b.data = [4.0, 5.0, 6.0]

        dp = nerve_extras.sheaf.parallel.SIMDStalkOperations.dot_product(a, b)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert math.isclose(dp, 32.0, rel_tol=1e-6)

    def test_normalize_stalk(self) -> None:
        a = nerve_extras.sheaf.parallel.StalkData(0, 3)
        a.data = [3.0, 4.0, 0.0]

        nerve_extras.sheaf.parallel.SIMDStalkOperations.normalize_stalk(a)
        assert len(a.data) == 3
        # magnitude = sqrt(9 + 16 + 0) = 5.0
        assert math.isclose(a.data[0], 3.0 / 5.0, rel_tol=1e-6)
        assert math.isclose(a.data[1], 4.0 / 5.0, rel_tol=1e-6)
        assert math.isclose(a.data[2], 0.0, rel_tol=1e-6)


@pytest.mark.nerve_extras
class TestStalkSpatialHash:
    """Tests for StalkSpatialHash."""

    def test_constructor(self) -> None:
        sh = nerve_extras.sheaf.parallel.StalkSpatialHash()
        assert sh is not None

    def test_insert_and_get_nearby(self) -> None:
        sh = nerve_extras.sheaf.parallel.StalkSpatialHash(2.0)
        pt1 = nerve_extras.sheaf.Point()
        pt1.x, pt1.y, pt1.z = 0.0, 0.0, 0.0
        pt2 = nerve_extras.sheaf.Point()
        pt2.x, pt2.y, pt2.z = 1.0, 0.0, 0.0
        pt3 = nerve_extras.sheaf.Point()
        pt3.x, pt3.y, pt3.z = 10.0, 10.0, 10.0

        sh.insert_stalk(0, pt1)
        sh.insert_stalk(1, pt2)
        sh.insert_stalk(2, pt3)

        # Points near (0, 0, 0) should include 0 and 1
        nearby = sh.get_nearby_stalks(pt1)
        assert 0 in nearby
        assert 1 in nearby

        # Point 2 is far away
        assert 2 not in nearby


@pytest.mark.nerve_extras
class TestSheafParallelBenchmark:
    """Tests for SheafParallelBenchmark struct."""

    def test_defaults(self) -> None:
        bm = nerve_extras.sheaf.parallel.SheafParallelBenchmark()
        assert bm.sequential_time_ms >= 0
        assert bm.parallel_time_ms >= 0
        assert bm.simd_time_ms >= 0
        assert bm.num_stalks >= 0
        assert bm.stalk_dim >= 0
        assert bm.num_threads >= 0
        assert hasattr(bm, "speedup_parallel")
        assert hasattr(bm, "speedup_simd")

    def test_field_assignment(self) -> None:
        bm = nerve_extras.sheaf.parallel.SheafParallelBenchmark()
        bm.sequential_time_ms = 100.0
        bm.parallel_time_ms = 50.0
        bm.speedup_parallel = 2.0
        assert bm.sequential_time_ms == 100.0
        assert bm.parallel_time_ms == 50.0
        assert bm.speedup_parallel == 2.0


@pytest.mark.nerve_extras
class TestBenchmarkParallelSheaf:
    """Tests for benchmark_parallel_sheaf function."""

    def test_benchmark_runs(self) -> None:
        result = nerve_extras.sheaf.parallel.benchmark_parallel_sheaf(10, 4, 2)
        assert isinstance(result, nerve_extras.sheaf.parallel.SheafParallelBenchmark)
        assert result.num_stalks == 10
        assert result.stalk_dim == 4
        assert result.num_threads == 2
        assert result.sequential_time_ms >= 0
        assert result.parallel_time_ms >= 0


# nerve_extras.sheaf.gpu


@pytest.mark.nerve_extras
class TestSheafGPUBenchmark:
    """Tests for SheafGPUBenchmark struct."""

    def test_defaults(self) -> None:
        bm = nerve_extras.sheaf.gpu.SheafGPUBenchmark()
        assert bm.cpu_time_ms >= 0
        assert bm.gpu_time_ms >= 0
        assert bm.speedup >= 0
        assert bm.num_stalks >= 0
        assert bm.stalk_dim >= 0

    def test_field_assignment(self) -> None:
        bm = nerve_extras.sheaf.gpu.SheafGPUBenchmark()
        bm.cpu_time_ms = 200.0
        bm.gpu_time_ms = 50.0
        bm.speedup = 4.0
        assert bm.cpu_time_ms == 200.0
        assert bm.gpu_time_ms == 50.0
        assert bm.speedup == 4.0


# nerve_extras.sheaf.morphism


@pytest.mark.nerve_extras
class TestSparseMorphism:
    """Tests for SparseMorphism."""

    def test_defaults(self) -> None:
        m = nerve_extras.sheaf.morphism.SparseMorphism()
        assert m.from_dim == 0
        assert m.to_dim == 0
        assert isinstance(m.row_ptr, list)
        assert isinstance(m.col_idx, list)
        assert isinstance(m.values, list)

    def test_field_assignment(self) -> None:
        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 3
        m.to_dim = 2
        m.row_ptr = [0, 2, 3]
        m.col_idx = [0, 1, 0]
        m.values = [1.0, 2.0, 3.0]
        assert m.from_dim == 3
        assert m.to_dim == 2
        assert len(m.row_ptr) == 3

    def test_apply(self) -> None:
        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 3
        m.to_dim = 2
        m.row_ptr = [0, 2, 3]
        m.col_idx = [0, 1, 0]
        m.values = [1.0, 2.0, 3.0]

        output = m.apply([1.0, 2.0, 3.0])
        assert len(output) == 2
        # output[0] = 1*1.0 + 2*2.0 + 3*3.0 -> row 0 has cols 0,1 with values 1.0,2.0
        # Actually wait, row_ptr[0]=0, row_ptr[1]=2 means row 0 has 2 entries: col_idx[0]=0 val=1.0, col_idx[1]=1 val=2.0
        # So output[0] = 1.0*1.0 + 2.0*2.0 = 1 + 4 = 5.0
        # row_ptr[1]=2, row_ptr[2]=3 means row 1 has 1 entry: col_idx[2]=0 val=3.0
        # output[1] = 3.0*1.0 = 3.0
        assert math.isclose(output[0], 5.0, rel_tol=1e-6)
        assert math.isclose(output[1], 3.0, rel_tol=1e-6)

    def test_apply_simd(self) -> None:
        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 2
        m.to_dim = 2
        m.row_ptr = [0, 1, 2]
        m.col_idx = [0, 1]
        m.values = [2.0, 3.0]

        output = m.apply_simd([1.0, 2.0])
        assert len(output) == 2
        assert math.isclose(output[0], 2.0, rel_tol=1e-6)
        assert math.isclose(output[1], 6.0, rel_tol=1e-6)


@pytest.mark.nerve_extras
class TestMorphismMemoryPool:
    """Tests for MorphismMemoryPool."""

    def test_constructor_default(self) -> None:
        pool = nerve_extras.sheaf.morphism.MorphismMemoryPool()
        assert pool is not None

    def test_constructor_custom_size(self) -> None:
        pool = nerve_extras.sheaf.morphism.MorphismMemoryPool(2 * 1024 * 1024)
        assert pool is not None

    def test_allocate_and_reset(self) -> None:
        pool = nerve_extras.sheaf.morphism.MorphismMemoryPool(4096)
        morph = pool.allocate_morphism(10)
        assert morph is not None
        pool.reset()


@pytest.mark.nerve_extras
class TestBatchedMorphismComputer:
    """Tests for BatchedMorphismComputer."""

    def test_constructor(self) -> None:
        computer = nerve_extras.sheaf.morphism.BatchedMorphismComputer()
        assert computer is not None

    def test_constructor_custom_block(self) -> None:
        computer = nerve_extras.sheaf.morphism.BatchedMorphismComputer(128)
        assert computer is not None

    def test_add_and_compute_batch(self) -> None:
        computer = nerve_extras.sheaf.morphism.BatchedMorphismComputer(256)

        # Create a morphism and add it
        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 2
        m.to_dim = 2
        m.row_ptr = [0, 1, 2]
        m.col_idx = [0, 1]
        m.values = [1.0, 1.0]

        computer.add_morphism(0, 1, m)
        # compute_batch should work even with simple inputs
        output = computer.compute_batch([0, 1], [[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(output, list)


@pytest.mark.nerve_extras
class TestMorphismCompositionOptimizer:
    """Tests for MorphismCompositionOptimizer."""

    def test_constructor(self) -> None:
        opt = nerve_extras.sheaf.morphism.MorphismCompositionOptimizer()
        assert opt is not None

    def test_add_morphism_direct(self) -> None:
        """add_morphism should accept a SparseMorphism between two stalks."""
        opt = nerve_extras.sheaf.morphism.MorphismCompositionOptimizer()

        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 2
        m.to_dim = 2
        m.row_ptr = [0, 1, 2]
        m.col_idx = [0, 1]
        m.values = [1.0, 1.0]

        # Adding a direct morphism should work without registering a chain first
        opt.add_morphism(0, 1, m)

    def test_register_chain_direct(self) -> None:
        """register_chain should accept a list of stalk IDs."""
        opt = nerve_extras.sheaf.morphism.MorphismCompositionOptimizer()

        m = nerve_extras.sheaf.morphism.SparseMorphism()
        m.from_dim = 2
        m.to_dim = 2
        m.row_ptr = [0, 1, 2]
        m.col_idx = [0, 1]
        m.values = [1.0, 1.0]

        # Add morphisms first, then register the chain
        opt.add_morphism(0, 1, m)
        opt.add_morphism(1, 2, m)
        opt.register_chain([0, 1, 2])


@pytest.mark.nerve_extras
class TestAsyncMorphismQueue:
    """Tests for AsyncMorphismQueue."""

    def test_constructor(self) -> None:
        queue = nerve_extras.sheaf.morphism.AsyncMorphismQueue()
        assert queue is not None

    def test_start_stop(self) -> None:
        queue = nerve_extras.sheaf.morphism.AsyncMorphismQueue()
        queue.start(2)
        queue.stop()
        # Should complete without hanging
        assert True

    def test_start_stop_default(self) -> None:
        queue = nerve_extras.sheaf.morphism.AsyncMorphismQueue()
        queue.start()
        queue.stop()
        assert True
