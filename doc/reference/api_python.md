# Python API Reference

## `pynerve.compute_persistence`

```python
pynerve.compute_persistence(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    *,
    engine: PersistenceEngine | str = PersistenceEngine.AUTO,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    error_tolerance: float | None = None,
    dtype: str | None = None,
    max_radius_cap: float | None = None,
) -> PersistenceResult
```

Compute persistent homology for a point cloud. Given the same inputs, results are bitwise deterministic.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `PointCloud` | required | Point cloud as an `(N, D)` array (NumPy or PyTorch) |
| `options` | `PersistenceOptions \| None` | `None` | Preset configuration object |
| `engine` | `PersistenceEngine \| str` | `AUTO` | Engine: `AUTO`, `PH0`, `PH3`--`PH6` |
| `max_dim` | `int` | `2` | Maximum homology dimension |
| `max_radius` | `float \| None` | `None` | Filtration radius cutoff (auto-selected if `None`) |
| `mode` | `PersistenceMode` | `EXACT` | `EXACT` or `APPROX` |
| `backend` | `PersistenceBackend \| None` | `None` | Computation backend (auto-selected) |
| `threads` | `int \| None` | `None` | Thread count (0 = auto) |
| `device` | `str \| None` | `None` | Device: `"cpu"`, `"cuda"`, `"cuda:N"` |
| `seed` | `int \| None` | `None` | RNG seed for determinism |
| `error_tolerance` | `float \| None` | `None` | Approximation tolerance |
| `dtype` | `str \| None` | `None` | NumPy dtype name for point array conversion (e.g. `"float32"`) |
| `max_radius_cap` | `float \| None` | `None` | Override upper bound for `max_radius=inf` |

An optional `PersistenceOptions` object can be passed; keyword arguments override its fields. `mode` selects between `EXACT` and `APPROX` computation. `backend` chooses the execution backend: `CPU_EXACT`, `CPU_ADAPTIVE_ACCELERATION`, or `CUDA_HYBRID`. `threads` controls CPU thread count, defaulting to auto-detection. `device` selects the compute device: `"cpu"` (default) or `"cuda"` (or `"cuda:N"` for a specific GPU), overriding `backend`. `seed` sets the RNG seed for reproducible results; all computations are bitwise reproducible given the same input and seed. `error_tolerance` sets the approximation tolerance for `APPROX` mode.

**Returns:** A `PersistenceResult` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `pairs` | `list[tuple[float, float, int]]` | `(birth, death, dimension)` tuples |
| `betti_numbers` | `list[int]` | Betti numbers [b0, b1, ...] |
| `num_pairs` | `int` | Total persistence pair count |
| `max_dim` | `int` | Computed homology dimension |
| `max_radius` | `float` | Filtration cutoff used |
| `diagnostics` | `dict[str, Any]` | Engine-specific diagnostic info |

**Complexity:**

The distance matrix computation is O(n^2 * d) sequentially and O(n^2 * d / p) in parallel. Filtration is O(m log m) where m represents O(n^{max_dim+1}) simplices. Reduction is O(m^3) worst-case, but typically O(m * r) where r is the number of columns. Host memory is O(n^2) for the distance matrix, while GPU memory is O(n * max_radius^{dim}) for sparse distance storage.

**Errors:**

`InvalidArgumentError` is raised when points dimension is not 2 or contains NaN or infinity values. `ShapeMismatchError` occurs when points has fewer than 2 rows. `DimensionError` occurs when max_dim exceeds 5 or available memory. `GPUMemoryError` indicates GPU out-of-memory during CUDA_HYBRID execution. `BackendRequiredError` is raised when CUDA_HYBRID is selected but no GPU is available. `PynerveMemoryError` signals host out-of-memory during distance matrix allocation. `ConvergenceError` occurs when PH5 iterative refinement does not converge.

**Examples:**

```python
import pynerve
import numpy as np

points = np.random.randn(500, 3)
result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)
print(result.betti_numbers)  # [1, 3, 1] (approx)

# GPU acceleration
result = pynerve.compute_persistence(
    points, max_dim=2, device="cuda"
)

# Approximate for speed
result = pynerve.compute_persistence(
    points, max_dim=2, mode="approx", error_tolerance=0.1
)
```


## Engine shortcuts

```python
def compute_persistence_ph3(points, options=None, **kwargs) -> PersistenceResult
def compute_persistence_ph4(points, options=None, **kwargs) -> PersistenceResult
def compute_persistence_ph5(points, options=None, **kwargs) -> PersistenceResult
def compute_persistence_ph6(points, options=None, **kwargs) -> PersistenceResult
```

Convenience wrappers with `engine` pre-selected.


## `pynerve.compute_persistence_ph3`

```python
pynerve.compute_persistence_ph3(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    **kwargs: Any,
) -> PersistenceResult
```

Compute persistence using the PH3 engine. Cohomology-based reduction for small-to-medium point clouds (n < 10K). Same parameters and return type as `compute_persistence`.

**Best for:** n < 10,000, cohomology-based exact computation.


## `pynerve.compute_persistence_ph4`

```python
pynerve.compute_persistence_ph4(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    **kwargs: Any,
) -> PersistenceResult
```

Compute persistence using the PH4 engine. Exact VR for small-to-moderate point clouds (n < 10K). Same parameters and return type as `compute_persistence`.

**Cost:** O(n^2 * d) distance + O(n^{d+1} log n) filtration + O(n^{3d+3}) worst-case reduction.

**Best for:** n < 10,000, exact results required.


## `pynerve.compute_persistence_ph5`

```python
pynerve.compute_persistence_ph5(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    **kwargs: Any,
) -> PersistenceResult
```

Compute persistence using the PH5 engine. Approximate VR with iterative refinement. Suitable for 10K-1M points. Uses sparse distance matrix with configurable compression.

**Cost:** O(n * k * d) for sparse distance (k = neighbors) + iterative refinement O(m^{d+1}).

**Best for:** 10K - 1M points, configurable accuracy.


## `pynerve.compute_persistence_ph6`

```python
pynerve.compute_persistence_ph6(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    **kwargs: Any,
) -> PersistenceResult
```

Compute persistence using the PH6 engine. Block-sparse reduction optimized for witness and large-scale VR. Suitable for 100K-10M+ points.

**Cost:** O(k * n * d) for witness distance (k = landmarks) + O(k^{d+1}) for block-sparse reduction.

**Best for:** 100K - 10M+ points, block-sparse reduction.


## `pynerve.persistence_image`

```python
pynerve.persistence_image(
    diagram: PersistenceDiagramLike,
    *,
    resolution: int | tuple[int, int] = 20,
    sigma: float = 0.1,
    birth_range: tuple[float, float] | None = None,
    persistence_range: tuple[float, float] | None = None,
    weight: str = "persistence",
) -> np.ndarray
```

Convert a persistence diagram to a persistence image (2D array).

**Parameters:**

The `diagram` parameter is a PersistenceDiagramLike array of shape (n, 2+) containing persistence pairs (birth, death, ...). Infinite deaths are filtered out automatically. `resolution` controls the output image resolution, defaulting to 20 for both dimensions, or can be specified as a (height, width) tuple. `sigma` sets the Gaussian kernel width and must be positive, defaulting to 0.1. `birth_range` and `persistence_range` optionally set axis bounds and are auto-scaled when left as None. `weight` selects between persistence-based and uniform weighting.

**Returns:** `np.ndarray` of shape `(height, width)`, dtype `float64`. Complexity is O(n * H * W) where n is the number of finite pairs.

**Errors:**

`InvalidArgumentError` is raised when the diagram is not 2D, sigma is not positive, or resolution is less than 1. `PrecisionError` occurs when all pairs are infinite or out of the specified range.

**Examples:**

```python
import pynerve
import numpy as np

points = np.random.randn(200, 2)
result = pynerve.compute_persistence(points)
diagram = np.array(result.pairs)[:, :2]  # (birth, death)

image = pynerve.persistence_image(
    diagram,
    resolution=(16, 16),
    sigma=0.2,
    weight="persistence",
)
print(image.shape)  # (16, 16)
```


## `pynerve.torch.alpha_persistence`

```python
pynerve.torch.alpha_persistence(
    points: torch.Tensor,
    max_dim: int = 2,
) -> PersistenceDiagram
```

Compute persistence via the Alpha complex (Delaunay triangulation). Requires PyTorch and only works for geometric data in dimensions 2-3. Available in the `pynerve.torch` sub-package.
See [PyTorch reference](../torch/torch.md) for details.


## `pynerve.torch.witness_persistence`

```python
pynerve.torch.witness_persistence(
    landmarks: torch.Tensor,
    points: torch.Tensor,
    max_dim: int = 2,
    max_radius: float = 1.0,
) -> PersistenceDiagram
```

Compute persistence via the Witness complex. Requires PyTorch. Available in the `pynerve.torch` sub-package.
See [PyTorch reference](../torch/torch.md) for details.


## Distributed persistence

Multi-node MPI-based persistence is available as a C++ feature only (requires MPI-enabled build).
Use `pynerve.mp_shared` for single-node parallelism, or configure MPI at the C++ level.
See the [Distributed guide](../guides/distributed.md) for details.


## Streaming persistence

Chunked persistence for datasets that exceed memory is available via `pynerve.async_api.stream_persistence`.
See the [Quickstart](../quickstart.md) and [Streaming](../streaming/streaming.md) docs for usage.


## `pynerve.update_persistence`

```python
pynerve.update_persistence(
    events: Iterable[tuple[EventType | str, Sequence[int]]],
    options: PersistenceOptions | None = None,
    *,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    error_tolerance: float | None = None,
) -> PersistenceResult
```

Update persistence with incremental add/remove events.

**Parameters:**

`events` is a list of tuples where each event is either `("add", [i, j, ...])` or `("remove", [i, j, ...])`. `options` is an optional PersistenceOptions instance for configuration.

**Cost:** O(k * log n) per event where k = size of the affected simplex.


## `pynerve.async_api`

Async-first operations for non-blocking persistence in event loops.

```python
import pynerve

# Single async compute
result = await pynerve.async_api.compute_persistence_async(points)

# Load diagrams concurrently
diagrams = await pynerve.async_api.load_diagrams_async(
    ["diag_1.npy", "diag_2.npy"],
    max_concurrent=8,
)

# Stream from file
async for result in pynerve.async_api.stream_persistence(
    "data.h5", chunk_size=500
):
    process(result)
```

**Functions:**

`compute_persistence_async` performs non-blocking persistence computation via a thread pool. `load_diagrams_async` loads multiple diagram files concurrently with configurable concurrency. `stream_persistence` is an async generator that yields persistence results as they become available.


## Determinism

All persistence computations are bitwise reproducible by default.
Use the `seed` parameter in any compute function to ensure consistent results across runs:

```python
import pynerve

result_a = pynerve.compute_persistence(points, seed=42)
result_b = pynerve.compute_persistence(points, seed=42)
assert result_a == result_b  # holds
```


## Package version

```python
pynerve.__version__  # e.g. "0.9.6"
```


## `pynerve.diagnostics`

```python
pynerve.diagnostics.report(result: dict) -> str
pynerve.diagnostics.plot_pairs(pairs: list) -> plt.Figure
pynerve.diagnostics.plot_betti(betti_numbers: list) -> plt.Figure
pynerve.diagnostics.comparison_plot(results: list[dict]) -> plt.Figure
```

`DiagnosticsCollector` is a context manager that tracks timing and memory across operations. `report` generates a text summary of a computation result. `plot_pairs` creates a persistence diagram plot showing birth versus death. `plot_betti` produces a bar chart of Betti numbers. `comparison_plot` generates a side-by-side comparison of multiple computation runs.


## `pynerve.nn` module -- Neural Network Layers

### `PersistentHomology`

```python
class pynerve.nn.PersistentHomology(
    max_dim: int = 2,
    max_radius: float = 1.0,
    device: str = "cpu",
    mode: str = "exact",
)
```

PyTorch module for differentiable persistence. Compatible with `nn.Sequential` and `autograd`.

**Parameters:**

`max_dim` sets the maximum homology dimension, defaulting to 2. `max_radius` controls the maximum VR edge length, defaulting to 1.0. `device` specifies the computation device, defaulting to CPU. `mode` selects between exact and approximate computation.

**Forward:** `(batch, n, d) -> (batch, feature_dim)`

### `SparsePH`

```python
class pynerve.nn.SparsePH(
    max_dim: int = 2,
    max_radius: float = float("inf"),
    landmark_ratio: float = 0.1,
    metric: str = "euclidean",
)
```

PyTorch module for sparse persistence.

### `WindowedPH`

```python
class pynerve.nn.WindowedPH(
    window_size: int = 512,
    stride: int = 256,
    max_dim: int = 1,
    max_radius: float = 2.0,
    overlap_handling: str = "concat",
)
```

PyTorch module for sliding-window persistence on time-series point clouds.

### `WitnessComplexPersistence`

```python
class pynerve.nn.WitnessComplexPersistence(
    n_landmarks: int = 100,
    max_dim: int = 2,
    method: str = "farthest",
)
```

PyTorch module for witness complex persistence.


## Type Aliases

| Alias | Definition | Description |
|-------|------------|-------------|
| `PointCloud` | `np.ndarray \| Tensor` | `(N, D)` array of points |
| `DistanceMatrix` | `np.ndarray \| Tensor` | `(N, N)` pairwise distances |
| `PersistencePair` | `tuple[float, float, int]` | `(birth, death, dimension)` |
| `PersistenceDiagramLike` | `np.ndarray` | `(n, 2+)` (birth, death, ...) |
| `ArrayLike` | `np.ndarray \| Tensor \| list \| tuple` | Generic array input |
| `Numeric` | `int \| float \| np.number \| Tensor` | Numeric value |
| `DistanceMetric` | `...` | `"euclidean"` \| `"manhattan"` \| `"cosine"` \| ... |
| `VectorizationMethod` | `...` | `"persistence_image"` \| `"betti"` \| `"landscape"` \| ... |
| `FilterFunction` | `callable` | `f(points) -> array` |
| `ClusteringAlgorithm` | `...` | `"kmeans"` \| `"dbscan"` \| `"hdbscan"` \| ... |


## Error Classes

All errors inherit from `PynerveError`.

| Exception | Raised when |
|-----------|-------------|
| `PynerveError` | Base class for all errors |
| `PersistenceError` | Persistence computation failure |
| `BackendRequiredError` | Required backend not loaded |
| `ShapeMismatchError` | Tensor shape validation failure |
| `InvalidArgumentError` | Invalid input parameter |
| `GPUError` | GPU-related failure |
| `PynerveMemoryError` | Memory allocation failure |
| `ConvergenceError` | Algorithm did not converge |

```python
pynerve.PynerveError                       # Base
pynerve.GPUError                         # GPU runtime errors
pynerve.GPUMemoryError                   # GPU OOM
pynerve.GPULaunchError                   # Kernel launch failure
pynerve.InvalidArgumentError             # Invalid input parameter
pynerve.ShapeMismatchError               # Wrong array shape
pynerve.DimensionError                   # Dimension exceeds max
pynerve.MatrixStructureError             # Boundary matrix malformed
pynerve.InvalidSimplexError              # Simplex invalid
pynerve.PynerveMemoryError               # Host OOM
pynerve.OutOfMemoryError                 # Resource exhausted
pynerve.AllocationError                  # Pool allocation failed
pynerve.ConvergenceError                 # Iterative algorithm did not converge
pynerve.PrecisionError                   # Numerical precision insufficient
pynerve.NumericalError                   # General numerical issue
pynerve.NumericalInstabilityError        # Unstable computation detected
pynerve.PersistenceError                 # PH computation failed
pynerve.BettiError                       # Betti number computation failed
pynerve.DeterminismError                 # Determinism contract violated
pynerve.PynerveIOError                   # File I/O failure
pynerve.NUMAError                        # NUMA policy violation
pynerve.BudgetExceededError              # Resource budget exceeded
pynerve.BackendRequiredError             # Required backend not available
```


## Accelerated Operations

Pynerve operations support multiple backends. Euclidean, Manhattan, and Cosine distance computations all support SIMD via AVX-512 and AVX2, GPU acceleration, multi-GPU (Euclidean only), MPI distribution, and deterministic execution. VR persistence uses SIMD for filtration and GPU for reduction, with MPI and deterministic support. Cohomology and matrix reduction use SIMD and GPU with deterministic output. Persistence image uses SIMD and GPU. Wasserstein and bottleneck distances are GPU-accelerated and deterministic. Mapper, sheaf Laplacian, spectral operations, filtration, and streaming distance all combine SIMD with GPU, MPI where noted, and deterministic execution.


## Diagram distances

Bottleneck and Wasserstein distances between persistence diagrams are available in `pynerve.torch`:

```python
pynerve.torch.diagram_bottleneck(diagram_1, diagram_2, matching=False)
pynerve.torch.diagram_wasserstein(diagram_1, diagram_2, p=2.0, q=2.0)
```


## `pynerve.fast_ops.betti_curve`

```python
pynerve.fast_ops.betti_curve(
    pairs: np.ndarray,
    max_dim: int = 3,
    resolution: int = 100,
    max_time: float | None = None,
) -> np.ndarray
```

Compute Betti curves from persistence pairs. Returns array of shape `(max_dim + 1, resolution)`.


## `pynerve.fast_ops.persistence_landscape`

```python
pynerve.fast_ops.persistence_landscape(
    pairs: np.ndarray,
    n_layers: int = 5,
    resolution: int = 100,
) -> np.ndarray
```

Compute persistence landscapes from birth-death pairs. Returns array of shape `(n_layers, resolution)`.


## `pynerve.PersistenceOptions`

```python
@dataclass(frozen=True)
class pynerve.PersistenceOptions:
    mode: PersistenceMode = PersistenceMode.EXACT
    backend: PersistenceBackend = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION
    max_dim: int = 2
    max_radius: float | None = None
    threads: int = 0
    error_tolerance: float = 0.0

    def replace(self, **kwargs) -> PersistenceOptions:
        """Return a new instance with the given fields replaced."""
```

Immutable config object for persistence computation. Use `replace()` to create modified copies.
`max_radius` is `None` by default (auto-selected). Pass an explicit finite value to override.


## `pynerve.PersistenceEngine`

```python
class pynerve.PersistenceEngine(Enum):
    AUTO = "auto"   # Auto-selects PH0, PH3--PH6 based on input
    PH0 = "ph0"     # Standard homology (compatibility mode)
    PH3 = "ph3"     # Cohomology (small-to-medium datasets)
    PH4 = "ph4"     # Cohomology + aggressive clearing (dim <= 4)
    PH5 = "ph5"     # Unified adaptive engine (10K--1M points)
    PH6 = "ph6"     # Block-sparse speculative engine (>1M points)
```

Selects the persistence computation engine. Use `AUTO` for automatic selection based on input size and dimensionality.


## `pynerve.PersistenceMode`

```python
class pynerve.PersistenceMode(Enum):
    EXACT = "EXACT"
    APPROX = "APPROX"
```


## `pynerve.PersistenceBackend`

```python
class pynerve.PersistenceBackend(Enum):
    CPU_EXACT = "CPU_EXACT"
    CPU_ADAPTIVE_ACCELERATION = "CPU_ADAPTIVE_ACCELERATION"
    CUDA_HYBRID = "CUDA_HYBRID"
```


## `pynerve.torch.PersistenceDiagram`

```python
class pynerve.torch.PersistenceDiagram:
    """Batched persistence diagram container with autograd support."""

    diagrams: torch.Tensor     # [batch, max_pairs, 3]
    mask: torch.Tensor         # [batch, max_pairs] valid-pair mask
```

Batched persistence diagram container for PyTorch integration.
See the [PyTorch reference](../torch/persistence_diagram.md) for details.


## `pynerve.cupy_ops` submodule

```python
pynerve.cupy_ops.CuPyPersistence(max_dim=2, max_radius=1.0)
pynerve.cupy_ops.compute_diagrams_cupy(points_cupy, max_dim=2)
```

CuPy-backed GPU operations for users who prefer CuPy over PyTorch.


## `pynerve.fast_ops` submodule

```python
pynerve.fast_ops.boundary_matrix(simplices, max_dim, max_radius) -> CSRMatrix
pynerve.fast_ops.boundary_matrix_sparse(simplices, max_dim, max_radius) -> CSRMatrix
pynerve.fast_ops.column_reduction(boundary) -> list[tuple]
pynerve.fast_ops.column_reduction_sparse(boundary) -> list[tuple]
pynerve.fast_ops.vietoris_rips_filtration(points, max_dim, max_radius) -> list[Simplex]
pynerve.fast_ops.vietoris_rips_filtration_fast(points, max_dim, max_radius) -> list[Simplex]
pynerve.fast_ops.nearest_neighbors(points, k) -> tuple[ndarray, ndarray]
pynerve.fast_ops.nearest_neighbors_fast(points, k) -> tuple[ndarray, ndarray]
```

Low-level C++ accelerated operations, accessible without going through the high-level API.


## `pynerve.mp_shared` submodule

```python
pynerve.mp_shared.compute_persistence_parallel(chunks, n_workers, use_shared_memory) -> list[PersistenceResult]
pynerve.mp_shared.ParallelPH(max_dim, n_workers) -> ParallelPH
pynerve.mp_shared.ChunkedParallel(chunk_size, n_workers) -> ChunkedParallel
pynerve.mp_shared.MapReducePH(n_workers) -> MapReducePH
```

Shared-memory multiprocessing for single-node parallelism. Uses Python's `multiprocessing` with shared memory for zero-copy data transfer.


## `pynerve.jit` submodule

```python
pynerve.jit.jit_distance(func) -> callable
pynerve.jit.jit_filtration(func) -> callable
pynerve.jit.jit_reduction(func) -> callable
```

JIT compilation of custom distance, filtration, or reduction functions using Numba or Cython.


## `pynerve.diagnostics` module

```python
pynerve.diagnostics.DiagnosticsCollector()  # context manager for timing
pynerve.diagnostics.report(result: PersistenceResult) -> str
pynerve.diagnostics.plot_pairs(pairs: list[tuple], **kwargs) -> plt.Figure
pynerve.diagnostics.plot_betti(betti: list[int], **kwargs) -> plt.Figure
pynerve.diagnostics.plot_image(image: np.ndarray, **kwargs) -> plt.Figure
pynerve.diagnostics.plot_landscape(landscape: np.ndarray, **kwargs) -> plt.Figure
pynerve.diagnostics.comparison_plot(results: list[PersistenceResult], **kwargs) -> plt.Figure
```

Utility functions for visualization and analysis. All plots return `matplotlib.figure.Figure` objects.


## `pynerve.torch` submodule

```python
pynerve.torch.PersistenceLayer(max_dim=2) -> nn.Module
pynerve.torch.SparsePersistenceLayer(max_dim=2, landmark_ratio=0.1) -> nn.Module
pynerve.torch.PersistenceImageLayer(resolution=20, sigma=0.1) -> nn.Module
pynerve.torch.BettiLayer(resolution=100) -> nn.Module
pynerve.torch.LandscapeLayer(resolution=100, num_landscapes=5) -> nn.Module
pynerve.torch.SilhouetteLayer(resolution=100) -> nn.Module
pynerve.torch.HeatKernelLayer(resolution=100, sigma=0.1) -> nn.Module
pynerve.torch.MapperLayer(resolution=20, gain=0.3) -> nn.Module
```

Differentiable PyTorch layers for all major vectorization methods. Compatible with `nn.Sequential` and `autograd`.


## Constants

```python
pynerve.__version__: str
```

### Medium data (1,000 < n < 10,000)

```python
# Exact VR, multi-threaded, SIMD dispatch
result = pynerve.compute_persistence(points, max_dim=2)
```

### Large data (10,000 < n < 100,000)

```python
# GPU acceleration or sparse mode
result = pynerve.compute_persistence(
    points, max_dim=2,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,  # GPU
    # or:
    # mode=pynerve.PersistenceMode.APPROX,  # sparse CPU
)
```

### Very large data (100,000 < n < 1,000,000)

```python
# Sparse + GPU
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=0.01,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)
```

### Massive data (n > 1,000,000)

```python
# Streaming + sparse + distributed (if MPI available)
from pynerve.async_api import stream_persistence

async for result in stream_persistence(
    "large_file.h5", chunk_size=5000, use_gpu=True, max_dim=2,
):
    print(result.betti_numbers)
```
