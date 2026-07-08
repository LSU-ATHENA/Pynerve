# Full API reference

### `pynerve.compute_persistence`

```python
def compute_persistence(
    points: np.ndarray,
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
) -> PersistenceResult:
```

The `compute_persistence` function accepts the following parameters. `points` is an `(n, d)` array and is required --- it provides the input point cloud. `max_dim` is an integer defaulting to 2, specifying the maximum homology dimension. `max_radius` is a float defaulting to None (auto-selected), setting the filtration cutoff radius. `engine` is a string or PersistenceEngine defaulting to `"auto"`, selecting the reduction engine. `error_tolerance` is a float defaulting to None, used to treat near-equal birth times as equal. `device` selects CPU or GPU backend.

**Returns** a `PersistenceResult` with fields:
- `pairs`: list of `(birth, death, dim)` tuples
- `betti_numbers`: list of Betti numbers per dimension
- `diagnostics`: dict with optional diagnostic info

### `pynerve.compute_persistence_up_to_dim_4`

```python
def compute_persistence_up_to_dim_4(
    points: np.ndarray,
    options: PersistenceOptions | None = None,
    *,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    error_tolerance: float | None = None,
) -> PersistenceResult:
```

Persistence capped at dimension 4 (same as Ph4). Uses clearing optimization
with cohomology-based reduction for maximum throughput on filtration sizes
up to $10^6$ simplices.

### `pynerve.compute_persistence_up_to_dim_5`

```python
def compute_persistence_up_to_dim_5(
    points: np.ndarray,
    options: PersistenceOptions | None = None,
    *,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    error_tolerance: float | None = None,
) -> PersistenceResult:
```

Persistence capped at dimension 5 (same as Ph5). Balanced exact/approximate
reduction that falls back to sparse approximation when the boundary matrix
exceeds a configurable threshold.

### `pynerve.compute_persistence_up_to_dim_6`

```python
def compute_persistence_up_to_dim_6(
    points: np.ndarray,
    options: PersistenceOptions | None = None,
    *,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    error_tolerance: float | None = None,
) -> PersistenceResult:
```

Persistence capped at dimension 6 (same as Ph6). High-precision accelerated
reduction using double-precision arithmetic throughout with iterative
refinement to resolve near-degeneracies.

### `pynerve.torch.vr_persistence`

```python
def vr_persistence(
    points: torch.Tensor,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    metric: str = "euclidean",
    return_simplices: bool = False,
) -> PersistenceDiagram | tuple[Any, ...]:
```

The `vr_persistence` function accepts these parameters. `points` is a `(*batch, n, d)` tensor and is required, supporting batched input. `max_dim` is an int defaulting to 1, setting the maximum homology dimension. `max_radius` is a float defaulting to inf, the filtration cutoff. `metric` is a string defaulting to `"euclidean"`, naming the metric. `return_simplices` is a boolean defaulting to False, controlling whether simplex indices are returned.

Returns a `PersistenceDiagram` with:
- `diagrams`: tensor `(*batch, max_pairs, 3)` where last dim is `(birth, death, dim)`
- `mask`: boolean tensor marking valid pairs
- `simplices` (if `return_simplices=True`): list of `(birth_idx, death_idx)` tuples
- `births()`: extract birth times
- `deaths()`: extract death times
- `total_persistence(p=2.0)`: sum of $(death - birth)^p$

### `pynerve.nn.PersistentHomology`

```python
class PersistentHomology(torch.nn.Module):
    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        reduction: str = "clearing",
        memory_mode: str = "standard",
        max_memory_gb: float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
```

The `PersistentHomology` constructor accepts these parameters. `max_dim` is an int defaulting to 1, the maximum homology dimension. `max_radius` is a float defaulting to inf, the filtration cutoff. `metric` is a string defaulting to `"euclidean"`, specifying the distance metric. `reduction` is a string defaulting to `"clearing"`, accepting `"standard"`, `"clearing"`, or `"cohomology"`. `memory_mode` is a string defaulting to `"standard"`, accepting `"standard"`, `"memory_mapped"`, `"streaming"`, or `"extreme"`. `max_memory_gb` is an optional float, the RAM limit in gigabytes used with `"extreme"` mode. `device` is an optional string or `torch.device`, specifying the target device. `dtype` is an optional `torch.dtype`, the floating-point data type.


<- [Vietoris-Rips Overview](index.md)
