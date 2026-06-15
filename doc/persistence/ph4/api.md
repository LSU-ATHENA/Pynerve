# API Reference

### compute_persistence_ph4

```python
pynerve.compute_persistence_ph4(
    points,
    options=None,
    *,
    max_dim=2,
    max_radius=None,
    mode=None,
    backend=None,
    threads=None,
    error_tolerance=None,
    dtype=None,
    max_radius_cap=None,
) -> PersistenceResult
```

### PersistenceOptions

The `mode` field accepts a `PersistenceMode` value of `EXACT` or `APPROX`, defaulting to `EXACT`. The `backend` field accepts a `PersistenceBackend` value of `CPU_EXACT`, `CPU_ADAPTIVE_ACCELERATION`, or `CUDA_HYBRID`, defaulting to `CPU_ADAPTIVE_ACCELERATION`. The `max_dim` field is an integer defaulting to 2, controlling the maximum homology dimension. The `threads` field is an integer defaulting to 0 (auto), specifying the number of CPU threads. The `error_tolerance` field is a float defaulting to 0.0, controlling approximation tolerance.

### PersistenceMode

`EXACT` computes exact persistence pairs (the default). `APPROX` uses witness sampling with landmark reduction. `STREAMING` enables streaming mode and requires `StreamingPersistence`.

### PersistenceBackend

`CPU_EXACT` runs on CPU with exact scalar operations and no SIMD. `CPU_ADAPTIVE_ACCELERATION` runs on CPU with runtime SIMD dispatch (the default). `CUDA_HYBRID` provides GPU acceleration for column operations while the CPU handles orchestration.

### Parameter Details

**max_dim**: Controls which dimensions of homology are computed. Higher values increase computation time and memory usage nonlinearly. At max_dim=0 (connected components only), the approximate relative cost is 1x. At max_dim=1 (loops and cycles), it is 10x. At max_dim=2 (voids and cavities), it is 100x. At max_dim=3 (3D holes, rarely needed), it is 1000x.

Each additional dimension adds roughly one order of magnitude more simplices in a Vietoris-Rips complex.

**max_radius**: Cutoff for Vietoris-Rips construction. Lower values produce smaller complexes. With max_radius=0.1 and N=1000 points, the approximate complex size is 10^4 simplices, suitable for a quick preview. With 0.3, the complex reaches 10^5 simplices for moderate detail. With 0.5, it reaches 10^6 simplices for full topological analysis. With infinity, the complex exceeds 10^7 simplices for a complete Rips complex, which is rarely feasible.

**error_tolerance**: Controls floating-point tie-breaking. When two simplices have nearly equal filtration values (within tolerance), they are treated as simultaneous:

```python
# Default: strict comparison
result = pynerve.compute_persistence_ph4(points, error_tolerance=0.0)

# Relaxed: treat values within 1e-8 as equal
result = pynerve.compute_persistence_ph4(points, error_tolerance=1e-8)
```

**Determinism guarantee levels**:

PH4 guarantees bitwise reproducibility by default. The same input always produces the same pairs. OpenMP loops use `schedule(static)` to ensure thread-to-column assignment is consistent.

### Return Value

```python
PersistenceResult(
    pairs=[(birth, death, dim), ...],      # Persistence pairs
    betti_numbers=[b0, b1, ...],            # Betti numbers
    max_dim=2,                              # Maximum homology dimension
    max_radius=0.5,                         # Filtration cutoff used
    diagnostics={                           # Optional diagnostic info
        "computation_time_ms": 123.4,
        "peak_memory_bytes": 524288000,
        "algorithm": "cohomology",
    }
)
```

Back to [PH4 Engine Overview](index.md)
