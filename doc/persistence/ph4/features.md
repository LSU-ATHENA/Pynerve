# Features

PH4 supports a wide range of features. It provides both standard reduction and cohomology reduction (via `compute_persistence_cohomology`). Clearing optimization and column compression are enabled automatically. Approximate computation is available via `PersistenceMode.APPROX`, and budgeted computation is memory-limit aware. Witness sampling is supported with landmark and max-persistence strategies, along with a stability certificate from numerical residual tracking. Deterministic mode ensures bitwise reproducibility. Parallel CPU execution uses OpenMP threading, and GPU acceleration is available via `PersistenceBackend.CUDA_HYBRID`. Streaming is supported through `StreamingPersistence`.

### Algorithm Selection

PH4 automatically selects the reduction strategy based on problem size:

- **Small dense** (n < 10^4): standard reduction with clearing
- **Large sparse** (n >= 10^4): cohomology-style reduction
- **Approximate**: witness sampling + landmark reduction
- **Budgeted**: falls back to compact summary when memory exceeds limit

The selection heuristic evaluates:

```python
def select_reduction_strategy(n_simplices, estimated_density, max_dim):
    # Small complexes: standard reduction (lower overhead)
    if n_simplices < 10_000:
        return "standard"

    # Dense complexes: standard reduction may be competitive
    if estimated_density > 0.5 and max_dim <= 2:
        return "standard"

    # Default: cohomology for larger sparse complexes
    return "cohomology"
```

The density estimate is computed as:

    density = actual_edges / possible_edges (for Vietoris-Rips)
    density = actual_cells / total_cells (for cubical complexes)

### Determinism

PH4 is bitwise reproducible by default. Every run with the same inputs produces identical persistence pairs, down to the last bit. This is enforced by:

- Fixed seed propagation through the `DeterminismContract`
- Deterministic reduction tree (no thread-dependent scheduling)
- No floating-point atomics in the GPU path

```python
r1 = pynerve.compute_persistence_up_to_dim_4(points)
r2 = pynerve.compute_persistence_up_to_dim_4(points)
assert r1.pairs == r2.pairs  # always passes
```

**Internal mechanism**: The determinism contract assigns each thread a fixed set of columns to reduce, and the column addition order is deterministic. OpenMP parallel loops use `schedule(static)` rather than `schedule(dynamic)` to ensure thread-to-column assignment is consistent across runs.

### Approximate Mode

For very large datasets, PH4 can compute approximate persistence using witness sampling:

```python
opts = PersistenceOptions(mode=PersistenceMode.APPROX)

result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_dim=2)
```

**How it works**:

1. **Landmark selection**: Choose m landmarks from the N input points using a sampling strategy (random, max-min, or sequential).
2. **Witness complex construction**: Each non-landmark point (witness) votes for the simplices formed by nearby landmarks. The resulting complex is much smaller than the full Vietoris-Rips complex.
3. **Persistence computation**: Run standard or cohomology reduction on the witness complex.
4. **Extrapolation**: Map persistence pairs from the witness complex back to the original data.

**Tuning the approximation**:

The engine selects landmarks internally using farthest-point sampling.
The `error_tolerance` parameter controls approximation quality:

```python
opts = PersistenceOptions(mode=PersistenceMode.APPROX, error_tolerance=0.01)

result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_dim=2)
```

**Quality-quality tradeoff**:

The approximation quality depends on the number of landmarks used. With 50 landmarks, the complex size is around 10^4, yielding a 50-100x speedup over exact computation and a normalized bottleneck distance of 0.05-0.15. With 100 landmarks, the complex is around 10^5, speedup drops to 20-50x, and quality improves to 0.02-0.08. At 200 landmarks the complex reaches roughly 10^6 simplices with a 5-20x speedup and quality of 0.01-0.03. At 500 landmarks the complex is roughly 10^6, speedup is 2-5x, and quality improves to 0.005-0.01.

Quality measured as normalized bottleneck distance between exact and approximate barcodes. Higher landmark counts improve quality at the cost of computation time.

### Budgeted Computation

PH4 can operate within a fixed memory budget:

```python
# Budgeted computation is available through PH5/PH6 engines
from pynerve import PH5PH6Config

config = PH5PH6Config()
result = pynerve.compute_persistence_up_to_dim_5(points, max_dim=2)
```

When the estimated memory exceeds the budget:

1. Switch to sparse column representation if using dense.
2. Enable more aggressive compression.
3. Fall back to witness sampling (approximate mode).
4. If still over budget, raise `MemoryBudgetExceeded`.

The memory estimator runs before computation:

```python
def estimate_memory(n_simplices, max_dim, representation):
    # Column storage
    cols_bytes = n_simplices * avg_column_size * 8  # uint64_t entries

    # Pivot table
    pivot_bytes = n_simplices * 4  # int32_t

    # Coface index (if cohomology)
    coface_bytes = n_simplices * avg_cofaces * 8 if use_cohomology else 0

    # Working memory for parallel threads
    thread_bytes = threads * column_buffer_size

    total = cols_bytes + pivot_bytes + coface_bytes + thread_bytes
    return total
```

### Streaming Mode

For datasets that do not fit in memory, PH4 supports streaming persistence:

```python
from pynerve import StreamingPersistence

# Stream simplices from a generator or file
stream = StreamingPersistence(max_dim=2, batch_size=10000)
for batch in simplex_generator():
    stream.insert_batch(batch)

result = stream.compute()
```

The streaming engine:
- Buffers simplices in memory up to `batch_size`.
- Periodically compacts the intermediate matrix (removing paired columns).
- Supports restart from checkpoint via serialized state.

**When streaming helps**:
- Filtration larger than available RAM
- Real-time data streams (incremental persistence)
- Distributed computation where each node processes a portion of the filtration

Back to [PH4 Engine Overview](index.md)
