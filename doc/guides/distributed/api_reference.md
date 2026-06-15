# API reference

```python
# Distributed persistence (multi-node)
pynerve.distributed_persistence(
    points,
    max_dim=2,
    max_radius=1.0,
    # optional:
    # checkpoint_path="/path/to/checkpoints",
    # checkpoint_interval_sec=300,
)

# Single-node multi-GPU via backend selection
pynerve.compute_persistence(
    points,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)

# Shared-memory multiprocessing (single-node parallel)
from pynerve.mp_shared import (
    ParallelPH,
    ChunkedParallel,
    MapReducePH,
    compute_persistence_parallel,
)

results = compute_persistence_parallel(
    np.split(points, 4),
    n_workers=4,
    use_shared_memory=True,
)
```

<- [Distributed Computing Overview](index.md)
