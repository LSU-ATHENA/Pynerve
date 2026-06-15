# Streaming Persistence

Pynerve supports streaming persistence computation for datasets too large to fit in memory. The streaming pipeline processes data in chunks, yielding persistence results incrementally.

## When to use streaming

- Dataset exceeds available RAM (millions of points)
- Data arrives incrementally (sensor streams, online learning)
- Interactive exploration of large point clouds
- Progressive computation where early results are useful before the full computation completes

## Streaming API

### `StreamingPersistence`

The `StreamingPersistence` class processes chunked point-cloud data from files or async iterators:

```python
from pynerve._streaming_persistence import StreamingPersistence

streamer = StreamingPersistence(
    chunk_size=1000,       # points per chunk
    max_buffered_chunks=3,  # max chunks held in memory
    use_gpu=True,           # use GPU when available
)

async for result in streamer.stream_compute(
    "large_dataset.npy",
    return_format="diagrams",  # "diagrams", "betti", or "stats"
    max_dim=2,
    max_radius=2.0,
):
    print(result)
```

### Async API (top-level)

For simpler use cases, the `pynerve.async_api` module provides convenience functions:

```python
import pynerve.async_api as nerve_async

# Single async computation
result = await nerve_async.compute_persistence_async(points, max_dim=2)

# Streaming computation over an async iterator
async for chunk in nerve_async.stream_persistence(data, chunk_size=500):
    process(chunk)
```

### Supported file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| NumPy binary | `.npy` | Full memory-mapped read |
| NumPy archive | `.npz` | Uses `"data"` key or first available array |
| HDF5 | `.h5` / `.hdf5` | Requires `h5py`; uses `"data"` dataset |

## Return formats

| Format | Description |
|--------|-------------|
| `"diagrams"` | Full persistence result with pairs, Betti numbers, diagnostics |
| `"betti"` | Betti number summary dict `{betti_0: n, betti_1: m, ...}` |
| `"stats"` | Aggregate statistics: `num_features`, `avg_persistence`, `max_persistence` |

## Chunked processing model

The streaming pipeline operates as follows:

1. **Chunking**: The input data is split into fixed-size chunks (`chunk_size` rows each)
2. **Buffering**: Up to `max_buffered_chunks` chunks are held in memory
3. **Compute**: Each chunk is processed independently through the standard persistence pipeline
4. **Yield**: Results are yielded as they complete, preserving iteration order

## Memory considerations

- Set `chunk_size` based on available RAM and point dimensionality
- For GPU streaming, ensure each chunk fits in GPU memory
- The `max_buffered_chunks` parameter controls the trade-off between throughput and memory usage
- File-backed streaming (NPY/NPZ) uses NumPy's memory-mapped I/O for efficient disk access

## Example: Processing a large NPY file

```python
import numpy as np
from pynerve._streaming_persistence import StreamingPersistence

# Generate or load a large dataset
data = np.random.randn(1_000_000, 3)
np.save("large_cloud.npy", data)

# Stream through it in chunks
async def run():
    streamer = StreamingPersistence(chunk_size=5000, use_gpu=False)
    async for betti in streamer.stream_compute(
        "large_cloud.npy",
        return_format="betti",
        max_dim=2,
        max_radius=1.0,
    ):
        print(f"Chunk Betti numbers: {betti}")

import asyncio
asyncio.run(run())
```

## Related topics

- [Performance Guide](../guides/performance.md) -- scaling and optimization details
- [GPU Guide](../cuda/cuda.md) -- GPU acceleration for persistence
- [API Reference](../reference/api_python.md) -- full API documentation
