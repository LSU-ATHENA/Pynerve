# API reference

```python
# Python streaming
from pynerve import StreamingPersistence

sp = StreamingPersistence(
    chunk_size=1000,         # points per chunk
    max_buffered_chunks=3,   # prefetch buffer depth
    use_gpu=True,            # GPU acceleration per chunk
    max_dim=2,               # passed to compute_persistence
)

# stream_compute yields dicts with keys:
#   "pairs"         -- list of (birth, death, dim)  (return_format="diagrams")
#   "betti_numbers" -- Betti numbers                 (return_format="diagrams")
#   "betti_{d}"     -- Betti per dimension           (return_format="betti")
#   "num_features"  -- total feature count           (return_format="stats")
#   "avg_persistence" -- mean persistence            (return_format="stats")
async for result in sp.stream_compute(
    "data.h5",
    return_format="diagrams",
):
    pass

# PyTorch windowed persistence
from pynerve.nn import WindowedPH
windowed = WindowedPH(
    window_size=512,
    stride=256,
    max_dim=1,
    overlap_handling="concat",
)

# Async persistence computer (lower-level)
from pynerve import AsyncPersistenceComputer

# Streaming persistence via the core C++ API
from pynerve import update_persistence
# update_persistence(events, options) -- incremental add/remove
```

[Back to index](index.md)
