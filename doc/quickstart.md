# Quickstart

Get persistent homology running in seconds.

## Basic usage

```python
import pynerve
import numpy as np

rng = np.random.default_rng(42)
theta = rng.uniform(0, 2 * np.pi, 500)
points = np.column_stack([np.cos(theta), np.sin(theta)])
points += rng.normal(0, 0.05, points.shape)

result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)
```

The result is a `PersistenceResult` object:

- `result.pairs` -- list of `(birth, death, dimension)` tuples
- `result.betti_numbers` -- list of Betti numbers per dimension
- `result.num_pairs` -- total pair count
- `result.max_dim` -- homology dimension computed
- `result.max_radius` -- filtration cutoff used

```python
pairs = result.pairs
for birth, death, dim in pairs[:5]:
    print(f"H{dim}: birth={birth:.3f}, death={death:.3f}")
```

All computations are deterministic and bitwise reproducible by default.

## Engine selection

Pynerve provides multiple persistence engines that trade off speed versus precision.

| Engine | Best for | Typical use case |
|--------|----------|-----------------|
| `AUTO` (default) | Recommended default | Most applications |
| `PH0` | Compatibility mode | Reproducing older results or small datasets |
| `PH3` | Small-to-medium datasets | General persistent homology workloads |
| `PH4` | Higher-dimensional homology | Up to dimension 4 with improved memory handling |
| `PH5` | Large datasets | Default choice for 10K--1M points |
| `PH6` | Very large problems | Million-scale datasets and sparse filtrations |

**Which engine should I use?**

| Dataset size | Recommendation |
|--------------|---------------|
| < 1,000 points | `AUTO` or `PH0` |
| 1,000--100,000 points | `AUTO` |
| > 100,000 points | `AUTO` or `PH6` |
| Need exact historical compatibility | `PH0` |

```python
from pynerve import PersistenceEngine

result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH3)
result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH6)
result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH0)
```

String names also work:

```python
result = pynerve.compute_persistence(points, max_dim=2, engine="ph3")
result = pynerve.compute_persistence(points, max_dim=2, engine="ph6")
```

## GPU acceleration

```python
result = pynerve.compute_persistence(points, max_dim=2, device="cuda")
```

Async GPU with streaming:

```python
import asyncio
from pynerve.async_api import stream_persistence

async def compute():
    async for chunk_result in stream_persistence(
        "data.h5", chunk_size=1000, use_gpu=True, max_dim=2,
    ):
        print(chunk_result.betti_numbers)

asyncio.run(compute())
```

## Persistence images

Convert a diagram to a fixed-size image for machine learning:

```python
result = pynerve.compute_persistence(points, max_dim=2)

image = pynerve.persistence_image(
    result.pairs,
    resolution=(8, 8),
    sigma=0.2,
    weight="persistence",
)
```

- `resolution`: (height, width) = (persistence axis, birth axis)
- `sigma`: standard deviation of Gaussian kernel, ~0.1--0.5 typical
- `weight`: `"persistence"` weights by (death-birth), `"uniform"` weights equally

NumPy-vectorized fast path:

```python
from pynerve.fast_ops import persistence_image, betti_curve

image = persistence_image(result.pairs_array, resolution=64, sigma=0.1)
betti = betti_curve(result.pairs_array, max_dim=3, resolution=100, max_time=2.0)
```

## PyTorch integration

```python
import torch
import pynerve.torch

points = torch.randn(4, 500, 3)
diagram = pynerve.torch.vr_persistence(points, max_dim=2, max_radius=1.0)
```

`diagram` is a `PersistenceDiagram` with:

- `.diagrams` -- tensor `[batch, max_pairs, 3]`
- `.mask` -- tensor `[batch, max_pairs]` valid-pair mask
- `.births()` -- birth times
- `.deaths()` -- death times
- `.total_persistence()` -- Lp norm of persistence values

```python
diagram = pynerve.torch.witness_persistence(landmarks, points, max_dim=1)
diagram = pynerve.torch.alpha_persistence(points[:, :, :3], max_dim=2)

d1 = pynerve.torch.diagram_wasserstein(diagram, diagram, p=2.0, q=2.0)
d2 = pynerve.torch.diagram_bottleneck(diagram, diagram)
```

## Streaming for large data

Process datasets that exceed RAM by streaming chunks from disk:

```python
from pynerve.async_api import stream_persistence
import asyncio

async def stream_example():
    async for result in stream_persistence(
        "large_dataset.h5",
        chunk_size=500,
        max_buffered_chunks=3,
        use_gpu=True,
        max_dim=2,
    ):
        print(f"Chunk betti: {result.betti_numbers}")

asyncio.run(stream_example())
```

## Distributed computation

```python
from pynerve.mp_shared import compute_persistence_parallel

batches = np.split(points, 4)
results = compute_persistence_parallel(
    batches, n_workers=4, use_shared_memory=True,
)
```

## Backend control

```python
from pynerve import PersistenceBackend

result = pynerve.compute_persistence(
    points,
    max_dim=2,
    backend=PersistenceBackend.CPU_EXACT,
    threads=8,
    error_tolerance=0.0,
)
```

## Synthetic datasets

```python
from pynerve.datasets import (
    load_sphere, load_torus, load_swiss_roll,
    load_circle, load_mobius_strip, load_klein_bottle,
)

points = load_torus(n_samples=1000, major_radius=3.0, minor_radius=1.0)
result = pynerve.compute_persistence(points, max_dim=2)
```

## FAQ

**What is the difference between PH4, PH5, and PH6?** PH4 is optimized for sparse, high-dimensional filtrations using aggressive clearing. PH5 provides balanced exact and approximate computation. PH6 is a high-precision engine designed for numerical accuracy. The `auto` engine selects the best choice for your data.

**Can I use Pynerve without a GPU?** Yes. Pynerve runs entirely on CPU and automatically detects available hardware. GPU acceleration is only used when explicitly requested.

**How do I get bitwise reproducible results?** Results are deterministic by default. Pass `seed=<value>` to any compute function to ensure reproducible output across runs. For GPU, also set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
