# Pynerve

> Fast TDA library.

Pynerve computes topological features (persistent homology) from point clouds, distance matrices, and scalar fields. It scales from interactive exploration on a laptop to billion-edge pipelines on HPC clusters.

```python
import pynerve
import numpy as np

points = np.random.randn(1000, 3)
result = pynerve.compute_persistence(points, max_dim=2, max_radius=1.0)
print(result.pairs)
```

!!! note "Version 1.0.0 — Windows native support"
    Windows native support via the `nerve::sys` platform abstraction layer (MSVC, Clang-cl). Full platform support: Linux, macOS, and Windows.

## When to use Pynerve

Pynerve is the right choice when you need to:

- **Compute persistent homology** from point clouds, distance matrices, or scalar fields
- **Scale to large datasets** with streaming, distributed, and GPU-accelerated pipelines
- **Integrate topology into ML pipelines** with differentiable persistence and PyTorch autograd
- **Guarantee reproducibility** with bitwise deterministic computation by default

Pynerve is not a general-purpose topological data analysis library. It focuses on fast persistent homology computation and its integration with machine learning.

## Installation

```bash
pip install pynerve
```

From source (requires C++20 compiler, CMake >=3.26, and CUDA Toolkit >=12.4 for GPU support):

```bash
git clone https://github.com/LSU-ATHENA/Pynerve.git
cd Pynerve
pip install -e .
```

## Quick start

```python
import pynerve
import numpy as np

from pynerve.datasets import load_circle
points, labels = load_circle(n_samples=500, noise=0.05)

result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)

pairs = result.pairs
betti = result.betti_numbers
print(f"Found {len(pairs)} persistence pairs")
print(f"Betti numbers: {betti}")
```

See the [Quickstart guide](quickstart.md) and [API Reference](reference/api_python.md) for more.

## Key concepts

- **Persistent homology** tracks topological features (connected components, loops, voids) across a range of spatial scales defined by a filtration.
- **Filtration** is a nested sequence of simplicial complexes built from input data (e.g., Vietoris-Rips, Alpha, or Witness complexes).
- **Persistence diagram** records the birth and death scales of each topological feature. Features with long persistence (death - birth) are considered signal; short-lived features are typically noise.
- **Determinism** means the same input always produces the same output, down to the last bit, across runs and platforms.

## Capabilities

| Area | What Pynerve provides |
|------|-----------------------|
| **Performance** | SIMD-optimized distance kernels (AVX-512, AVX2, SSE4.1) with runtime dispatch, cache-aware boundary matrix in CSR format |
| **GPU** | 92 CUDA kernels covering matrix reduction, cohomology, clearing, apparent pairs, and tensor cores. CuPy-backed GPU pipeline |
| **Streaming** | Tile-streaming PH, sliding-window persistence, async iterator, memory-bounded pipelines |
| **Distributed** | MPI with CUDA-aware communication (NCCL, NVSHMEM), multi-GPU work distribution |
| **Determinism** | Bitwise reproducibility by default through seeded PRNG and deterministic seed propagation |
| **PyTorch** | Native `pynerve.torch` sub-package with autograd, batched `PersistenceDiagram`, Wasserstein and bottleneck distances, Mapper |
| **Differentiable** | Analytical subgradients through birth and death indices, learned filtration layers, topology-aware neural network layers |

## Platform support

| Platform | Status |
| -------- | ------ |
| Linux    | Full support (primary target) |
| macOS    | Full support |
| Windows  | Native support via `nerve::sys` abstraction layer (MSVC, Clang-cl) |

GPU acceleration requires an NVIDIA GPU with compute capability 7.5 or higher.

## Module overview

The core function is `pynerve.compute_persistence`, which auto-selects the best engine from PH0, PH3--PH6. Individual engines can be selected explicitly:

```python
from pynerve import PersistenceEngine

result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH3)
result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH6)
result = pynerve.compute_persistence(points, max_dim=2, engine=PersistenceEngine.PH0)
```

The PyTorch sub-package provides differentiable persistence:

```python
import pynerve.torch

diagram = pynerve.torch.vr_persistence(points, max_dim=1, max_radius=2.0)
loss = diagram.total_persistence()
loss.backward()
```

Neural network modules in `pynerve.nn` include `PersistentHomology` as an `nn.Module` wrapper, `SparsePH` for landmark-based approximate persistence, and `WindowedPH` for sliding-window persistence.

## Documentation structure

| Section | Contents |
|---------|----------|
| [Quickstart](quickstart.md) | Get running in seconds |
| [API Reference](reference/api_python.md) | Full API documentation |
| [Algebra](algebra/algebra.md) | Simplicial complexes, boundary matrices, chain complexes |
| [Algorithms](algorithms/algorithms.md) | Distance computation, Mapper, kernel methods, vectorization |
| [Filtrations](filtrations/vietoris_rips.md) | Vietoris-Rips, Alpha, Witness, Sparse VR, Level set |
| [Persistence](persistence/standard_reduction.md) | Standard and cohomology reduction, PH4/PH5/PH6 engines |
| [CUDA](cuda/cuda.md) | GPU kernels, streams, graphs, determinism |
| [PyTorch](torch/torch.md) | Autograd, persistence diagram, simplex tree, float8 |
| [Neural Networks](nn/nn.md) | Persistent homology layers, diagram convolution |
| [Guides](guides/performance.md) | Performance tuning, GPU setup, distributed, streaming, sparse data |
| [Memory](memory/memory.md) | Pool allocator, NUMA-aware layout, memory-mapped persistence |
| [Reference](reference/architecture.md) | Architecture, C++/Python API, correctness, decision guide |
| [Streaming](streaming/streaming.md) | Lock-free streaming, windowed persistence |
| [GPU](gpu/gpu.md) | GPU tuning, memory management |
| [Core](core/core.md) | Thread pools, RNG, SIMD operations |
| [Batching](batching/batching.md) | Batched persistence computation |
| [Caching](cache/cache.md) | Result caching |
| [Compression](compression/compression.md) | Data compression for persistence |
| [Determinism](determinism/determinism.md) | Bitwise reproducibility guarantees |
| [Validation](validation/validation.md) | Contracts, benchmarks, determinism checks |
| [Serialization](serialization/serialization.md) | FlatBuffers, Arrow, version manager |
| [IO](io/io.md) | File formats, async IO, memory-mapped, NPY |
| [Metrics](metrics/metrics.md) | Bottleneck, Wasserstein, diagram distances |
| [Optimization](optimization/optimization.md) | SIMD, GPU, streaming PH, compact summary |
| [Sheaf](sheaf/sheaf.md) | Sheaf Laplacian, learning, morphisms |
| [Spectral](spectral/spectral.md) | Laplacians, Dirac operator, eigensolver |
| [Graphs](graphs/graphs.md) | Graph algorithms, GNN, GPU, MPI |
| [Encoders](encoders/encoders.md) | ML encoders, SIMD, GPU |
| [Differentiable](differentiable/differentiable.md) | Differentiable persistence layers |
| [ML](ml/ml.md) | Machine learning integration |
| [Regularization](regularization/regularization.md) | Topology-aware regularization |
| [Probabilistic](probabilistic/probabilistic.md) | Randomized algorithms |
| [Approximation](approximation/approximation.md) | Approximation methods |
| [Features](features/features.md) | Feature extraction |
| [Instrumentation](instrumentation/instrumentation.md) | Metrics, diagnostics |
| [DMT](dmt/dmt.md) | Discrete Morse Theory |
| [Error Handling](errors/errors.md) | Error taxonomy, error propagation |
| [Precision](precision/precision.md) | Precision policies |

## Citation

If you use Pynerve in academic work, please cite:

```bibtex
@software{pynerve2026,
  title = {Pynerve: Fast Persistent Homology for Large Sparse Filtrations},
  year = {2026},
  url = {https://github.com/LSU-ATHENA/Pynerve}
}
```

## License

MIT. See `LICENSE`.
