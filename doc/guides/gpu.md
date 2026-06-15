# GPU Acceleration

Enable GPU-accelerated persistence reduction, distance computation, spectral decomposition, graph algorithms, and sheaf Laplacians with zero code changes -- pass a CUDA tensor or set `backend="CUDA_HYBRID"`.

```python
import pynerve
import torch

points = torch.randn(10000, 3, device="cuda")

# Auto: detects CUDA tensor, uses CUDA_HYBRID backend
result = pynerve.compute_persistence(points, max_dim=2)

# Explicit backend selection
result = pynerve.compute_persistence(
    points.cpu().numpy(),
    max_dim=2,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)

# GPU persistence via PyTorch module
from pynerve.nn import PersistentHomology
ph = PersistentHomology(max_dim=2, device="cuda")
```


This page has been split into subpages for easier navigation:

| Section | Description |
|---|---|
| [Available operations](gpu_setup/operations.md) | List of GPU-accelerated operations |
| [CUDA kernel inventory](gpu_setup/kernel_inventory.md) | Complete kernel file listing by subdirectory |
| [GPU cohomology reduction](gpu_setup/cohomology.md) | Warp shuffle, shared memory tree, apparent pairs, clearing, hypha scan |
| [Tensor Cores](gpu_setup/tensor_cores.md) | Mixed-precision distance, auto-padding, Tensor Core pipeline |
| [Multi-GPU](gpu_setup/multi_gpu.md) | NCCL collectives, NVLink P2P, topology detection |
| [Multi-stream execution](gpu_setup/multi_stream.md) | CUDA stream pool, synchronization |
| [Determinism](gpu_setup/determinism.md) | Compiler flags, fixed-tree reductions, RFA |
| [Occupancy analysis by GPU architecture](gpu_setup/occupancy.md) | Per-architecture tuning (Turing, Ampere, Hopper, Blackwell) |
| [Occupancy targets by kernel type](gpu_setup/occupancy_targets.md) | Target occupancy per kernel domain |
| [When GPU helps](gpu_setup/when_gpu_helps.md) | Performance decision guide |
| [Memory management](gpu_setup/memory.md) | Device memory limit, P2P exchange, unified memory, DeviceMemoryPool |
| [Device selection](gpu_setup/device_selection.md) | CUDA tensor placement, CuPy, CUDA_VISIBLE_DEVICES |
| [API reference](gpu_setup/api_reference.md) | Core function, PyTorch module, CuPy pipeline, streaming |
| [Performance tuning for GPU](gpu_setup/performance_tuning.md) | Launch config, profiler hints, precision trade-offs, launch bounds |
| [Debugging GPU computations](gpu_setup/debugging.md) | Common issues, Nsight debugging |
| [FAQ](gpu_setup/faq.md) | Supported GPUs, verification, speedup, multi-GPU, determinism |
