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

## Sections

- [Available operations](operations.md)
- [CUDA kernel inventory](kernel_inventory.md)
- [GPU cohomology reduction](cohomology.md)
- [Tensor Cores](tensor_cores.md)
- [Multi-GPU](multi_gpu.md)
- [Multi-stream execution](multi_stream.md)
- [Determinism](determinism.md)
- [Occupancy analysis by GPU architecture](occupancy.md)
- [Occupancy targets by kernel type](occupancy_targets.md)
- [When GPU helps](when_gpu_helps.md)
- [Memory management](memory.md)
- [Device selection](device_selection.md)
- [API reference](api_reference.md)
- [Performance tuning for GPU](performance_tuning.md)
- [Debugging GPU computations](debugging.md)
- [FAQ](faq.md)
