# API reference

```python
# Core function -- GPU enabled via backend or CUDA tensor
pynerve.compute_persistence(
    points,
    backend=PersistenceBackend.CUDA_HYBRID,  # or pass CUDA tensor (auto)
    max_dim=2,
    max_radius=1.0,
    error_tolerance=1e-12,
)

# PyTorch module
from pynerve.nn import PersistentHomology
ph = PersistentHomology(max_dim=2, device="cuda")

# CuPy pipeline
from pynerve.cupy_ops import CuPyPersistence, compute_diagrams_cupy
diagrams = compute_diagrams_cupy(points_cupy, max_dim=2)

# Async GPU with streaming
from pynerve import StreamingPersistence
sp = StreamingPersistence(chunk_size=500, use_gpu=True, max_dim=2)
# see doc/guides/streaming.md
```


<- [Back to GPU Acceleration index](index.md)
