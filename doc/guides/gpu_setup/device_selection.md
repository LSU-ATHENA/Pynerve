# Device selection

```python
# Default GPU (cuda:0)
result = pynerve.compute_persistence(
    points, backend=pynerve.PersistenceBackend.CUDA_HYBRID
)

# Specific GPU -- via CUDA tensor placement
points = torch.randn(10000, 3, device="cuda:2")
result = pynerve.compute_persistence(points, max_dim=2)

# CuPy array on specific device
import cupy as cp
points_cp = cp.asarray(points_np)
result = pynerve.compute_persistence(points_cp, max_dim=2)
```

## CUDA_VISIBLE_DEVICES

Standard CUDA device masking works:

```bash
CUDA_VISIBLE_DEVICES=2,3 python my_script.py
# Pynerve sees only devices 2 and 3, mapped as cuda:0 and cuda:1
```


<- [Back to GPU Acceleration index](index.md)
