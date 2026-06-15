# Precision Policies

Pynerve supports configurable precision for different stages of the computation:

The `P64` policy uses double precision (64-bit) for both distance and reduction, providing maximum accuracy as the default. `P32_DISTANCE` uses float (32-bit) for distance with double for reduction, offering faster distance computation while maintaining full reduction accuracy. `P32` uses float for both distance and reduction, delivering maximum speed at the cost of reduced accuracy. `P16_DISTANCE` uses half precision (16-bit) via Tensor Cores for distance with double for reduction, combining GPU Tensor Core throughput with full reduction precision.

```python
# Use float32 for distances, float64 for reduction
import numpy as np
result = pynerve.compute_persistence(
    points.astype(np.float32), max_dim=2,
)
```


[Back to Correctness Index](index.md)
