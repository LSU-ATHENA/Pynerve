# Windowed persistence

Sliding-window persistence tracks topological features across sequential point-cloud frames:

```python
from pynerve.nn import WindowedPH
import torch

# Time-series of point clouds: (T, n, dim)
stream = torch.randn(1000, 500, 3)

windowed = WindowedPH(
    window_size=512,
    stride=256,
    max_dim=1,
    max_radius=2.0,
    overlap_handling="concat",  # "mean" | "max" | "concat"
)

# Returns (num_windows, feature_dim)
features = windowed(stream)
```

### Window overlap handling

There are three overlap handling modes. `concat` concatenates consecutive window outputs, suitable for ML feature vectors. `mean` averages overlapping window outputs for smooth tracking. `max` max-pools overlapping window outputs for conservative estimates.

### Sliding window protocol

```
Input: stream of T frames, each with n points, dim dimensions
Output: feature vector per window

1. Initialize:
   window_start = 0
   window_end = min(window_size, T)
   features = []

2. For each window:
   a. Extract sub-sequence: frames[window_start : window_end]
   b. Compute persistence diagram for each frame (or batch)
   c. Aggregate window features:
      - Each frame -> persistence diagram
      - Diagram -> vectorization (persistence image, Betti curve, etc.)
      - Per-frame vectors aggregated via overlap_handling mode
   d. Append to features list
   e. Advance: window_start += stride, window_end = window_start + window_size

3. Return features: (num_windows, feature_dim)
```

### Memory budget

Each window uses a **fixed memory budget** -- no per-window accumulation. The budget is the maximum of:
- Distance matrix: `window_size * window_size * 4` bytes (float32)
- Boundary matrix: `O(window_size^3)` entries compressed to CSR

For window_size=1024 and max_dim=2, peak memory is ~tens of megabytes per window.

[Back to index](index.md)
