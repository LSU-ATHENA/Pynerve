# Failure modes

### Combinatorial explosion

The number of $k$-simplices in VR grows as $O(n^{k+1})$. At dim 4 with
10K points, the boundary matrix can exceed hundreds of gigabytes. Symptoms:

- Memory allocation failure ("Cannot allocate memory").
- Extreme slowdown due to swapping (thrashing).
- Hours-long reduction time with no progress.

Mitigations:
- Use `max_dim=2` unless $H^2$ or higher is required.
- Switch to [sparse VR](../sparse_vr.md) with `sparse=True`.
- Limit the filtration with a smaller `max_radius`.
- Use `memory_mode="streaming"` to trade memory for time.

### Near-zero distances

Duplicate points or very close points produce near-zero birth times:

```python
# Two identical points at (0, 0, 0)
points = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
result = pynerve.compute_persistence(points, max_dim=2)
# Birth times: approximately [0, 0, 0, 1.0, ...]
```

This causes:
- Numerical instability in matrix reduction (ties in pivot selection).
- Unstable persistence pairs (splitting of near-equal births).
- Non-deterministic output across platforms.

Mitigations:
- Remove duplicates with `np.unique(np.round(points, decimals=12), axis=0)`.
- Use `error_tolerance=1e-9` to treat near-simultaneous births as equal.
- Use a higher precision engine (`ph6`).

### Memory pressure

For $n > 5000$ at dim 3, the edge list alone requires $O(n^2)$ storage
($n^2 / 2$ entries, approximately a couple hundred megabytes for $n = 10^4$ as 32-bit floats). The boundary
matrix adds another factor of $k$ per simplex.

Mitigations:
- Enable `memory_mode="streaming"` or `"memory_mapped"`.
- Reduce `max_dim` to 2.
- Use a landmark-based approximation.

### Metric violations

A precomputed distance matrix that violates the triangle inequality does
not affect VR construction (VR works on any metric space, including
non-metric dissimilarity matrices), but it can produce unintuitive
persistence diagrams where birth and death times appear inconsistent.

```python
import numpy as np

# Triangle inequality violation: d(a,c) > d(a,b) + d(b,c)
D = np.array([
    [0,   1,   10],
    [1,   0,   1],
    [10,  1,   0 ]
])

# VR will still produce a diagram, but the geometric meaning is unclear
```

### Numerical overflow

For points with very large coordinates (e.g., astronomical data), squared
distance computations can overflow 32-bit float:

```python
points = np.array([[1e10, 0], [0, 1e10]], dtype=np.float64)
result = pynerve.compute_persistence(points, max_dim=2)  # Use float64 input
```

Use `float64` (double precision) for large-coordinate data. The SIMD
distance kernel handles both float32 and float64.


<- [Vietoris-Rips Overview](index.md)
