# Precomputed distance matrix

Pass an (n x n) array and specify `metric="precomputed"`:

```python
import numpy as np
import pynerve

# Custom distance matrix computed externally
n = 1000
distances = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(i + 1, n):
        d = my_expensive_distance(raw_data[i], raw_data[j])
        distances[i, j] = d
        distances[j, i] = d

result = pynerve.compute_persistence(
    distances,
    max_dim=2,
    max_radius=2.0,
    metric="precomputed",
)
```

### Requirements

The matrix must be a square of shape `(n, n)`. It should be symmetric (`D[i,j] == D[j,i]`), though this is not enforced. The diagonal should ideally be zero (`D[i,i] == 0`). The dtype should be `float64` (converted if needed). Infinite values are allowed and are ignored during filtration. Memory usage is O(n^2) -- for n=10^5 this requires approximately tens of gigabytes.

### Precomputed matrix format

The precomputed matrix is a dense square NumPy array. C++ accesses via `BufferView<const double>` with stride `n`. The distance for edge (i, j) is read as `D[i * n + j]`. The matrix is assumed symmetric -- only the upper triangle is needed in principle, but the full matrix is read.

### Sparse precomputed

For n > 10,000, pass a sparse distance matrix:

```python
from scipy.sparse import csr_matrix

# Sparse precomputed: only store distances < threshold
rows, cols, vals = [], [], []
for i in range(n):
    for j in range(i + 1, n):
        d = compute_distance(i, j)
        if d < max_radius:  # only store relevant distances
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([d, d])

sparse_dist = csr_matrix((vals, (rows, cols)), shape=(n, n))
result = pynerve.compute_persistence(
    sparse_dist,
    max_dim=2,
    metric="precomputed",
)
```

Sparse precomputed uses the same CSR format as the sparse boundary matrix. Only distances below `max_radius` need to be stored. During VR construction, edges are directly read from the CSR matrix, skipping the O(n^2) distance computation entirely.

[Back to index](index.md)
