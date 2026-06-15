# Landmark selection details

[Back to index](index.md)

### Greedy permutation algorithm

The greedy permutation builds an **ordered** set of landmarks where the
$i$-th landmark is the point farthest from the first $i-1$ landmarks:

```
Input: X (n points), m (landmark count, optional)
Output: L (ordered landmarks), rad (covering radii)

1. Pick initial point l_0 arbitrarily (or by max norm).
2. For i = 1, ..., m:
   a. For each point x in X, maintain dist[x] = min_{j < i} d(x, l_j).
   b. Pick l_i = argmax_x dist[x].
   c. Set rad[i] = dist[l_i].
3. Return L = [l_0, ..., l_{m-1}], rad = [rad[0], ..., rad[m-1]].
```

The radii **rad** are decreasing: $\text{rad}[i] \geq \text{rad}[i+1]$.
The $i$-th landmark is at most $\text{rad}[i]$ from some earlier landmark,
so the first $i$ landmarks form a $\text{rad}[i]$-net of $X$.

### Stopping criteria

The greedy permutation can stop when:
- A fixed number $m$ of landmarks is selected.
- The covering radius drops below a threshold ($\text{rad}[i] < \varepsilon_{\min}$).
- A memory or time budget is exhausted.

```python
from pynerve.nn._building_blocks_persistence import SparseRipsPersistence

# Fixed landmark count
sparse = SparseRipsPersistence(
    sparse_parameter=0.3,
    max_dim=2,
    n_landmarks=500,           # fixed number of landmarks
    algorithm="greedy_permutation",
)

# Or let the algorithm choose based on sparse_parameter
sparse = SparseRipsPersistence(
    sparse_parameter=0.3,
    max_dim=2,
    algorithm="greedy_permutation",
)
# n_landmarks is chosen automatically: m ~ sparse_parameter * n
```
