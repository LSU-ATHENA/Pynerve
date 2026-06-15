# Complexity and memory analysis

### Time complexity

The witness complex involves several phases with different time costs. Landmark selection using farthest-point sampling costs $O(n \cdot m)$ and is the dominant phase for large $m$. Distance matrix computation costs $O(n \cdot m \cdot d)$ where $d$ is the ambient dimension. Landmark-distance sorting costs $O(n \cdot m \log m)$ for per-witness nearest-landmark ordering. Simplex enumeration costs $O(m^{k+1})$ on landmarks only. Filtration sorting costs $O(m^{k+1} \log m)$ for birth radius sorting. Matrix reduction costs $O(m^{k \cdot \omega})$ where $\omega \approx 2$ with clearing.

### Memory layout

- **Distance matrix.** An $n \times m$ float32/64 array. For $n = 10^5$ and $m = 500$, this uses a couple hundred megabytes with float32 or a few hundred megabytes with float64.
- **Nearest-landmark indices.** An $n \times (k+1)$ int32 array caching the nearest landmark ordering.
- **Boundary matrix.** $O(m^{k+1})$ columns. For $m = 500$ and $k = 2$, this amounts to around $2 \times 10^7$ entries in CSR format, using several hundred megabytes.

### Memory optimization

```python
from pynerve.nn._building_blocks_persistence import WitnessComplexPersistence

# Reduce m to save memory (at cost of approximation quality)
wc = WitnessComplexPersistence(
    n_landmarks=200,
    max_dim=2,
    method="farthest",
)

# Reduce max_dim to save memory
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=1,              # only H0 and H1
    method="farthest",
)
```

<- [Witness Complex Overview](index.md)
