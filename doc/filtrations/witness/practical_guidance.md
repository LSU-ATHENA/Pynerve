# Practical guidance

### Number of landmarks

A good default is $m = 500$ for $n \leq 10^5$. For noisier data, increase
$m$ to capture more topological structure. For very clean data, $m$ can
be as low as 100.

**Heuristics:**
- $m \geq 100$ for $n = 10^4$-$10^5$.
- $m \geq 500$ for $n = 10^5$-$10^6$.
- $m \approx 0.01n$ for uniform coverage.
- $m \approx 50 \times 2^{d_{\text{intrinsic}}}$ for intrinsic dimension $d$.

### Farthest-point sampling

Farthest-point sampling is **deterministic** (tie-breaking by index) and
produces consistent landmark sets. It is the recommended default.

```python
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=2,
    method="farthest",
)
```

### Max radius

Setting `max_radius` reduces the effective number of simplices. For
high-dimensional data, a smaller radius bounds the witness complex size
and prevents combinatorial explosion.

```python
# Bound the filtration to prevent excessive simplex generation
diagram = pynerve.torch.witness_persistence(
    landmarks, witnesses,
    max_dim=2,
    max_radius=2.0,   # limits simplex count dramatically
)
```

### Batched mode

The PyTorch witness API supports batched inputs for GPU acceleration:

```python
diagram = pynerve.torch.witness_persistence(
    landmarks_batch, witnesses_batch,
    max_dim=2,
)
# diagram.diagrams: [batch, max_pairs, 3]
```

### Strong witness implementation

For applications requiring tighter approximation, Pynerve supports strong
witness via the C++ backend:

```python
# Strong witness: pass strong_witness=True (C++ backend)
from pynerve.nn.sparse_ph import compute_witness_persistence

pairs = compute_witness_persistence(
    landmarks,
    witnesses,
    max_dim=2,
    max_radius=float("inf"),
    metric="euclidean",
    strong_witness=True,  # use strong witness condition
)
```

The strong witness complex is smaller (fewer simplices) and has a better
approximation ratio (2 vs 3), but may miss features that the weak witness
captures. Strong witness is recommended when:
- Landmark density is high relative to feature size.
- Filtration size must be minimized for memory reasons.
- The noise level in the data is low.

<- [Witness Complex Overview](index.md)
