# Performance tuning for sparse mode

### Landmark selection strategy comparison

**Random** selection takes 1 ms for n=100K, offers fair coverage quality, is not deterministic (seed-dependent), and may miss sparse regions in the worst case. **Maxmin (farthest point)** takes 500 ms, offers excellent coverage quality, is deterministic, and has O(k * n) distance worst-case complexity. **Greedy permutation** takes 1 second, offers excellent coverage quality with a theoretical guarantee, is deterministic, and has O(k * n) distance plus sorting worst-case complexity. **K-means centers** takes 2 seconds, offers good coverage for clustered data, is not deterministic (seed-dependent), and has O(k * n * iter) worst-case complexity.

### Trade-off: landmark count vs quality

At a landmark ratio of 0.1%, memory usage is hundreds of kilobytes, processing time for n=100K and d=3 is 200 ms, and the bottleneck error bound is epsilon ~ 0.5. At 0.5%, memory is a few megabytes, time is 500 ms, and error bound is epsilon ~ 0.2. At 1%, memory is a few megabytes, time is 1 second, and error bound is epsilon ~ 0.1. At 5%, memory is tens of megabytes, time is 5 seconds, and error bound is epsilon ~ 0.02. At 10%, memory is tens of megabytes, time is 10 seconds, and error bound is epsilon ~ 0.01.

### Sparse vs dense crossover

The crossover point where sparse mode becomes faster than dense VR depends on:

1. **n**: Dense is O(n^2), sparse is O(k^2). Crossover at n ~ 5,000 for dim=3.
2. **dim**: Higher dim favors dense (SIMD distance benefits). Crossover at n ~ 10,000 for dim=10.
3. **max_dim**: Higher max_dim favors sparse (boundary matrix is O(n^{d+1}) vs O(k^{d+1})).
4. **max_radius**: Lower max_radius favors dense (fewer edges).

```python
# Empirical crossover detection
# Pynerve automatically switches to sparse when n > 5,000 and max_dim > 1
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,  # explicit sparse selection
)
```

### Memory-efficient landmark operations

When n is very large ( > 10^6 ), landmark operations must avoid O(n) memory:

```python
# Streaming landmark selection (O(k) memory, not O(n))
from pynerve.nn import farthest_point_sampling_streaming

# Requires only k * dim memory for landmarks
# Iterates through data one batch at a time
landmarks = farthest_point_sampling_streaming(
    data_iterator, n_landmarks=500, max_batches=100
)
```

### Sparse VR with edge collapse

Combining edge collapse with sparse VR gives maximum compression:

Full VR at 10^12 simplices requires a few terabytes of memory and is exact but not practically feasible. Sparse VR (1%) at 10^8 simplices uses a few megabytes, takes 1 second, and is epsilon-interleaved. Sparse VR with edge collapse at 10^6 simplices uses tens of kilobytes, takes 200 ms, and is epsilon-interleaved. Sparse VR with collapse and clearing at 10^5 simplices uses a few kilobytes, takes 100 ms, and is epsilon-interleaved.

### Sparse with precomputed distance matrix

Sparse VR works with precomputed distance matrices. The landmark selection operates on the full n x n matrix:

```python
# Precomputed distance matrix with sparse VR
dist_mat = my_compute_distance(points)
result = pynerve.compute_persistence(
    dist_mat,
    max_dim=2,
    metric="precomputed",
    mode=pynerve.PersistenceMode.APPROX,
)
```

Landmark selection on precomputed distances uses random sampling (no distance computation needed).

### Sparse with GPU acceleration

```python
# Sparse VR on GPU: each landmark batch processed by GPU
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)
```

GPU acceleration for sparse VR:
1. Landmark selection on CPU (O(k*n) farthest-point)
2. Landmark distance matrix on GPU (Tensor Core accelerated)
3. Edge extraction on GPU
4. Column reduction on GPU (sparse columns)
5. Pair extraction on GPU

The GPU sparse path handles up to 100K landmarks and 10M+ simplices.

### Approximate mode tuning checklist

```python
# Aggressive sparsification (max speed, 100K+ points)
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=0.5,
)

# Moderate sparsification (balanced, 50K-100K points)
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=0.01,
)

# Mild sparsification (near-exact, 10K-50K points)
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=1e-6,
)
```

### Choosing landmark selection strategy

```python
from pynerve.nn import farthest_point_sampling

# Random: fastest, good for uniform data
landmarks_random = farthest_point_sampling(
    points, n_landmarks=500, method="random"
)

# Maxmin: best coverage, deterministic
landmarks_maxmin = farthest_point_sampling(
    points, n_landmarks=500, method="maxmin"
)

# Greedy permutation: theoretical guarantee
landmarks_greedy = farthest_point_sampling(
    points, n_landmarks=500, method="greedy"
)
```

**Random** selection has O(n) complexity, is seed-dependent for repeatability, offers fair coverage, and is best for dense, uniform data. **Maxmin** has O(k*n) complexity, is repeatable, offers excellent coverage, and is best for sparse, non-uniform data. **Greedy** permutation has O(k*n + k log k) complexity, is repeatable, offers excellent coverage with a guarantee, and is best for data of unknown structure.

### When NOT to use sparse

1. **Data is already small** (n < 1,000): Landmark selection overhead > reduction savings
2. **Data is dense and low-dim** (dim <= 2, uniform): Full VR fits in memory and is faster
3. **Exact results required**: Sparse VR is always approximate
4. **Landmark selection is expensive**: For very high dim ( > 100 ), farthest-point sampling is O(k*n*dim)
5. **Persistence of small features matters**: Features with death-birth < epsilon are lost


### Sparse persistence for time-varying data

Sparse VR with streaming enables analysis of large time-varying point clouds:

```python
# Streaming sparse persistence
sp = StreamingPersistence(
    chunk_size=5000,
    use_gpu=True,
    max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=0.01,
)

async for result in sp.stream_compute("timeseries.h5"):
    betti = result.betti_numbers
    # Track Betti numbers over time
```

Each chunk independently selects landmarks and builds a sparse VR. Results are merged for cross-window feature tracking.

### Sparse reduction algorithm

The sparse reduction algorithm in `pynerve.fast_ops.column_reduction_sparse`:

```
Input: CSR boundary matrix with columns {c_0, ..., c_{m-1}}
Output: Reduced matrix with paired columns

1. Initialize pivot map: pivot[col] = lowest_nonzero_index(col)

2. For each column c in 0..m-1 (left to right for homology):
   a. While pivot[c] != -1 AND pivot[c] is paired:
      - Let p = pivot[c]
      - Let paired_col = pair_of[pivot[c]]
      - XOR operation: c = symmetric_difference(c, paired_col)
        (O(nnz(c) + nnz(paired_col)) merge of sorted index lists)
      - Update pivot[c] = lowest_nonzero_index(c)
   b. If pivot[c] != -1:
      - Record pair: (pivot[c], c)
      - Mark pivot[c] as paired

3. For Z2 coefficients: XOR is symmetric difference.
   For Zp (p > 2): multiply paired column by (coeff[c] / coeff[pivot]) mod p.

Complexity: O(m * avg_nnz_per_column) in practice.
Clearing optimization reduces m by eliminating columns paired in lower dimensions.
```

### Error bounds for sparse approximation

The bottleneck distance between the sparse VR diagram and the full VR diagram is bounded by:

```
d_B(Dgm_sparse, Dgm_full) <= 2 * epsilon * max_radius
```

where epsilon is the covering radius of the landmark set (maximum distance from any point to its nearest landmark). For random landmark selection with ratio r:

```
epsilon ~ (1 / n)^(1/dim) * (dim * log(n))^(1/dim)   (asymptotic)
```

For farthest-point sampling:
```
epsilon <= 2 * (1/k)^(1/dim) * diam(X)   (deterministic bound)
```

where k = number of landmarks and diam(X) = diameter of the point set.

### Sparse parameter quick reference

```
Parameter               Recommended range       Effect
landmark_ratio          0.001 - 0.1             Lower = less memory, less accurate
error_tolerance         1e-12 - 1.0             Higher = more aggressive sparsification
max_radius              0.1 - 10.0              Lower = fewer edges, faster
num_landmarks           50 - 10000              Higher = more accurate, more memory

Typical configurations:
  n=50K,  dim<=3: landmark_ratio=0.05,  error_tolerance=0.01
  n=100K, dim<=3: landmark_ratio=0.02,  error_tolerance=0.01
  n=500K, dim<=3: landmark_ratio=0.005, error_tolerance=0.05
  n=1M+,  dim<=3: landmark_ratio=0.001, error_tolerance=0.1
```

Back to [Sparse Workflows Overview](index.md)
