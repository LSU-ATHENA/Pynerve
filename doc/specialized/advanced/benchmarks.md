# Performance benchmarks

```python
from pynerve.validation import benchmark_specialized

# Cup product scaling
for n in [100, 500, 1000]:
    bm = benchmark_specialized("cup_product", n_points=n, max_dim=2)
    print(f"n={n}: {bm.mean_time_ms:.1f}ms (GPU: {bm.gpu_time_ms:.1f}ms)")

# Reeb graph scaling
for n in [1000, 10000, 100000]:
    bm = benchmark_specialized("reeb_graph", n_points=n)
    print(f"n={n}: {bm.mean_time_ms:.1f}ms")

# Zigzag scaling
for t in [5, 10, 20]:
    bm = benchmark_specialized("zigzag", n_times=t, n_points=500)
    print(f"t={t}: {bm.mean_time_ms:.1f}ms")
```


[Back to index](index.md)
