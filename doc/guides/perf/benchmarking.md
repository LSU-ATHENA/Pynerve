# Benchmarking your own data

```python
import pynerve
import time
import numpy as np

def benchmark_persistence(points, max_dim=2, repeat=3):
    """Benchmark pynerve on a specific dataset."""
    times = []
    mems = []

    # Warmup
    _ = pynerve.compute_persistence(points[:100], max_dim=max_dim)

    for i in range(repeat):
        t0 = time.perf_counter()
        result = pynerve.compute_persistence(points, max_dim=max_dim)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        mems.append(result.diagnostics.get("peak_memory_bytes", 0))

    median_t = np.median(times)
    median_m = np.median(mems)

    print(f"Results ({repeat} runs):")
    print(f"  Time:   {median_t*1000:.1f} ms  (min: {min(times)*1000:.1f}, "
          f"max: {max(times)*1000:.1f})")
    print(f"  Memory: {median_m / 1e6:.1f} MB")
    print(f"  Pairs:  {len(result.pairs)}")
    print(f"  Betti:  {result.betti_numbers}")

    return median_t

# Compare backends
points = np.random.randn(10000, 3)
cpu_time = benchmark_persistence(points)

points_gpu = torch.from_numpy(points).cuda()
gpu_time = benchmark_persistence(points_gpu)

print(f"GPU speedup: {cpu_time / gpu_time:.1f}x")
```

[Back to index](index.md)
