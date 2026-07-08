# Benchmark Methodology

To reliably measure PH6 performance, control for the following variables:

```python
import time
import statistics

def benchmark_ph6(points, config, num_trials=5):
    """Benchmark PH6 with warmup and multiple trials."""
    # Warmup: one run to warm caches
    _ = pynerve.compute_persistence_up_to_dim_6(points, config, max_dim=2)

    # Timed trials
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        _ = pynerve.compute_persistence_up_to_dim_6(points, config, max_dim=2)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
        "trials": num_trials,
    }
```

**What to report**:

1. Hardware: CPU model, cores, cache sizes, RAM
2. Software: library version, compiler, flags
3. Data: number of points, dimensions, max_radius, max_dim
4. Complex statistics: n (simplices), k (avg column size)
5. Timing: mean, std, min, max across trials
6. PH6 features: which experimental features were active


[Back to PH6 Index](index.md)
