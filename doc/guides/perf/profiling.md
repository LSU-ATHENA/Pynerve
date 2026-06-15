# Profiling with perf

```bash
# CPU profiling
perf stat -e cycles,instructions,cache-misses,branch-misses \
    python -c "
import pynerve; import numpy as np
points = np.random.randn(10000, 3)
result = pynerve.compute_persistence(points, max_dim=2)
"

# Cache miss analysis
perf record -e cache-misses,L1-dcache-load-misses \
    python benchmark_script.py
perf report --stdio
```

### Key metrics to watch

Key metrics to watch include IPC (instructions per cycle) which should be above 2.0; if below, reduce memory latency and improve data locality. The L1 miss rate should be under 10%; if below that target, block computation to fit in L1. The L3 miss rate should be under 20%; if not, use NUMA-aware allocation and thread pinning. Branch mispredict rate should be under 5%; if higher, use __builtin_expect in hot paths. SIMD utilization should be above 80% of vector width; if not, ensure 16-byte alignment and contiguous data layout.

[Back to index](index.md)
