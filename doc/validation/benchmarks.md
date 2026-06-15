## Benchmarks

Per-kernel microbenchmarks for individual kernels or full pipelines.
Supports CPU and GPU backends.

```python
from pynerve.validation import benchmark

result = benchmark(
    kernel="persistence",
    n_points=5000,
    max_dim=2,
    max_radius=2.0,
    backend="cuda",
    trials=5,
    warmup=2,
    detailed=True,
)
```

### Result fields

```python
result = {
    "mean_time_ms": float,
    "std_time_ms": float,
    "min_time_ms": float,
    "max_time_ms": float,
    "trials": int,
    "kernel": str,
    "backend": str,
    "breakdown": {
        "distance_ms": float,
        "filtration_ms": float,
        "reduction_ms": float,
        "pair_extraction_ms": float,
    },
    "gpu_info": {...},
    "hardware_info": {...},
}
```

### Available kernel benchmarks

The benchmark suite supports over a dozen kernels covering the full pipeline and individual stages. Full-pipeline benchmarks include **persistence**. Distance matrix and simplex construction kernels include **distance** and **filtration**. Matrix reduction kernels include **reduction**, **cohomology_reduction**, **clearing**, and **apparent_pairs**. Spectral construction kernels include **laplacian**, **dirac**, **eigensolver**, and **sheaf_laplacian**. Optimization kernels include **sheaf_learning**. Algebraic topology kernels include **cup_product**. Graph and topological operation kernels include **reeb_graph**, **zigzag**, **bfs**, **pagerank**, and **message_passing**. Comparison and visualization kernels include **wasserstein**, **bottleneck**, and **mapper**. All kernels have GPU support.

### CPU vs GPU comparison

```python
result_cpu = benchmark(kernel="reduction", n_points=2000, backend="cpu")
result_gpu = benchmark(kernel="reduction", n_points=20000, backend="cuda")
print(f"CPU: {result_cpu['mean_time_ms']:.1f} ms")
print(f"GPU: {result_gpu['mean_time_ms']:.1f} ms")
```

### PH5/PH6 benchmark CI

Automated regression suite in the CI pipeline. Runs 10 benchmark
configurations and fails if any regresses by >5%.

The CI suite defines expected thresholds across three data sizes. At 1,000 points with max_dim=2 on CPU, PH5 expects under 50 ms and PH6 under 100 ms. At 10,000 points with max_dim=2, CPU thresholds are under 2 seconds (PH5) and 5 seconds (PH6), while GPU thresholds are under 200 ms and 500 ms. At 100,000 points with max_dim=1, CPU expects under 5 seconds (PH5) and 10 seconds (PH6), and GPU expects under 500 ms and 1 second.


## Writing custom benchmarks

```python
from pynerve.validation import benchmark, BenchmarkConfig

cfg = BenchmarkConfig(
    kernel="persistence",
    n_points=5000,
    max_dim=2,
    backend="cuda",
    trials=10,
    warmup=3,
    detailed=True,
    profile_cuda_kernels=True,  # NVProf-style kernel timing
)

result = benchmark(cfg)
```

### Custom kernel registration

```cpp
#include <nerve/validation/benchmark_registry.hpp>

REGISTER_BENCHMARK("my_kernel", [](const BenchmarkArgs& args) {
    auto data = generateData(args.n_points);
    BENCHMARK_BLOCK("computation") {
        my_kernel(data);
    };
    return BenchmarkResult{
        .breakdown = {{"computation_ms", block_time}},
        .metadata = {{"data_size", data.size()}},
    };
});
```

Then call from Python:

```python
result = benchmark(kernel="my_kernel", n_points=1000)
```

### Benchmark report generation

```python
from pynerve.validation import BenchmarkReport, compare_to_baseline

# Generate report
report = BenchmarkReport()
report.add_result(result_cpu)
report.add_result(result_gpu)
report.save("benchmark_results.json")

# Compare to baseline
baseline = BenchmarkReport.load("benchmark_expected.json")
comparison = compare_to_baseline(report, baseline)

for name, delta in comparison.regressions():
    print(f"REGRESSION: {name} is {delta*100:.1f}% slower")
```

### Scaling benchmarks

```python
from pynerve.validation import scaling_benchmark

# Measure scaling with data size
results = scaling_benchmark(
    kernel="reduction",
    sizes=[100, 500, 1000, 5000, 10000],
    backend="cuda",
    log_scale=True,
)

# Fit complexity model
# Returns estimated exponent: O(n^exponent)
print(f"Estimated complexity: O(n^{results.estimated_exponent:.2f})")
```

### Benchmark environment tracking

Each benchmark result includes environment metadata for reproducibility:

```python
print(result.hardware_info)
# {
#   "cpu": "AMD EPYC 9654 96-Core",
#   "gpu": "NVIDIA H100 (tens of gigabytes HBM3)",
#   "cuda_version": "12.4",
#   "driver_version": "550.54.15",
#   "ram_gb": 512,
#   "num_cpu_cores": 96,
# }
```


## Continuous benchmark dashboard

The CI pipeline generates a benchmark dashboard:

```yaml
# .github/workflows/benchmark-dashboard.yml
- name: Generate dashboard
  run: |
    python -m pynerve.validation.benchmark --suite=full \
      --output-dir=benchmark-results/
    python -m pynerve.validation.report_dashboard \
      --input-dir=benchmark-results/ \
      --output=benchmark-dashboard.html
```

The dashboard includes:
- Histograms of kernel runtimes
- Scaling plots (time vs n_points)
- GPU utilization and memory bandwidth
- Comparison with previous run (regression highlighting)
- Hardware configuration summary


## FAQ

**Q: How many trials should I use?**
A: For stable measurements, use 5-10 trials with 2-3 warmup iterations. GPU benchmarks are more stable (lower variance) than CPU benchmarks due to dedicated hardware.

**Q: Why does the first trial take longer?**
A: CUDA kernel compilation, GPU warm-up, and caching effects. The warmup parameter discards the first N trials to get steady-state performance.

**Q: Can I benchmark a single kernel without the full pipeline?**
A: Yes. Use `kernel="reduction"` to benchmark only the matrix reduction step, or specify any kernel from the available list. Each kernel can be benchmarked independently.

**Q: How do I add a new kernel to the benchmark suite?**
A: Use `REGISTER_BENCHMARK` in C++ to register the kernel, then call `benchmark(kernel="new_kernel")` from Python. The kernel will appear in the available kernels list automatically.


### Cross-references

- `pynerve.validation`: Validation overview
- `pynerve.cuda`: GPU benchmark kernels
- `pynerve.algorithms`: Algorithm benchmarks
- `pynerve.cuda.determinism`: Deterministic benchmark configuration
