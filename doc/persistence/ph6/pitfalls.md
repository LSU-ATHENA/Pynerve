# Common Pitfalls

## Pitfall 1: Relying on PH6 for Production

PH6 algorithms may change or be removed in future releases without a major version bump. If your pipeline depends on a specific PH6 behavior, pin the library version and algorithm configuration:

```python
# Pin exact version
# requirements.txt: nerve==3.2.1

# Pin algorithm configuration
config = PH5PH6Config()
config.experimental_reduction_ordering = "adaptive"
config.experimental_adaptive_pivoting = True
config.experimental_verify_against_ph4 = True
```

## Pitfall 2: Benchmarking Without Baselines

PH6's performance claims are algorithm- and data-dependent. Always compare against PH4/PH5 on your specific data:

```python
import time

def benchmark(compute_fn, points, name):
    start = time.time()
    result = compute_fn(points, max_dim=2)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s, {len(result.pairs)} pairs")

points = np.random.rand(500, 3)
benchmark(lambda p: pynerve.compute_persistence_ph4(p), points, "PH4")
benchmark(lambda p: pynerve.compute_persistence_ph5(p), points, "PH5")
benchmark(lambda p: pynerve.compute_persistence_ph6(p), points, "PH6")
```

## Pitfall 3: Speculative Reduction Memory Blowup

Each speculative thread maintains its own copy of the boundary matrix. For k=8 threads and n=10^6 columns:

```
Memory per thread: ~megabytes
Total (8 threads): ~gigabytes
Baseline (1 thread): ~megabytes
Overhead: 8x
```

If memory is constrained, limit speculative threads:

```python
config.experimental_speculative_threads = 2  # 2x memory, ~1.5x speedup
```

## Pitfall 4: Approximate Clearing Correctness

Approximate clearing can produce incorrect results if the heuristic clears a column that later proves necessary. PH6's verification pass catches most errors, but the error detection is not 100% reliable:

```python
# PH6 will log a warning if false clearing is detected
config.experimental_clearing = "approximate"
config.structured_logging = True

result = pynerve.compute_persistence_ph6(points, max_dim=2)
engine = PH5PH6Engine(config)
metrics = engine.getComputationMetrics()

if metrics.false_clear_rate > 0.01:  # > 1% false clears
    print("WARNING: High false clear rate. Consider exact clearing.")
```

## Pitfall 5: Verification Pass Performance

When `experimental_verify_against_ph4 = True` (default), PH6 runs a PH4 computation alongside the PH6 computation and compares results. This doubles computation time. For research use, you may want to disable verification:

```python
config.experimental_verify_against_ph4 = False  # 2x faster, no cross-check
```


[Back to PH6 Index](index.md)
