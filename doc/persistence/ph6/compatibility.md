# Compatibility

PH6 shares its configuration interface with PH5:

```python
from pynerve import PH5PH6Config, PH5PH6Engine, PH5PH6Metrics

# The same Config and Metrics classes work for both PH5 and PH6
config = PH5PH6Config()
config.numerical_tolerance = 1e-12
config.enable_stability_checks = True

engine = PH5PH6Engine(config)
result = pynerve.compute_persistence_up_to_dim_6(points, max_dim=2)

metrics: PH5PH6Metrics = engine.getComputationMetrics()
print(f"Time: {metrics.computation_time_ms:.1f}ms")
print(f"Quality: {metrics.quality_score:.4f}")
```

## PH6-Specific Configuration Options

PH6 extends `PH5PH6Config` with additional fields (only available in PH6). The `experimental_reduction_ordering` field is a string that defaults to `"default"` and accepts `"default"`, `"adaptive"`, or `"reverse"`. The `experimental_clearing` field is a string that defaults to `"exact"` and accepts `"exact"`, `"approximate"`, or `"aggressive"`. The `experimental_speculative_threads` field is an integer defaulting to `1` that controls the number of speculative threads (1 means off). The `experimental_adaptive_pivoting` field is a boolean defaulting to `True` that enables adaptive pivot strategy selection. The `experimental_block_size` field is an integer defaulting to `0` that sets the block size for cache-blocked reduction (0 means auto). The `experimental_verify_against_ph4` field is a boolean defaulting to `True` that cross-verifies results against PH4.

```python
config = PH5PH6Config(
    experimental_reduction_ordering="adaptive",
    experimental_clearing="approximate",
    experimental_adaptive_pivoting=True,
    experimental_block_size=4096,
)
```


[Back to PH6 Index](index.md)
