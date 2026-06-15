# Migration Path

An experimental algorithm that proves successful in PH6 graduates to PH5 (and later PH4's adaptive selector) in a subsequent release. To stay informed about graduation status, check the release notes.

## Graduation History

In release 1.0, clearing optimization graduated from PH6 to PH4 and PH5. In release 1.2, cohomology reduction graduated from PH6 to become the PH4 default. In release 2.0, SIMD acceleration graduated from PH6 to PH4 adaptive. In release 3.0, adaptive pivoting graduated from PH6 to PH5 (planned).

## Tracking Experimental Features

Enable experimental logging to track which algorithms were used:

```python
config = PH5PH6Config()
config.structured_logging = True

result = pynerve.compute_persistence_ph6(points, max_dim=2)
engine = PH5PH6Engine(config)
metrics = engine.getComputationMetrics()

# Check which experimental features were active
for entry in metrics.experimental_log:
    print(f"{entry.feature}: active={entry.active}, "
          f"speedup={entry.estimated_speedup:.2f}x, "
          f"calls={entry.total_calls}")
```


[Back to PH6 Index](index.md)
