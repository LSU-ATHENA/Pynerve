# Quick Start

```python
import pynerve
import numpy as np

points = np.random.rand(200, 3)
result = pynerve.compute_persistence_up_to_dim_6(points, max_dim=2)
print(f"Found {len(result.pairs)} pairs")
```

With explicit engine configuration:

```python
config = PH5PH6Config(
    numerical_tolerance=1e-12,
    enable_stability_checks=True,
)

engine = PH5PH6Engine(config)
result = pynerve.compute_persistence_up_to_dim_6(points, max_dim=2)

metrics: PH5PH6Metrics = engine.getComputationMetrics()
print(f"Time: {metrics.computation_time_ms:.1f}ms")
print(f"Quality: {metrics.quality_score:.4f}")
```


[Back to PH6 Index](index.md)
