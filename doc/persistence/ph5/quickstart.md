# Quick Start

```python
import pynerve
import numpy as np

points = np.random.rand(200, 3)
result = pynerve.compute_persistence_up_to_dim_5(points, max_dim=2)
print(f"Found {len(result.pairs)} pairs")
```

With explicit configuration:

```python
from pynerve import PH5PH6Config, PH5PH6Engine

config = PH5PH6Config(
    enable_stability_checks=True,
    require_bitwise_reproducibility=True,
    enable_checksum_validation=True,
)
engine = PH5PH6Engine(config)
result = pynerve.compute_persistence_up_to_dim_5(
    points, max_dim=2, max_radius=0.5
)
```

Back to [PH5 Engine Overview](index.md)
