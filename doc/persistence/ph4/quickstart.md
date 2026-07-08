# Quick Start

```python
import pynerve
import numpy as np

points = np.random.rand(200, 3)

# PH4 is the default engine
result = pynerve.compute_persistence(points)
# Equivalent to:
result = pynerve.compute_persistence_up_to_dim_4(points)

print(f"Found {len(result.pairs)} persistence pairs")
print(f"Betti numbers: {result.betti_numbers}")
```

With explicit options:

```python
from pynerve import PersistenceOptions, PersistenceMode, PersistenceBackend

opts = PersistenceOptions(max_dim=2, threads=8)

result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_radius=0.8)
```

Back to [PH4 Engine Overview](index.md)
