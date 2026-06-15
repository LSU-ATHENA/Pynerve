# Quick Start

```python
import pynerve
import numpy as np

points = np.random.rand(100, 3)
result = pynerve.compute_persistence(points, max_dim=2)
print(result.pairs[:5])  # [(birth, death, dim), ...]
```

Via the PyTorch module:

```python
from pynerve.nn import PersistentHomology

ph = PersistentHomology(max_dim=2)
diagrams = ph(torch.randn(1, 100, 3))  # list [dim_0, dim_1, dim_2]
```

<- [Standard Reduction Overview](index.md)
