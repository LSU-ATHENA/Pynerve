# Quick Start

[Back to Index](index.md)

> Engine: `PersistenceEngine.PH3`. This is the default algorithm used by AUTO
> for most inputs.

Reduce the *coboundary matrix* instead of the boundary matrix, processing columns in reverse filtration order. Each column kills the highest-dimensional row it can, leading to sparser intermediate matrices and faster computation.

```python
import pynerve
import numpy as np

points = np.random.rand(500, 3)
result = pynerve.compute_persistence_ph3(
    points, max_dim=2, max_radius=0.5
)
print(result.pairs[:5])
```

Via the PyTorch module:

```python
from pynerve.nn import PersistentHomology

ph = PersistentHomology(max_dim=2, reduction="cohomology")
diagrams = ph(torch.randn(1, 500, 3))
```
