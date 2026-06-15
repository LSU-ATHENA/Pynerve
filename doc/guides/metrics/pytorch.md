# Distance metric integration with PyTorch modules

Custom metrics work with all PyTorch persistence modules:

```python
from pynerve.nn import PersistentHomology, SparsePH, WitnessComplexPersistence
import torch

# Custom metric for PersistentHomology
def my_metric(a, b):
    return torch.sum(torch.abs(a - b))

ph = PersistentHomology(max_dim=2, metric=my_metric)
diagrams = ph(points_tensor)

# Custom metric for SparsePH
sparse_ph = SparsePH(max_dim=2, landmark_ratio=0.1, metric="cosine")
diagrams = sparse_ph(points_tensor)

# Precomputed distance matrix with PyTorch
dist_mat = torch.cdist(points_tensor, points_tensor)
result = pynerve.compute_persistence(
    dist_mat.numpy(), max_dim=2, metric="precomputed"
)
```

[Back to index](index.md)
