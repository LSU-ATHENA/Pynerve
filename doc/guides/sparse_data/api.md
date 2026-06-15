# API reference

```python
# Core persistence -- sparse via mode=APPROX
pynerve.compute_persistence(
    points,
    mode=pynerve.PersistenceMode.APPROX,  # enables sparse VR + edge collapse
    max_dim=2,
    error_tolerance=0.01,               # controls approximation quality
)

# PyTorch sparse persistence
from pynerve.nn import SparsePH
sparse = SparsePH(
    max_dim=2,
    max_radius=float("inf"),
    landmark_ratio=0.1,  # fraction of points as landmarks
    metric="euclidean",
)

# Sparse Rips with explicit parameter
from pynerve.nn import SparseRipsPersistence
sr = SparseRipsPersistence(
    sparse_parameter=0.1,  # epsilon for greedy cover
    max_dim=1,
)

# Witness complex
from pynerve.nn import WitnessComplexPersistence
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=2,
    method="farthest",  # "farthest" | "random" | "kmeans"
)

# Low-level sparse ops
from pynerve.fast_ops import (
    boundary_matrix_sparse,
    column_reduction_sparse,
    vietoris_rips_filtration_fast,
    nearest_neighbors_fast,
)
```

Back to [Sparse Workflows Overview](index.md)
