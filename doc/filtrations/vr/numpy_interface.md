# NumPy interface

```python
import pynerve
import numpy as np

points = np.random.randn(2000, 3)

# Basic VR persistence
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=1.0,        # filtration cutoff (inf = no limit)
)

# Result format
pairs = result.pairs           # [(birth, death, dim), ...]
betti = result.betti_numbers   # [b0, b1, b2, ...]
```

### Explicit engine selection

```python
# PH4 -- sparse optimization engine, recommended for VR
result = pynerve.compute_persistence_ph4(points, max_dim=2, max_radius=1.0)

# PH5 -- balanced exact/approximate
result = pynerve.compute_persistence_ph5(points, max_dim=2, max_radius=1.0)

# PH6 -- high-precision accelerated
result = pynerve.compute_persistence_ph6(points, max_dim=2, max_radius=1.0)

# PH3 (cohomology) -- faster for H1 and above on large data
result = pynerve.compute_persistence_ph3(points, max_dim=2, max_radius=1.0)
```

### PyTorch interface (with autograd)

```python
import torch
import pynerve.torch

points = torch.randn(4, 2000, 3, device="cuda")

# Differentiable VR persistence
diagram = pynerve.torch.vr_persistence(
    points,
    max_dim=2,
    max_radius=1.0,
    metric="euclidean",
    return_simplices=False,  # set True for simplex indices
)

# diagram is a PersistenceDiagram with:
#   diagram.diagrams       -- tensor [batch, max_pairs, 3]
#   diagram.mask           -- valid-pair boolean mask
#   diagram.births()       -- birth times
#   diagram.deaths()       -- death times
#   diagram.total_persistence(p=2.0)

# Gradients flow through births and deaths
loss = diagram.total_persistence().sum()
loss.backward()
```

### nn.Module interface

```python
from pynerve.nn import PersistentHomology

ph = PersistentHomology(
    max_dim=2,
    max_radius=1.0,
    metric="euclidean",
    reduction="clearing",          # "standard" | "clearing" | "cohomology"
    memory_mode="standard",        # "standard" | "memory_mapped" | "streaming" | "extreme"
    max_memory_gigabytes=4.0,
)

diagrams = ph(points)  # list of [batch, max_pairs, 2] tensors per dimension
```

### Pipeline interface

```python
from pynerve.pipeline import vr_pipeline, analysis_pipeline

pipe = vr_pipeline(max_dim=2, max_radius=1.0, min_persistence=0.01)
result = pipe(points)

# Full analysis pipeline
pipe = analysis_pipeline(
    compute_fn=lambda data: pynerve.compute_persistence(data, max_dim=2, max_radius=1.0),
    representations=["diagram", "image", "landscape"],
)
output = pipe(points)
```


<- [Vietoris-Rips Overview](index.md)
