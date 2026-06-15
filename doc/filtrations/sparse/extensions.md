# Extensions

[Back to index](index.md)

### Streaming sparse VR

For infinite or very large streams, the sparse VR can be updated
incrementally:

```python
# Pseudocode for streaming sparse VR
class StreamingSparseVR:
    def __init__(self, epsilon=0.3, max_dim=2):
        self.landmarks = []
        self.radii = []
        self.epsilon = epsilon

    def update(self, new_points):
        for p in new_points:
            d = min distance from p to existing landmarks
            if d > covering_radius:
                self.landmarks.append(p)
                self.radii.append(d)
                self._recompute_filtration()
```

The streaming variant adds points that become new landmarks (because
they are far from existing ones) and recomputes the sparse VR on the
updated landmark set.

### Distributed sparse VR

For distributed computation, each node processes a partition of the
data independently:

```python
# Each node builds its own sparse VR
node_result = pynerve.compute_persistence(
    node_points,
    max_dim=2,
    max_radius=2.0,
    sparse=True,
    sparse_parameter=0.3,
)

# Merge diagrams (use built-in diagram distances)
from pynerve.torch import diagram_wasserstein
distance = diagram_wasserstein(node_result.pairs_array, node_result.pairs_array, p=2.0, q=2.0)
```
