# Differentiable

## Quick start

```python
import torch
from pynerve.diff import DifferentiableVietorisRips

# Differentiable persistence layer
ph_layer = DifferentiableVietorisRips(max_dim=2)

# Forward: returns list of diagrams (one per dimension)
points = torch.randn(4, 100, 3, requires_grad=True)
diagrams = ph_layer(points)
# -> [tensor(batch, n0, 2), tensor(batch, n1, 2), tensor(batch, n2, 2)]

# Backward: gradients flow through persistence
loss = diagrams[1].sum()
loss.backward()
# -> points.grad is populated
```

Autograd-compatible persistent homology computation. Gradients flow through
birth and death coordinates into the input point cloud. Supports Vietoris-Rips
filtration with clearing reduction.


## API

```python
from pynerve.diff import (
    DifferentiableVietorisRips,
    DifferentiableAlphaComplex,
    DifferentiableCubical,
    FiltrationLearningLayer,
    LearnableFiltrationPersistence,
    PersistenceLoss,
    BettiNumberLoss,
    TopologicalComplexityLoss,
    StabilityLoss,
    TopologyLoss,
)

class DifferentiableVietorisRips:
    def __init__(self, max_dim=1, max_radius=None): ...
    def forward(self, points: Tensor, **options) -> List[Tensor]: ...

class FiltrationLearningLayer:
    def __init__(self, input_dim: int,
                 hidden_dims: Optional[List[int]] = None): ...
    def forward(self, points: Tensor) -> Tensor: ...

class LearnableFiltrationPersistence:
    def __init__(self, input_dim: int, max_dim=1,
                 hidden_dims: Optional[List[int]] = None): ...
    def forward(self, points: Tensor) -> Tuple[List[Tensor], Tensor]: ...

# Differentiable diagram losses
class PersistenceLoss:
    @staticmethod
    def diagram_wasserstein(d1: Tensor, d2: Tensor, p=2.0,
                            temperature=0.1) -> Tensor: ...
    @staticmethod
    def diagram_bottleneck(d1: Tensor, d2: Tensor,
                           temperature=0.01) -> Tensor: ...
    @staticmethod
    def persistence_kernel(d1: Tensor, d2: Tensor,
                           sigma=1.0) -> Tensor: ...

class BettiNumberLoss:
    def forward(self, diagram: Tensor,
                target_betti: Tensor) -> Tensor: ...

class TopologyLoss:
    def forward(self, pred_diagram: Tensor, target_diagram: Tensor,
                target_betti: Optional[Tensor] = None, ...) -> dict: ...
```

### Gradient flow

The backward pass computes:

```
dL / d_points[i] = sum over pairs (dL/d_birth + dL/d_death) * d_distance/d_points[i]
```

where `d_distance/d_points[i]` uses the distance matrix gradient `(x_i - x_j) / ||x_i - x_j||`.
Only dimension-0 pairs participate in the current implementation; higher-dimensional
gradients are zeroed.

### Use cases

- Topology-aware deep learning (shape reconstruction, generative models)
- Topological regularization of neural network representations
- Differentiable hyperparameter tuning for persistence computation
