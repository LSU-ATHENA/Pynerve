## Diagram operations: landscapes, silhouettes, persistence images

```python
# pynerve.diff._ph_representations

def compute_persistence_landscape(
    diagram: Tensor,           # (n_pairs, 2) -- [birth, death]
    n_layers: int = 5,
    resolution: int = 100,
) -> Tensor:                   # (n_layers, resolution)
```

```python
def persistence_image(
    diagram: Tensor,           # (n_pairs, 2)
    resolution: int = 20,
    sigma: float = 0.1,
) -> Tensor:                   # (resolution, resolution)
```

Both functions are built from pure PyTorch ops and are fully differentiable.


### Persistence landscape

At layer k, parameter t:

```
lambda_k(t) = k-th largest max(0, death - |t - (birth+death)/2|)
```

The landscape is sorted in descending order per column. Layers beyond the
number of pairs are zero-padded.

**Implementation (vectorized):**

```python
def compute_persistence_landscape(diagram, n_layers, resolution):
    # diagram: (n_pairs, 2)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    midpoints = (births + deaths) / 2
    persistences = deaths - births

    # Sample points
    t = torch.linspace(x_min, x_max, resolution, device=diagram.device)
    t = t.unsqueeze(0)  # (1, resolution)

    # Compute tent functions
    tents = torch.max(
        torch.zeros_like(t),
        persistences.unsqueeze(-1) - torch.abs(t - midpoints.unsqueeze(-1))
    )  # (n_pairs, resolution)

    # Sort descending
    tents_sorted, _ = torch.sort(tents, dim=0, descending=True)

    # Take top n_layers
    landscape = tents_sorted[:n_layers, :]  # (n_layers, resolution)
    return landscape
```


### Persistence image

Places a Gaussian bump (sigma) at each (birth, death) point, weighted by
persistence length. The result is a 2D grid.

**Implementation (vectorized):**

```python
def persistence_image(diagram, resolution, sigma):
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistences = deaths - births

    # Weight by persistence
    weights = persistences / persistences.sum()

    # Grid
    x = torch.linspace(birth_min, birth_max, resolution)
    y = torch.linspace(pers_min, pers_max, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Accumulate Gaussians
    image = torch.zeros(resolution, resolution, device=diagram.device)
    for i in range(len(births)):
        gauss = torch.exp(-((xx - births[i])**2 + (yy - persistences[i])**2)
                          / (2 * sigma**2))
        image += weights[i] * gauss

    return image
```

**Differentiability:** The image is differentiable w.r.t. both birth and
death coordinates, enabling gradient-based tuning of diagram representations.


### Diagram losses (Python)

```python
# pynerve.diff.topology_loss

class PersistenceLoss(nn.Module):
    # Sum of persistence lengths as a regularizer
    # loss = sum(death - birth) over all pairs

class BettiNumberLoss(nn.Module):
    # Penalize deviation from target Betti numbers
    # loss = ||betti(points) - target_betti||^2

class TopologicalComplexityLoss(nn.Module):
    # Encourage simpler topology (fewer, shorter features)
    # loss = sum(persistence^p) for p < 1

class StabilityLoss(nn.Module):
    # Penalize large diagram changes under perturbation
    # loss = W_2(diagram(x), diagram(x + epsilon))

class MultiScaleTopologyLoss(nn.Module):
    # Combine losses at multiple persistence scales

class LandscapeLoss(nn.Module):
    # L2 difference between persistence landscapes
    # loss = ||landscape(x) - target_landscape||^2

class TopologyLoss(nn.Module):  # compound
    # Weighted combination of multiple topology losses
```


### PyTorch neural network layers

```python
# pynerve.torch.nn_layers

class PersistenceLayer(nn.Module):
    # Differentiable persistence as an nn.Module
    def __init__(self, max_dim=1, max_radius=inf, metric="euclidean",
                 preprocessing=None, return_raw=False): ...
    def forward(self, x: Tensor) -> PersistenceDiagram | Tensor: ...

class VectorizationLayer(nn.Module):
    # Diagram -> fixed-size vector
    def __init__(self, method="landscape", **params): ...
    def forward(self, x: PersistenceDiagram) -> Tensor: ...

class TopologicalFeatureExtractor(nn.Module):
    # Persistence + vectorization pipeline
    def __init__(self, max_dim=1, max_radius=inf, metric="euclidean",
                 preprocessing=None, vectorization="landscape",
                 vectorization_params=None): ...
    def forward(self, x: Tensor) -> Tensor: ...

class TopologicalAttention(nn.Module):
    # Attention with topology gating
    def __init__(self, feature_dim, n_heads=4, dropout=0.1): ...
    def forward(self, features, diagrams=None) -> Tensor: ...

class DiagramPooling(nn.Module):
    # Pool diagram collections (mean/max/sum/attention)
    def __init__(self, method="mean", dim=1): ...
```


### Example: topology-regularized autoencoder

```python
from pynerve.torch import PersistenceLayer, VectorizationLayer

pers = PersistenceLayer(max_dim=1, max_radius=2.0)
vec = VectorizationLayer(method="landscape", num_levels=5)

x = torch.randn(100, 3, requires_grad=True)
diagram = pers(x.unsqueeze(0))
features = vec(diagram)

# Loss: reconstruction + topological regularization
recon_loss = F.mse_loss(reconstructed, x)
topo_loss = diagram.total_persistence()
loss = recon_loss + 0.01 * topo_loss
loss.backward()
```


### Complexity

- **Landscape**: O(p x L x R) forward and backward
- **Image**: O(p x R^2) forward and backward
- **PersistenceLayer**: O(n^2 log n + m x r) forward, O(n^2 + m x r) backward
- **TopologicalFeatureExtractor**: O(n^2 log n + m x r + L x R) forward and backward


## FAQ

**Are persistence landscapes and images fully differentiable?**

Yes. Both are built from pure PyTorch ops (linear algebra, Gaussian convolution, sorting) and support full backward propagation through birth and death coordinates.

**When should I use a landscape vs. a persistence image?**

Landscapes preserve the exact birth--death pairing and are efficient for low-dimensional diagrams. Images smooth over the diagram with Gaussian kernels, making them more robust to noise but losing pair-level resolution. The choice depends on whether pair identity matters for your loss.

**Can I backprop through the `PersistenceLayer`?**

Yes. `PersistenceLayer` wraps the differentiable VR persistence computation as an `nn.Module`. Call `backward()` or use `loss.backward()` on any downstream loss to obtain gradients with respect to the input point coordinates.


### Cross-references

- `pynerve.algorithms.vectorization`: C++ vectorization (non-differentiable)
- `pynerve.nn`: Neural network layers operating on diagrams
- `pynerve.ml`: ML pipeline integrating diagram features
