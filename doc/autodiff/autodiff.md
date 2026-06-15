# Automatic Differentiation for Persistence

## Quick start

```python
import torch
import pynerve.torch as nt

# Differentiable VR persistence
points = torch.randn(4, 32, 3, requires_grad=True)
diagram = nt.vr_persistence(points, max_dim=1, max_radius=2.0)

# Use diagram in a loss
loss = diagram.total_persistence()
loss.backward()
print(points.grad)  # gradients w.r.t. each point coordinate
```

Pynerve provides differentiable persistent homology through multiple mechanisms:
a native C++ autodiff engine (`nerve::autodiff`), a PyTorch integration
(`pynerve.torch`) with autograd-compatible persistence layers, and a high-level
Python API for topology-regularized loss functions.


## Module structure

- [Gradients](gradients.md) -- Autograd persistence function, differentiable persistence (C++)
- [Graph operations](graph_ops.md) -- Tensor, Variable, ComputationalGraph, autograd graph topology
- [Tensor operations](tensor_ops.md) -- Diagram landscapes, silhouettes, images, losses, PyTorch layers
- [GPU kernels](gpu_kernels.md) -- CUDA autodiff kernels for elementwise, topological, reduction gradients


## Use cases

**Topology-regularized loss:**
```python
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
points = model(data)  # latent representation
diagram = nt.vr_persistence(points.unsqueeze(0), max_dim=1)
loss = F.mse_loss(output, target) + 0.01 * diagram.total_persistence()
loss.backward()
```

**Topological autoencoder:**
```python
class TopoAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 32)
        self.decoder = nn.Linear(32, 784)
        self.pers = PersistenceLayer(max_dim=1)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        diag = self.pers(z.unsqueeze(0))
        topo_loss = diag.total_persistence()
        return recon, topo_loss
```

**Learnable filtration:**
```python
# pynerve.diff.ph_layer.LearnableFiltrationPersistence
model = LearnableFiltrationPersistence(input_dim=3, max_dim=1)
diagrams, filt_values = model(points)
```


## How gradient propagation works

The key insight is that persistence is piecewise-constant in the filtration
values. Gradients are zero almost everywhere, but the subgradient at
critical points (where births/deaths change) is well-defined.

**H0 gradient (union-find merge tree):**

For H0, the birth of a component is when a vertex appears (its filtration
value). The death is when an edge connects two components. The merge tree
records this hierarchy, and gradients flow through:

1. death gradient -> edge weight (distance between points)
2. edge weight -> point coordinates

**H1+ gradient:**

For higher dimensions, the reduction algorithm creates a linear system
between simplex pairs. The gradient of each pair's birth/death w.r.t. the
input simplices' filtration values is computed by differentiating through
the column operations.


## Complexity notes

- VR persistence (H0): O(n^2 log n) forward, O(n^2) backward
- VR persistence (H1+): O(m x r) forward and backward
- Persistence landscape: O(p x k x R) forward and backward
- Persistence image: O(p x R^2) forward and backward
- Gradient validation: O(p x n x d) forward only


## Differentiability limitations

- **Distance matrix** -- differentiable (smooth in point coordinates)
- **Filtration sorting** -- not differentiable (straight-through estimator)
- **Boundary matrix** -- not differentiable (discrete combinatorial structure)
- **Column reduction** -- not differentiable (treated as constant in backward)
- **Birth/death extraction** -- differentiable (linear in filtration values)
- **Vectorization (landscape)** -- differentiable (piecewise-linear)
- **Vectorization (image)** -- differentiable (Gaussian smoothing)
- **Essential class detection** -- not differentiable (thresholding operation)


## Common pitfalls

1. **Vanishing gradients**: If no births/deaths change under perturbation,
   backward returns zero gradients. This is correct behavior.

2. **Max radius clamping**: Features born at max_radius do not propagate
   gradients (they appear as essential classes).

3. **Batch dependence**: The number of output pairs varies across batch
   elements. Padded tensors with masks are used; gradients on masked
   positions are zero.

4. **Memory**: The backward pass stores the merge tree (for H0) and
   reduction state (for H1+), which can be O(m) in simplices.


## FAQ

**Why are gradients zero for some persistence pairs?**

When births and deaths are locally constant with respect to input perturbations, the subgradient is zero. This is correct behavior -- only pairs at critical points (where a perturbation changes the pairing) carry non-zero gradients.

**Does Pynerve support gradients for H1 and above?**

Yes. H0 uses the merge-tree hierarchy; H1+ differentiates through the column reduction algorithm by solving a linear system at each simplex pair.

**Can I use differentiable persistence without PyTorch?**

Yes. The native C++ autodiff engine (`nerve::autodiff`) works independently of PyTorch. The `AutodiffScalar<T>` type tracks gradients through all arithmetic operations.


### Cross-references

- `pynerve.torch`: PyTorch integration with autograd persistence
- `pynerve.diff`: Python differentiable persistence modules
- `pynerve.nn`: Neural network layers using differentiable topology
- `pynerve.ml`: ML pipeline leveraging topology gradients
