## Differentiable Persistence

Autograd-compatible persistent homology with gradients flowing back to input
point coordinates. Uses the merge tree (union-find for H0) to trace
birth/death gradients through the distance matrix.

```python
import torch
import pynerve.torch as nt

points = torch.randn(32, 3, requires_grad=True)
diagram = nt.vr_persistence(points, max_dim=1, max_radius=2.0)
loss = diagram.total_persistence()
loss.backward()  # gradients flow back to points
```


### Autograd function

```python
# pynerve.torch._persistence_vr

class _VRPersistenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, max_dim, max_radius, metric):
        # Returns (diagram_tensor, mask, num_pairs)
        # Calls C++ persistence engine
        ...

    @staticmethod
    def backward(ctx, grad_diagram, grad_mask, grad_num_pairs):
        # Returns (grad_points, None, None, None)
        # Reconstructs merge tree for H0
        # Backpropagates through distance matrix
        ...
```

### Forward pass

```python
def vr_persistence(points, max_dim=1, max_radius=inf, metric="euclidean",
                   return_simplices=False) -> PersistenceDiagram | tuple:
    # 1. Build pairwise distance matrix (differentiable)
    # 2. Build VR filtration (non-differentiable)
    # 3. Compute persistence (non-differentiable)
    # 4. Return padded tensor with mask
    ...
```

**Forward:** Uses the C++ persistence engine. Returns padded tensor
(batch, max_pairs, 2) with a mask for valid pairs.

### Backward pass

```python
def backward(ctx, grad_diagram, grad_mask, grad_num_pairs):
    # 1. For each pair (b,d), compute gradient contribution
    # 2. For H0: trace merge tree from death edge back to points
    # 3. For H1+: solve linear system at each simplex pair
    # 4. Backprop through distance computation
    ...
```

**Backward:** Reconstructs the merge tree (union-find for H0), traces
gradients from each birth/death coordinate back to the distance matrix,
then to the input points.

### Example: topology-regularized training

```python
import torch
import torch.nn as nn
import pynerve.torch as nt

class TopoRegularizedModel(nn.Module):
    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)

        # Topology regularization on latent space
        diagram = nt.vr_persistence(z.unsqueeze(0), max_dim=1)
        topo_loss = diagram.total_persistence()

        return recon, topo_loss

model = TopoRegularizedModel()
x = torch.randn(100, 64)
recon, topo_loss = model(x)
loss = nn.MSELoss()(recon, x) + 0.01 * topo_loss
loss.backward()
```


## Gradient computation details

### H0 merge tree backpropagation

For H0, the death time of a component is determined by the edge weight that merges it with another component:

```
death_i = w_merge = max(birth_i, weight_of_merge_edge)
```

The gradient of death_i with respect to point coordinates is:

```
d(death_i) / d(x_k) = d(w_merge) / d(x_k)
```

For the birth time:

```
birth_i = w_birth = weight_of_birth_edge
d(birth_i) / d(x_k) = d(w_birth) / d(x_k)
```

### H1+ boundary matrix backpropagation

For H1 and higher, the death is determined by the pivot during column reduction:

```
d(death_i) / d(x_k) = d(pivot_value) / d(x_k)
```

where `pivot_value` is the maximum simplex weight in the pivot column.

### Gradient flow example

```python
# Track gradient flow through persistence
points = torch.randn(32, 3, requires_grad=True)
diagram = nt.vr_persistence(points, max_dim=1)

# Compute gradient of total persistence w.r.t. each point
loss = diagram.total_persistence(p=2.0)
loss.backward()

# points.grad now contains influence of each point on topology
# High gradient magnitude = point strongly affects topological features
influence = points.grad.norm(dim=1)
print(f"Topologically influential points: {(influence > 0.1).sum()}")
```

### Practical autograd examples

```python
import torch
import pynerve.torch as nt

# Topology-preserving point cloud denoising
noisy_points = clean_points + torch.randn_like(clean_points) * 0.1
noisy_points.requires_grad_(True)

optimizer = torch.optim.SGD([noisy_points], lr=0.01)
for step in range(100):
    diagram = nt.vr_persistence(noisy_points.unsqueeze(0), max_dim=1)

    # Encourage long persistence (topological signal)
    persistences = diagram.diagrams[0, :, 1] - diagram.diagrams[0, :, 0]
    topo_loss = -persistences.topk(5).values.sum()

    # Discourage point movement (denoising)
    mse_loss = nn.MSELoss()(noisy_points, clean_points)

    loss = topo_loss + 0.1 * mse_loss
    loss.backward()
    optimizer.step()
```

### Custom autograd function

```python
# Create a custom differentiable persistence function
class CustomPersistence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, max_dim, max_radius):
        diagram = nt.vr_persistence(points, max_dim, max_radius)
        ctx.save_for_backward(points, diagram.diagrams)
        ctx.max_dim = max_dim
        ctx.max_radius = max_radius
        return diagram.diagrams

    @staticmethod
    def backward(ctx, grad_output):
        points, diagrams = ctx.saved_tensors
        # Custom backward logic
        grad_points = compute_custom_gradient(grad_output, diagrams, points)
        return grad_points, None, None
```


## Numerical considerations

Numerical precision affects gradient error, memory, and speed. float32 offers gradient error of roughly 1e-3 with baseline memory and speed. float64 reduces gradient error to roughly 1e-10 but uses double the memory at roughly 80% speed. Mixed precision (float32 forward, float64 backward) gives gradient error of roughly 1e-5 with 1.5x memory usage and 1.2x speed relative to float32 baseline.

Use float64 for applications where gradient accuracy matters (e.g., topology-regularized training with tight convergence requirements).


## FAQ

**Q: Why does loss.backward() produce NaN gradients?**
A: NaN gradients typically occur when: (1) two points are exactly coincident (zero-distance edge), (2) a merge event involves an edge with weight equal to existing birth time (degenerate), or (3) the loss involves essential classes (infinite death). Add a small epsilon to distances or filter essential classes.

**Q: Can I compute higher-order gradients?**
A: Yes. Set `create_graph=True` in `loss.backward()` to compute second-order gradients. This is useful for Hessian-based optimization or meta-learning with topological features.

**Q: How do I handle batches with different numbers of pairs?**
A: The forward pass returns padded tensors with a mask. The mask indicates which pairs are valid. Apply your loss only to masked positions. The gradient through masked positions is zero.


### Cross-references

- `pynerve.torch`: PyTorch integration overview
- `pynerve.autodiff.gradients`: C++ autodiff engine
- `pynerve.diff`: Python differentiable persistence layers
- `pynerve.ml`: ML pipeline using differentiable topology
