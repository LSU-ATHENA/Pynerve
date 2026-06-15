## Differentiable simplex operations

Located in `src/ml/diff/`. Provides differentiable operations on simplicial
complexes for gradient-based topology optimization.

```cpp
// Differentiable birth/death computation
// Allows gradients to flow through persistence computation
// Used by pynerve.nn.PersistentHomology and pynerve.torch.vr_persistence
```

These underpin the autograd through persistence in
`pynerve.torch.vr_persistence` and `pynerve.nn.PersistentHomology`. See also
`python/pynerve/diff/` for the Python layer.

```python
import torch
from pynerve.nn import PersistentHomology

# Differentiable persistence: gradients flow through birth/death times
layer = PersistentHomology(max_dim=2)
diagram = layer(points)  # points: [batch, n, 3], diagram: [batch, n_pairs, 2]

# Loss based on topological features can backpropagate
loss = torch.sum(diagram[:, :, 1] - diagram[:, :, 0])  # sum of persistences
loss.backward()
```

Key operations:

The key operations are: `DifferentiableBirth` for birth time with gradient w.r.t. point positions, `DifferentiableDeath` for death time with gradient w.r.t. point positions, `PersistenceLoss` as a loss function on birth-death pairs, and `TopologyPreservingLoss` as a regularizer that preserves topological features.


## Gradient flow through persistence

The key insight: death times of H0 features are determined by merge events in the union-find tree, which are differentiable with respect to the input point coordinates.

### Birth gradient

```
d(birth_i) / d(x_k) = sum over edges e in the birth simplex of d(w_e) / d(x_k)
```

where w_e is the edge weight (distance between points).

### Death gradient (H0)

```
d(death_i) / d(x_k) = d(w_merge) / d(x_k)
```

where w_merge is the weight of the edge that causes the merge (death of the younger component).

### Example: topology-preserving autoencoder

```python
import torch
import torch.nn as nn
from pynerve.nn import PersistentHomology

class TopoAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
        self.topo_layer = PersistentHomology(max_dim=1)

    def forward(self, x):
        z = self.encoder(x)

        # Topology loss on latent space
        diagram = self.topo_layer(z.unsqueeze(0))
        # Encourage separation: maximize persistence of top features
        persistences = diagram[:, :, 1] - diagram[:, :, 0]
        top_k = persistences.topk(3).values
        topo_loss = -top_k.sum()  # maximize top persistences

        recon = self.decoder(z)
        recon_loss = nn.MSELoss()(recon, x)

        return recon, topo_loss

model = TopoAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x in dataloader:
    recon, topo_loss = model(x)
    loss = recon + 0.01 * topo_loss
    loss.backward()
    optimizer.step()
```

### Loss functions

```python
from pynerve.diff import PersistenceLoss, TopologyPreservingLoss

# Persistence loss: encourage features with specific persistence
loss_fn = PersistenceLoss(target_persistence=0.5)
loss = loss_fn(diagram)  # penalizes deviation from target persistence

# Topology-preserving loss: regularizer to maintain topology
loss_fn = TopologyPreservingLoss(
    reference_diagram=reference,
    p=2.0,
)
loss = loss_fn(diagram)  # penalizes changes in persistence pairing
```

### Numerical stability

```python
# Use double precision for stable gradient computation
points = points.double().requires_grad_(True)
diagram = layer(points)
loss = diagram.total_persistence()
loss.backward()

# Gradient clipping to prevent explosion
torch.nn.utils.clip_grad_norm_(points, max_norm=1.0)
```

Gradients through persistence can be large (especially near merge events). Gradient clipping and double precision help maintain stable training.


## FAQ

**Q: Why does the backward pass reconstruct the merge tree?**
A: The merge tree (union-find structure) is computed in the forward pass but not stored. The backward pass reconstructs it from the edge weights and persistence pairs, using the same deterministic sorting as the forward pass.

**Q: Are gradients defined for H1 and higher?**
A: Yes. H1+ death gradients flow through the column reduction of the boundary matrix. The gradient traces which simplex pair caused the death, similar to the H0 merge tree but using the matrix reduction pivot chain.

**Q: What happens when a feature is born at infinity?**
A: Features with infinite death have no death coordinate. The `deaths_grad()` method returns zero for these features. Loss functions should handle this by masking or ignoring essential classes.

**Q: Is differentiable persistence fast enough for training?**
A: The forward+backward pass is 2-3x slower than forward-only persistence. For a point cloud of 1000 points with max_dim=1, expect ~10ms on GPU. For large batches, use mixed precision and gradient checkpointing.


### Cross-references

- `pynerve.autodiff`: Automatic differentiation engine
- `pynerve.torch.autograd`: PyTorch autograd integration
- `pynerve.diff`: Python differentiable persistence layers
- `pynerve.optimization.gpu`: GPU optimizer for topology training
