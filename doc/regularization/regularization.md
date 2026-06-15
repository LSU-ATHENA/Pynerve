# Regularization

## Quick start

```python
import torch
from pynerve.regularization import TopologyLoss

# Topological loss for diagram-guided training
loss_fn = TopologyLoss(
    wasserstein_weight=1.0,
    betti_weight=0.1,
    complexity_weight=0.01,
)

pred_diagram = torch.randn(10, 3)       # (n_pairs, 3) = (birth, death, dim)
target_diagram = torch.randn(8, 3)
target_betti = torch.tensor([1, 0, 0])   # expected Betti numbers

losses = loss_fn(pred_diagram, target_diagram, target_betti=target_betti)
# -> {'wasserstein': ..., 'betti': ..., 'complexity': ..., 'total': ...}
```

Topological regularization losses for ML pipelines, GPU-accelerated
augmentation kernels, and components for integrating topology constraints
into neural network training.


## API

```python
# ---- Combined loss ----
class TopologyLoss(nn.Module):
    def __init__(self, wasserstein_weight=1.0, betti_weight=0.1,
                 complexity_weight=0.01, stability_weight=0.0): ...
    def forward(self, pred_diagram, target_diagram,
                target_betti=None, points=None,
                persistence_fn=None) -> dict: ...

# ---- Individual losses ----
class PersistenceLoss(nn.Module):
    @staticmethod def diagram_wasserstein(dgm1, dgm2, p=2.0, temperature=0.1): ...
    @staticmethod def diagram_bottleneck(dgm1, dgm2, temperature=0.01): ...
    @staticmethod def persistence_kernel(dgm1, dgm2, sigma=1.0): ...

class BettiNumberLoss(threshold=0.1, temperature=0.1):
    def forward(self, diagram, target_betti): ...

class TopologicalComplexityLoss(measure="total_persistence"):
    def forward(self, diagram): ...

class StabilityLoss(epsilon=0.01, num_samples=5):
    def forward(self, points, persistence_fn): ...

class MultiScaleTopologyLoss(scales=(0.01, 0.1, 0.5, 1.0)):
    def forward(self, diagram, target_diagrams): ...

class LandscapeLoss(n_layers=5, resolution=100):
    def forward(self, diagram1, diagram2): ...

# ---- Dropout ----
class PersistentDropout(p=0.5, persistence_aware=True, temperature=0.1):
    def forward(self, x, persistence_scores=None): ...
class TopologyPreservingDropout(p=0.3, betti_preserve=True, connectivity_preserve=True):
    def forward(self, activations, adjacency=None): ...

# ---- Batch norm ----
class PersistentBatchNorm(num_features, persistence_weighted=True, eps=1e-5):
    def forward(self, x, persistence_scores=None): ...

# ---- Regularizers ----
class MorseRegularizer(lambda_critical=0.1, lambda_morse=0.05):
    def forward(self, function_values, gradient_values=None): ...
class BettiConstraintLayer(target_betti, persistence_fn, lambda_constraint=0.1):
    def forward(self, x): ...
class TopologicalSmoothness(lambda_smooth=0.1, neighborhood_size=5):
    def forward(self, features, persistence_diagrams): ...
class HomotopyRegularizer(lambda_homotopy=0.01):
    def forward(self, current_output, target_topology): ...
```


## TopologyLoss

Combined loss that sums multiple topological objectives:

```
total = w_wass * L_wasserstein(pred, target)
      + w_betti * L_betti(pred, target_betti)
      + w_comp * L_complexity(pred)
      + w_stab * L_stability(points, persistence_fn)
```

Each component can be disabled by setting its weight to 0 (default for
`stability_weight`).

`forward()` returns a dict of individual loss terms plus `"total"` for
convenience.


## Individual losses

### PersistenceLoss

```python
# Smooth Wasserstein distance (Sinkhorn-based)
loss = PersistenceLoss.diagram_wasserstein(pred, target, p=2.0, temperature=0.1)

# Smooth bottleneck distance (softmin-based)
loss = PersistenceLoss.diagram_bottleneck(pred, target, temperature=0.01)

# Persistence kernel (RBF on birth-death plane)
kernel = PersistenceLoss.persistence_kernel(dgm_a, dgm_b, sigma=1.0)
```

All three are differentiable with respect to diagram coordinates.

### BettiNumberLoss

Differentiable Betti number matching. Uses a sigmoid-based `soft_step` to
count persistent points.

```python
betti_loss = BettiNumberLoss(threshold=0.1, temperature=0.1)
loss = betti_loss(pred_diagram, target_betti)  # MSE on Betti numbers
```

### TopologicalComplexityLoss

Regularizes diagram complexity:

- `total_persistence`: sum(death - birth)
- `persistence_entropy`: -sum(p_i * log p_i)
- `num_features`: count(persistence > 0.1)
- `max_persistence`: max(death - birth)


## GPU augmentation kernels

`src/regularization/loss_kernels.cu` and `src/regularization/regularizer_gpu.cu`:

```cpp
namespace nerve::regularization::gpu {

// Forward pass for topological loss
__global__ void loss_forward_kernel(
    const float* pred_diagram, const float* target_diagram,
    float* loss, int n_pairs, int n_target);

// Backward pass
__global__ void loss_backward_kernel(
    const float* grad_loss, const float* pred_diagram,
    const float* target_diagram, float* grad_pred,
    int n_pairs, int n_target);

// Data augmentation preserving topology
__global__ void augmentation_kernel(
    float* data, int n, int dim,
    float noise_scale, uint64_t seed);

// Regularizer forward
__global__ void regularizer_forward_kernel(
    const float* features, const float* betti_target,
    float* loss, int n_features);

}
```

Benchmarks:

```cpp
struct RegularizerBenchmark {
    double cpu_betti_ms, gpu_betti_ms, speedup_betti;
    double cpu_augment_ms, gpu_augment_ms, speedup_augment;
};
RegularizerBenchmark benchmarkRegularizer(int num_pairs, int feature_dim);

struct AugmentationBenchmark {
    double cpu_time_ms, gpu_time_ms, speedup;
};
AugmentationBenchmark benchmarkAugmentation(
    int num_samples, int feature_dim, int num_augs);
```


## Regularization components

### PersistentDropout

Dropout that preserves features with high persistence scores during training.
`persistence_aware=True` uses feature importance (from gradients) to derive
per-feature keep probabilities.

### TopologyPreservingDropout

Dropout variant that preserves Betti numbers and connectivity structure.
Requires an optional adjacency matrix for connectivity preservation.

### PersistentBatchNorm

Batch normalization weighted by persistence scores. Features with higher
persistence have stronger influence on the normalization statistics.

### MorseRegularizer

Regularizer for Morse theory properties of a function. Penalizes critical
points with low persistence and enlaces Morse inequalities.

### BettiConstraintLayer

Constrains the Betti numbers of learned representations by applying a
topological penalty during training.

### TopologicalSmoothness

Encourages smoothness of features with respect to topological structure,
penalizing features that change rapidly across persistent features.

### HomotopyRegularizer

Penalizes the homotopy difference between current and target topology,
useful for training topology-preserving transformations.


## FAQ

**What batch size works well for topology loss?**

The losses are computed per-batch element, so any batch size is fine. Larger batches (a few hundred samples) give more stable Betti number estimates.

**Does TopologyLoss require a GPU?**

No -- pure PyTorch fallbacks are used when CUDA is unavailable. The GPU kernels in `pynerve.regularization.gpu` accelerate Wasserstein and bottleneck computations for large diagrams (thousands of points), but everything runs on CPU without modification.

**Should I pretrain before adding topological regularization?**

Not required, but recommended. Start with `wasserstein_weight=1.0` and low `betti_weight` / `complexity_weight` (~0.01), then ramp up after the model produces reasonable diagrams (typically 10--50 epochs depending on dataset size).
