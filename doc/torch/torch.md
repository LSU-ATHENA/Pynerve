# PyTorch Integration

Differentiable persistent homology, simplex tree operations on GPU, diagram
operations with ATen tensors, TorchScript export, and ML-oriented layers.

```python
import torch
import pynerve.torch as nt

points = torch.randn(4, 32, 3, requires_grad=True)
diagram = nt.vr_persistence(points, max_dim=1, max_radius=2.0)
landscape = diagram.to_persistence_landscape(k=5)
loss = landscape.sum()
loss.backward()
```

## Modules

The following modules are available. [autograd.md](autograd.md) covers differentiable persistence and autograd functions. [simplex_tree.md](simplex_tree.md) covers the TorchScript simplex tree, boundary matrix, and filtration. [persistence_diagram.md](persistence_diagram.md) covers the PersistenceDiagram tensor class. [ml.md](ml.md) covers ML operations including vectorization, kernels, and statistics. [float8.md](float8.md) covers Float8 (FP8) training support.


## Extension module

Pynerve builds a `nerve_torch_internal` extension (pybind11 + Torch C++ API)
when `ENABLE_PYTORCH=ON` and Torch is found. Registers 20+ operators under
`torch.ops.pynerve.*` via `TORCH_LIBRARY`.

### Registered operators

```cpp
// VR complex and persistence
at::Tensor vr_build(Tensor points, float max_radius);
at::Tensor vr_fast(Tensor points, float max_radius, string metric);
at::Tensor vr_persistence(Tensor points, int max_dim);

// Diagram distances
float diagram_wasserstein(Tensor d1, Tensor d2, float p);
float diagram_bottleneck(Tensor d1, Tensor d2);
at::Tensor diagram_landscape(Tensor diagram, int k);
at::Tensor diagram_betti(Tensor diagram, int dim);

// Filtration factories
at::Tensor filtration_distance_matrix(Tensor points, string metric);
at::Tensor filtration_alpha(Tensor points);
at::Tensor filtration_witness(Tensor points, Tensor landmarks);

// Persistence computation
at::Tensor ph_compute(Tensor filtration, int max_dim);
at::Tensor ph_grad(Tensor diagram, int max_dim);
at::Tensor ph_vr(Tensor points, int max_dim, float max_radius);
at::Tensor ph_witness(Tensor points, Tensor landmarks, int max_dim, float max_radius);
at::Tensor ph_alpha(Tensor points, int max_dim);
at::Tensor ph_persistence(Tensor filtration, Tensor values, int max_dim);

// Diagram distances (ph_ prefix)
float ph_wasserstein(Tensor d1, Tensor d2, float p, float q);
float ph_bottleneck(Tensor d1, Tensor d2);
at::Tensor ph_image(Tensor diagram, int resolution_birth, int resolution_death, float sigma);
```


## Neural network layers

```python
# pynerve.torch.nn_layers

class PersistenceLayer(nn.Module):
    def __init__(self, max_dim=1, max_radius=inf, metric="euclidean",
                 preprocessing=None, return_raw=False): ...
    def forward(self, x: Tensor) -> PersistenceDiagram | Tensor: ...

class VectorizationLayer(nn.Module):
    def __init__(self, method="landscape", **params): ...
    def forward(self, x: PersistenceDiagram) -> Tensor: ...

class TopologicalFeatureExtractor(nn.Module):
    def __init__(self, max_dim=1, max_radius=inf, metric="euclidean",
                 preprocessing=None, vectorization="landscape",
                 vectorization_params=None): ...
    def forward(self, x: Tensor) -> Tensor: ...

class DiagramPooling(nn.Module):
    def __init__(self, method="mean", dim=1): ...
    def forward(self, diagrams) -> Tensor: ...

class TopologicalAttention(nn.Module):
    def __init__(self, feature_dim, n_heads=4, dropout=0.1): ...
    def forward(self, features, diagrams=None) -> Tensor: ...

class PersistenceReadout(nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_dims=(128,), dropout=0, activation="relu"): ...
    def forward(self, x) -> Tensor: ...

def make_topo_network(input_dim, hidden_dims=(128,64), max_dim=1,
                      vectorization="landscape", vectorization_params=None,
                      num_classes=10, dropout=0.2) -> nn.Module:
    pass
```


## TorchScript support

Classes in `nerve::torch` use ATen tensor storage and do not depend on Python
objects, making them compatible with `torch.jit.script` and `torch.jit.trace`.

**Export caveats:**
- `SimplexTree::build_vr` uses a C++ loop; traced models must supply
  `max_radius` as a constant
- `PersistenceDiagram::backward()` calls C++ autograd; only trace through
  `compute_persistence_pairs` if the full graph is needed
- Tensor shapes are dynamic (num_pairs depends on the data); TorchScript
  `@torch.jit.export` annotations work for methods with fixed output shapes


## Complexity

Complexity varies by operation. VR persistence for H0 has O(n^2 log n) forward and O(n^2) backward complexity using a union-find merge tree. For H1 and higher, VR persistence has O(m * r) both forward and backward using column reduction. Persistence image has O(p * R^2) both forward and backward, controlled by the resolution parameter. Persistence landscape has O(p * k * R) both forward and backward, determined by layers and resolution. Wasserstein distance has O(p^3) forward complexity using the Hungarian algorithm with no backward pass. Bottleneck distance has O(p^1.5 log p) forward complexity using sparse matching with no backward pass. Betti curve has O(p * R) both forward and backward, linear in the number of pairs. Simplex tree insert has O(k) forward complexity where k is the simplex dimension, with no backward pass. Boundary matrix build has O(m * f) forward complexity where f is the average face count, with no backward pass.


### Common pitfalls

1. **Dynamic shapes**: Persistence output size varies with input. Use
   padding + masking for batch processing.

2. **Essential classes**: Features with infinite death have no death
   gradient. Ensure losses can handle NaN/inf gradients.

3. **TorchScript compatibility**: Not all operations support TorchScript
   export. Test with `torch.jit.script` before deployment.

4. **Memory fragmentation**: Repeated VR computations can fragment GPU
   memory. Use `torch.cuda.empty_cache()` between large computations.



## Extension build details

The `nerve_torch_internal` extension is built when CMake finds Torch:

```cmake
find_package(Torch REQUIRED)
if(Torch_FOUND)
    add_library(nerve_torch_internal SHARED
        src/torch/extension.cpp
        src/torch/persistence_diagram.cpp
        src/torch/simplex_tree.cpp
        src/torch/boundary_matrix.cpp
        src/torch/filtration.cpp
        src/torch/autograd.cpp
    )
    target_link_libraries(nerve_torch_internal ${TORCH_LIBRARIES})
    target_compile_definitions(nerve_torch_internal PRIVATE ENABLE_PYTORCH=1)
endif()
```

Operators are registered via `TORCH_LIBRARY`:

```cpp
TORCH_LIBRARY(nerve, m) {
    m.def("vr_build(Tensor points, float max_radius) -> Tensor");
    m.def("vr_persistence(Tensor points, int max_dim, float max_radius) -> Tensor");
    m.def("diagram_wasserstein(Tensor d1, Tensor d2, float p) -> float");
    m.def("diagram_bottleneck(Tensor d1, Tensor d2) -> float");
    // ... 20+ operators total
}
```

## Neural network layer details

### PersistenceLayer

```python
layer = PersistenceLayer(
    max_dim=1,
    max_radius=float('inf'),
    metric="euclidean",
    preprocessing=None,      # "normalize" | "center" | None
    return_raw=False,        # return padded tensor instead of PersistenceDiagram
)

# With preprocessing
layer = PersistenceLayer(
    max_dim=2,
    preprocessing="normalize",  # auto-normalize input point cloud
)
diagram = layer(points)  # automatically normalized
```

### TopologicalFeatureExtractor

```python
extractor = TopologicalFeatureExtractor(
    max_dim=1,
    max_radius=2.0,
    vectorization="landscape",
    vectorization_params={
        "k": 5,
        "resolution": 100,
    },
)

features = extractor(points)
# Returns fixed-size feature vector
# Suitable as input to standard MLP classifier
```

### Training loop example

```python
import torch
import torch.nn as nn
from pynerve.torch import TopologicalFeatureExtractor

class TopoClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=10):
        super().__init__()
        self.extractor = TopologicalFeatureExtractor(
            max_dim=1,
            max_radius=2.0,
            vectorization="persistence_image",
            vectorization_params={"resolution": 32, "sigma": 0.1},
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        topo_feats = self.extractor(x)
        return self.classifier(topo_feats)

model = TopoClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for points, labels in dataloader:
    logits = model(points)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
```

## Tips for differentiable persistence

1. **Gradient clipping**: Gradients from persistence can be large near merge events. Clip gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
2. **Double precision**: Use `points.double()` for stable gradient computation. Float32 can underflow for very close birth/death values.
3. **Avoid essential classes**: Features with infinite death have no gradient. Either filter them out or design losses that are robust to missing gradients.
4. **Batch with padding**: Use `PersistenceLayer(return_raw=True)` to get padded tensors, then apply masking in your loss function.


## FAQ

**Q: What determines the number of pairs in a PersistenceDiagram?**
A: The number of pairs depends on the data topology, not the input size. VR persistence on n points in R^3 typically produces O(n) pairs for H0 and O(n^2) for H1.

**Q: Can I use pynerve.torch operations with torch.compile?**
A: Yes, most operations support `torch.compile`. Test with `torch.compile(model, mode="reduce-overhead")`. Some operations (SimplexTree methods with dynamic output shapes) may need `torch.jit.export` annotations.

**Q: How do I save and load a PersistenceDiagram?**
A: Use `diagram.state_dict()` and `diagram.load_state_dict()`. For persistent storage, use `pynerve.serialization` which supports FlatBuffers and Arrow formats.

**Q: What is the difference between vr_persistence and ph_vr?**
A: `vr_persistence` builds the VR complex and computes persistence in one call. `ph_vr` is the lower-level registered operator. `vr_persistence` wraps `ph_vr` with additional preprocessing and return type conversion.


### Cross-references

- `pynerve.nn`: Neural network layers
- `pynerve.autodiff`: Differentiable persistence engine
- `pynerve.ml`: ML pipeline using torch operations
- `pynerve.serialization`: Persistence diagram serialization
