## PersistenceDiagram

Tensor-based persistence diagram class with filtering, statistics, ML
representations, distances, batching, and serialization.

```python
import torch
import pynerve.torch as nt

diagram = nt.vr_persistence(points, max_dim=1)
print(diagram.diagrams.shape)  # [batch, max_pairs, 2]
print(diagram.mask.shape)      # [batch, max_pairs] -- valid pair mask
```


### Python API

```python
class PersistenceDiagram:
    def __init__(self, diagrams, mask=None, num_pairs=None,
                 birth_idx=None, death_idx=None): ...

    @property
    def diagrams(self) -> Tensor: ...
    @property
    def mask(self) -> Tensor: ...

    # Access
    def births(self, apply_mask=True) -> Tensor: ...
    def deaths(self, apply_mask=True) -> Tensor: ...
    def dimensions(self) -> Tensor: ...
    def total_persistence(self, p=2.0) -> Tensor: ...
    def persistence_entropy(self) -> Tensor: ...
    def filter_by_dimension(self, dim) -> PersistenceDiagram: ...

    # Device management
    def to(self, device) -> PersistenceDiagram: ...
    def to_dtype(self, dtype) -> PersistenceDiagram: ...

    # ML representations
    def to_persistence_image(self, resolution=64, sigma=0.1) -> Tensor: ...
    def to_persistence_landscape(self, k=5, resolution=100) -> Tensor: ...
    def to_betti_curve(self, resolution=100) -> Tensor: ...
    def to_vector(self, bins_per_dim=10) -> Tensor: ...

    # Distances
    def wasserstein_distance(self, other, p=2.0) -> float: ...
    def bottleneck_distance(self, other) -> float: ...
    def persistence_kernel(self, other, sigma=1.0) -> float: ...

    # Batching
    @staticmethod
    def batch(diagrams) -> PersistenceDiagram: ...
    def get_batch_item(self, idx) -> PersistenceDiagram: ...

    # Serialization
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict): ...

    # Differentiability
    def requires_grad(self) -> bool: ...
    def set_requires_grad(self, requires_grad): ...
    def backward(self) -> Tensor: ...
    def births_grad(self) -> Tensor: ...
    def deaths_grad(self) -> Tensor: ...
```

### Example: filtering and analysis

```python
# Filter by dimension
h1 = diagram.filter_by_dimension(1)

# Compute statistics
total_pers = diagram.total_persistence()
entropy = diagram.persistence_entropy()

# Convert to ML features
image = diagram.to_persistence_image(resolution=64)
landscape = diagram.to_persistence_landscape(k=5, resolution=100)

# Compute distance between diagrams
d = dgm1.wasserstein_distance(dgm2, p=2.0)
```

### Tensor storage layout

- `diagram_`: `[num_pairs, 2]` float tensor, `[birth, death]`
- `dimensions_`: `[num_pairs]` int64 tensor
- `birth_idx_`, `death_idx_`: index into filtration
- Optional: `birth_simplices_`, `death_simplices_`: vertex indices

### C++ API

```cpp
// Header: src/include/nerve/torch/persistence_diagram.hpp

PersistenceDiagram();
PersistenceDiagram(at::Tensor diagram, at::Tensor dimensions,
                   at::Tensor birth_idx, at::Tensor death_idx);

// Tensor access
at::Tensor diagram() const;
at::Tensor births() const;
at::Tensor deaths() const;
at::Tensor dimensions() const;
at::Tensor persistence_lengths() const;

// Filtering
PersistenceDiagram get_dimension(int64_t dim) const;
PersistenceDiagram get_finite_points() const;
PersistenceDiagram get_infinite_points() const;
PersistenceDiagram threshold(double min_persistence) const;

// Statistics
at::Tensor total_persistence() const;
at::Tensor persistence_variance() const;
at::Tensor mean_persistence_by_dimension() const;
int64_t betti_number(double threshold, int64_t dim) const;
double persistence_entropy() const;
double landscape_norm(int64_t k) const;

// ML representations
at::Tensor to_persistence_image(int64_t resolution=64, ...) const;
at::Tensor to_persistence_landscape(int64_t k=5, ...) const;
at::Tensor to_betti_curve(...) const;
at::Tensor to_vector(int64_t bins_per_dim=10) const;

// Distances
double wasserstein_distance(const PersistenceDiagram& other, double p=2.0) const;
double bottleneck_distance(const PersistenceDiagram& other) const;
double persistence_kernel(const PersistenceDiagram& other, double sigma=1.0) const;

// Operations
void append(const PersistenceDiagram& other);
void sort_by_persistence();
void normalize();

// Batching
static PersistenceDiagram batch(const std::vector<PersistenceDiagram>& diagrams);
PersistenceDiagram get_batch_item(int64_t idx) const;

// Serialization
std::vector<std::pair<std::string, at::Tensor>> state_dict() const;
void load_state_dict(const std::vector<std::pair<std::string, at::Tensor>>&);

// Differentiability
bool requires_grad() const;
void set_requires_grad(bool requires_grad);
void backward() const;
at::Tensor births_grad() const;
at::Tensor deaths_grad() const;
```


## Batching details

```python
# Batch multiple diagrams
dgm_list = [dgm1, dgm2, dgm3]
batched = PersistenceDiagram.batch(dgm_list)

# Access individual items
dgm1_restored = batched.get_batch_item(0)

# Batch shape
print(batched.diagrams.shape)  # [3, max_pairs, 2]
print(batched.mask.shape)      # [3, max_pairs]
```

Batch padding fills shorter diagrams with zeros and sets their mask entries to 0.

## Serialization example

```python
# Save diagram state
state = dgm.state_dict()
torch.save(state, "diagram.pt")

# Load diagram state
new_dgm = PersistenceDiagram()
new_dgm.load_state_dict(torch.load("diagram.pt"))
```

For persistent cross-platform storage, use `pynerve.serialization` (FlatBuffers or Arrow).

## Gradient computation

```python
# Enable gradients
dgm = nt.vr_persistence(points.requires_grad_(True), max_dim=1)
print(f"Requires grad: {dgm.requires_grad()}")

# Backward through a loss
loss = dgm.total_persistence()
loss.backward()

# Access gradients
birth_grads = dgm.births_grad()
death_grads = dgm.deaths_grad()

# The gradient of birth_i is:
# d(total_persistence) / d(birth_i) = -p * (death_i - birth_i)^(p-1)
# The gradient of death_i is:
# d(total_persistence) / d(death_i) = p * (death_i - birth_i)^(p-1)
```

## Filtering examples

```python
# Filter by dimension
h0 = dgm.filter_by_dimension(0)
h1 = dgm.filter_by_dimension(1)

# Filter by persistence
persistent = dgm.threshold(min_persistence=0.5)

# Filter finite vs essential (infinite death)
finite = dgm.get_finite_points()
essential = dgm.get_infinite_points()

# Analyze separately
print(f"Finite pairs: {finite.diagrams.shape[0]}")
print(f"Essential classes: {essential.diagrams.shape[0]}")
```

## C++ performance notes

The C++ `PersistenceDiagram` avoids Python object overhead:

```cpp
// C++: direct tensor access without Python overhead
at::Tensor births = diagram.births();   // no Python call
at::Tensor deaths = diagram.deaths();   // no Python call
at::Tensor lengths = diagram.persistence_lengths();

// Efficient filtering
PersistenceDiagram h0 = diagram.get_dimension(0);
PersistenceDiagram significant = diagram.threshold(0.1);
```

All operations return tensors/objects by value; no reference counting overhead.


## FAQ

**Q: What is the max_pairs dimension?**
A: `max_pairs` is the maximum number of persistence pairs across the batch. It varies with the data: for H0 on n points, expect roughly n pairs. For H1 on n points in R^3, expect O(n^2) pairs. The mask tensor indicates which entries are valid.

**Q: How do I convert a PersistenceDiagram to a numpy array?**
A: Use `diagram.diagrams.detach().cpu().numpy()` for the tensor, or `diagram.to_persistence_image().numpy()` for the image representation.

**Q: Can I use PersistenceDiagram with torch.nn.DataParallel?**
A: Yes. The `batch` method splits diagrams across devices. Each device gets a subset of diagrams. After computation, use `get_batch_item` to retrieve individual results.


### Cross-references

- `pynerve.torch`: PyTorch integration overview
- `pynerve.torch.autograd`: Autograd implementation
- `pynerve.torch.ml`: ML operations on diagrams
- `pynerve.nn`: Neural network layers using diagrams
- `pynerve.serialization`: Diagram serialization
