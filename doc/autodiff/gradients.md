## Autograd persistence function

```python
# pynerve.torch._persistence_vr
pynerve.torch.vr_persistence(
    points: Tensor,           # (batch, n_points, dim)
    max_dim: int = 1,
    max_radius: float = inf,
    metric: str = "euclidean",
    return_simplices: bool = False,
) -> PersistenceDiagram | tuple
```

Returns a `PersistenceDiagram` with `requires_grad=True`. The backward pass
computes gradients of diagram coordinates (birth, death) w.r.t. each input
point, then propagates through the distance computation.


### Forward pass (simplified)

1. Compute pairwise distance matrix from points
2. Build VR filtration up to max_dim
3. Run union-find for H0, standard reduction for H1+
4. Return birth/death tensors

```python
diagram = nt.vr_persistence(points, max_dim=1)
births = diagram.births()      # tensor of birth values
deaths = diagram.deaths()      # tensor of death values
```

### Backward pass

1. Receive gradient on diagram (birth_grad, death_grad)
2. For each persistence pair, sum birth + death gradients
3. Propagate through the merge tree (union-find edges for H0)
4. Map back through distance computation

```python
diagram = nt.vr_persistence(points, max_dim=1)
g = torch.autograd.grad(diagram.births().sum(), points)
# g: gradient of sum(births) w.r.t. points
```


### Limitations

- Only VR filtration is differentiable in the current build
- H0 backward uses merge-tree reconstruction; H1+ backward requires
  solving a linear system at each simplex pair
- Non-differentiable operations: max_radius clamping, essential class
  detection, infinite-death features


## Differentiable persistence (C++)

```cpp
// File: src/include/nerve/differentiable/autodiff_persistence.hpp
namespace nerve::differentiable {

template <std::floating_point T = double>
class AutodiffScalar<T> {
public:
    explicit AutodiffScalar(T value);
    AutodiffScalar(T value, T grad, bool requires_grad);

    const T& value() const noexcept;
    const T& grad() const noexcept;
    bool requiresGradient() const noexcept;

    void setGradient(T grad) noexcept;
    void resetGradient() noexcept;

    AutodiffScalar operator+(const AutodiffScalar& other) const;
    AutodiffScalar operator*(const AutodiffScalar& other) const;
    AutodiffScalar operator-(const AutodiffScalar& other) const;
    AutodiffScalar operator/(const AutodiffScalar& other) const;
    AutodiffScalar operator-() const;

    static AutodiffScalar abs(const AutodiffScalar& x);
    static AutodiffScalar sqrt(const AutodiffScalar& x);
    static AutodiffScalar exp(const AutodiffScalar& x);
    static AutodiffScalar log(const AutodiffScalar& x);
    static AutodiffScalar min(const AutodiffScalar& a, const AutodiffScalar& b);
    static AutodiffScalar max(const AutodiffScalar& a, const AutodiffScalar& b);
};
```

`AutodiffScalar` wraps a floating-point value with gradient tracking. All
arithmetic operations record the gradient computation graph.


### AutodiffPersistenceDiagram

```cpp
template <typename T = double>
struct AutodiffPersistencePair {
    AutodiffScalar<T> birth;
    AutodiffScalar<T> death;
    AutodiffScalar<T> persistence;
    AutodiffScalar<T> dimension;
};

template <typename T = double>
class AutodiffPersistenceDiagram {
public:
    void addPair(const AutodiffPersistencePair<T>& pair);
    const std::vector<AutodiffPersistencePair<T>>& getPairs() const;
    void enableGradients();
    void resetGradients();
    void computeGradients(const std::vector<AutodiffScalar<T>>& loss_gradients);
};
```


### DifferentiablePersistence

```cpp
template <typename T = double>
class DifferentiablePersistence {
public:
    struct ComputationConfig {
        bool enableGradients = true;
        bool track_intermediate_values = false;
        T gradient_tolerance = T(1e-8);
        size_t max_iterations = 1000;
        bool validateGradients = true;
    };

    explicit DifferentiablePersistence(const ComputationConfig& config = {});

    AutodiffPersistenceDiagram<T> compute(
        const std::vector<algebra::Simplex>& complex);

    AutodiffPersistenceDiagram<T> computePersistenceWithGradients(
        const std::vector<algebra::Simplex>& complex,
        std::function<void(const AutodiffPersistenceDiagram<T>&)> gradient_callback = {});

    void computeGradients(const std::vector<algebra::Simplex>& complex,
                          const std::vector<AutodiffScalar<T>>& loss_gradients);

    bool validateGradients(const std::vector<algebra::Simplex>& complex,
                           T finite_difference_tolerance = T(1e-6)) const;
};
```

`DifferentiablePersistence` wraps the standard persistence algorithm with
autodiff tracking. Each birth/death value is an `AutodiffScalar` that records
gradient contributions from downstream loss functions.


### Gradient validation

`validateGradients` compares computed gradients against finite-difference
approximations:

```python
config = DifferentiablePersistence.ComputationConfig()
config.validateGradients = True
config.gradient_tolerance = 1e-8

pers = DifferentiablePersistence(config)
diagram = pers.compute(complex)

# Compute loss
loss = sum(pair.persistence for pair in diagram.getPairs())

# Backward
loss.backward()

# Validate
valid = pers.validateGradients(complex, tolerance=1e-6)
# Prints ratio of analytical to numerical gradient for each pair
```

This is essential for debugging custom loss functions involving topology.


### Python integration

```python
from pynerve.diff import DifferentiablePersistence, PersistenceLoss

# C++ autodiff engine (no PyTorch dependency)
pers = DifferentiablePersistence(max_dim=1)
diagram = pers.compute(simplices, filtration_values)

# Compute custom loss
loss = diagram.total_persistence()
diagram.backward()

# Access gradients
for pair in diagram.pairs:
    print(f"birth grad: {pair.birth.grad}, death grad: {pair.death.grad}")
```


## FAQ

**What filtration types support differentiability?**

Only VR (Vietoris--Rips) filtration is differentiable in the current build. Alpha and cubical filtrations are on the roadmap.

**How do I validate computed gradients?**

Use `validateGradients()` on the `DifferentiablePersistence` object. It compares analytical gradients against finite-difference approximations and reports the ratio for each pair.

**Can I use the C++ autodiff engine without PyTorch?**

Yes. `DifferentiablePersistence` and `AutodiffScalar<T>` are pure C++ templates with no PyTorch dependency. The Python bindings in `pynerve.diff` wrap this engine.


### Cross-references

- `pynerve.autodiff.graph_ops`: Native autodiff engine
- `pynerve.torch.autograd`: PyTorch autograd integration
- `pynerve.metrics`: Loss functions using diagram distances
- `pynerve.nn`: Neural network layers with persistence layers
