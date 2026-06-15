## Native C++ autodiff engine

```cpp
// File: src/include/nerve/autodiff/autodiff.hpp
namespace nerve::autodiff {

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<double>& data, const Shape& shape);
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);

    // Arithmetic
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    // Activation functions
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;

    // Reduction
    Tensor sum() const;
    Tensor max() const;
    Tensor min() const;
    Tensor mean() const;

    // Autograd
    void backward();
    void setRequiresGrad(bool requiresGrad);
    bool requiresGrad() const;
    Tensor grad() const;
    void zeroGrad();

    // Data access
    const std::vector<double>& data() const;
    const Shape& shape() const;
    Size size() const;
    Size ndim() const;
    double item() const;  // for scalar tensors
};

// Shape definition
using Shape = std::vector<Size>;
```

Multi-dimensional array with autograd tracking. Designed for use inside the
C++ persistence pipeline without PyTorch dependency.


### Variable wrapper

```cpp
class Variable {
public:
    explicit Variable(const Tensor& data, bool requiresGrad = true);

    Tensor data() const;
    Tensor grad() const;
    bool requiresGrad() const;

    Variable operator+(const Variable& other) const;
    Variable relu() const;
    Variable sigmoid() const;
    Variable sum() const;
    void backward();
    void zeroGrad();
};
```

`Variable` wraps a `Tensor` and tracks operations in the computational graph.
It is the user-facing autograd type.


### ComputationalGraph

```cpp
class ComputationalGraph {
public:
    void addNode(const Tensor& tensor);
    void addEdge(const Tensor& from, const Tensor& to, const Tensor& grad);
    void clear();
    void backward();
    void zeroGrad();
    std::vector<Tensor> getParameters() const;
    void optimize();
};
```

`ComputationalGraph` manages the DAG of tensor operations. Supports:

- **CSE** (common subexpression elimination): deduplicates repeated subgraphs
- **Graph fusion**: merges compatible operations into fused kernels
- **Dead-code elimination**: removes nodes unreachable from loss

```cpp
// Build a simple computation graph
auto w = Tensor::ones({3, 4});  w.setRequiresGrad(true);
auto x = Tensor::ones({4, 1});  x.setRequiresGrad(false);

auto y = w * x;   // matmul
auto z = y.sum();

graph.addNode(w);
graph.addNode(x);
graph.addNode(y);
graph.addNode(z);
graph.addEdge(w, y, /*grad=*/???);
graph.addEdge(x, y, /*grad=*/???);
graph.addEdge(y, z, /*grad=*/???);

graph.backward();
```


### AutoDiffUtils

```cpp
class AutoDiffUtils {
public:
    static bool checkGradient(const std::function<Tensor(const Tensor&)>& func,
                              const Tensor& input, double epsilon = 1e-6);
    static Tensor computeJacobian(const std::function<Tensor(const Tensor&)>& func,
                                  const Tensor& input);
    static Tensor computeHessian(const std::function<Tensor(const Tensor&)>& func,
                                  const Tensor& input);
    static void saveGraph(const ComputationalGraph& graph, const std::string& filename);
    static void profileComputation(const std::function<void()>& computation,
                                    const std::string& name);
};
```


### Autograd graph for topology operations

The topology computation autograd graph (conceptual):

```
points (N, d)
  -> distance_matrix (N, N)     [cdist kernel, differentiable]
    -> filtration (M,)          [sort by edge length, non-diff]
      -> boundary_matrix        [build, non-diff]
        -> reduction            [column ops, non-diff]
          -> pairs (P, 2)       [birth, death, differentiable]
            -> loss
```

Only the distance computation and the pair coordinate extraction carry
gradients. The sort and reduction steps are treated as constants in the
backward pass (straight-through estimator for sort).

**Why this works:**
- The sort operation is non-differentiable, but its effect on persistence
  pairs is locally constant. Small perturbations do not change the sorted
  order, so the straight-through estimator is correct.
- The reduction step is also piecewise-constant; small changes to
  filtration values do not change the column operations.


### Fusion optimization

The `ComputationalGraph::optimize()` method applies:

1. **Op fusion**: Combine consecutive linear ops (no intervening
   non-linearities) into a single matrix multiply

2. **Kernel fusion**: Fuse activation functions into preceding
   operations to avoid separate kernel launches

3. **Memory optimization**: In-place operations where safe

```cpp
graph.optimize();  // Apply all optimizations

// After optimization:
// - Dead code removed
// - Fused ops combined
// - Memory reuse applied
```


### Profiling

```cpp
AutoDiffUtils::profileComputation([&]() {
    auto result = compute_persistence(points);
    auto loss = result.total_persistence();
    loss.backward();
}, "persistence_forward_backward");
```

Outputs timing breakdown to stderr or log file.


### Common pitfalls

1. **Gradient accumulation**: `backward()` accumulates gradients.
   Call `zeroGrad()` between optimization steps.

2. **In-place operations**: NOT supported. Operations like `tensor += 1`
   break the computation graph. Always create new tensors.

3. **Shape mismatches**: Broadcasting follows NumPy rules but is
   restricted to contiguous dimensions.

4. **Memory**: The computation graph is retained until `clear()` or
   destruction. For long-running processes, manage graph lifetime.


## FAQ

**Does the native C++ autodiff engine support GPU tensors?**

No. The `nerve::autodiff::Tensor` class is CPU-only. For GPU-accelerated autograd, use the PyTorch integration (`pynerve.torch`), which dispatches to CUDA kernels automatically.

**How do I prevent gradient accumulation across optimization steps?**

Call `zeroGrad()` on each `Variable` or on the `ComputationalGraph` between training iterations. Gradients accumulate by default when `backward()` is called repeatedly.

**What happens if I use in-place operations?**

In-place operations break the computation graph and produce incorrect gradients. Always create new tensors instead of mutating existing ones.


### Cross-references

- `pynerve.autodiff.gradients`: Differentiable persistence (higher-level)
- `pynerve.torch`: PyTorch-based autograd (preferred for most users)
- `pynerve.metrics`: Loss functions for diagram distances
