## Graph neural network layer

Topology-aware message passing convolution layer.

```python
from pynerve.graphs import GraphNeuralLayer

layer = GraphNeuralLayer(
    graph=graph,
    input_dim=64,
    output_dim=128,
)

x = np.random.randn(100, 64)
h = layer.forward(x)

h = layer.graph_convolution(x, weights)

h = layer.graph_attention(
    x,
    queries=Q,
    keys=K,
    values=V,
)

grad = layer.backward(grad_output)
```

### GPU-accelerated GNN

```python
from pynerve.graphs.gpu import benchmark_message_passing
bm = benchmark_message_passing(num_nodes=5000, num_edges=50000, feature_dim=128)

from pynerve.graphs.gpu import benchmark_multi_head_attention
bm = benchmark_multi_head_attention(num_nodes=5000, feature_dim=128, num_heads=8)
```

### Complexity

GNN convolution on CPU costs O(m * d_in * d_out) where d is the feature dimension. On GPU it costs O(m / cores * d) with Tensor Core acceleration.


## Detailed GNN layer API

### GraphNeuralLayer

```cpp
// C++ API
#include <nerve/graphs/gnn_layer.hpp>

class GraphNeuralLayer {
public:
    struct Config {
        int input_dim, output_dim;
        int hidden_dim = 128;
        int n_heads = 4;
        float dropout = 0.1;
        bool use_attention = true;
        bool use_residual = true;
        ActivationType activation = ActivationType::RELU;
    };

    explicit GraphNeuralLayer(const Graph& graph, const Config& config);

    // Forward pass: message passing + update
    std::vector<float> forward(const std::vector<float>& node_features);

    // Graph convolution without attention
    std::vector<float> graph_convolution(
        const std::vector<float>& features,
        const std::vector<float>& weights);

    // Graph attention layer
    std::vector<float> graph_attention(
        const std::vector<float>& features,
        const std::vector<float>& queries,
        const std::vector<float>& keys,
        const std::vector<float>& values);

    // Backward pass
    std::vector<float> backward(const std::vector<float>& grad_output);
};
```

### Message passing formula

```
h_i^{(l+1)} = sigma( W * sum_{j in N(i)} alpha_ij * h_j^{(l)} )
```

where alpha_ij is the attention coefficient (when `use_attention=True`):

```
alpha_ij = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )
```

### GPU benchmarks

```python
from pynerve.graphs.gpu import benchmark_message_passing, benchmark_multi_head_attention

# Message passing scaling
for nodes in [1000, 5000, 10000]:
    bm = benchmark_message_passing(
        num_nodes=nodes,
        num_edges=nodes * 10,
        feature_dim=128,
    )
    print(f"n={nodes}: {bm.gpu_time_ms:.1f}ms (GPU) vs "
          f"{bm.cpu_time_ms:.1f}ms (CPU)")

# Multi-head attention scaling
for heads in [1, 4, 8, 16]:
    bm = benchmark_multi_head_attention(
        num_nodes=5000,
        feature_dim=128,
        num_heads=heads,
    )
    print(f"heads={heads}: {bm.time_ms:.1f}ms")
```

### Training loop

```python
from pynerve.graphs import GraphNeuralLayer
import numpy as np

# Build graph and layer
g = Graph.from_knn(points, k=10)
layer = GraphNeuralLayer(
    graph=g,
    input_dim=64,
    output_dim=32,
    use_attention=True,
)

# Training step
features = np.random.randn(g.num_vertices(), 64)
output = layer.forward(features)
grad = layer.backward(np.random.randn(*output.shape))

# Update weights manually or use optimizer
# Layer exposes weights via get_weights() / set_weights()
```


## Common pitfalls

1. **Graph structure mismatch**: The layer caches the graph adjacency at construction. If edges change, create a new layer or call `update_graph()`.
2. **Attention memory**: Full attention computes O(n^2) pairwise scores. For graphs with >10k nodes, use `use_attention=False` or neighbor sampling.
3. **Deep GNN over-smoothing**: Beyond 3-4 layers, all node representations converge. Use residual connections (`use_residual=True`) and layer normalization.

## FAQ

**Q: What is the advantage of attention-based message passing over standard graph convolution?**
A: Attention weights let the model focus on the most relevant neighbors for each node, improving expressivity. Standard convolution treats all neighbors equally. Attention is especially useful for graphs with heterogeneous or noisy edges.

**Q: How do I handle graphs with varying numbers of neighbors?**
A: The layer handles variable-degree neighbors naturally through the adjacency structure. For mini-batch training, use neighbor sampling to fix the number of sampled neighbors per node.

**Q: Does the layer support edge features?**
A: Yes, pass edge features via `edge_weights` in the forward call. They are incorporated into the message as multiplicative or additive terms alongside the attention coefficients.


### Cross-references

- `pynerve.graphs`: Graphs module overview
- `pynerve.nn`: Diagram-based neural network layers
- `pynerve.ml.message_passing`: Diagram message passing
- `pynerve.spectral.laplacian`: Spectral GNN convolution
