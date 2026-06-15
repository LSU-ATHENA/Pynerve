## Diagram message passing (GNN)

Graph neural network layers that operate on persistence diagrams as sets of
points, using message passing between diagram points.

Located in `src/ml/nn/diagram_message_passing_layers.inl`:

```cpp
class GPUMultiScaleDiagramConv {
    // Multi-scale convolution over diagram points
    // Aggregates features across scales using attention-based
    // message passing between nearby birth-death pairs
};
```

These power the `pynerve.nn.DiagramDeepSet` and
`pynerve.nn.DiagramMultiHeadAttention` layers.

```python
from pynerve.nn import DiagramDeepSet, DiagramMultiHeadAttention

# DeepSets: permutation-invariant diagram processing
layer = DiagramDeepSet(in_channels=3, out_channels=64)
features = layer(diagram_batch)  # [batch, n_pairs, 64]

# Multi-head attention over diagram points
attn = DiagramMultiHeadAttention(
    d_model=64, n_heads=4, dropout=0.1
)
output = attn(features)  # [batch, n_pairs, 64]
```

### GPUMultiScaleDiagramConv

Multi-scale convolutional layer that aggregates features across scales
using attention-based message passing between nearby birth-death pairs.

```cpp
template <typename T>
class GPUMultiScaleDiagramConv {
    Config config;
public:
    struct Config {
        int in_channels;
        int out_channels;
        std::vector<float> scales;  // neighborhood radii
        int n_heads = 4;
    };

    explicit GPUMultiScaleDiagramConv(Config cfg);

    std::vector<T> forward(
        std::span<const T> diagram,   // [batch, n_pairs, 3]
        std::span<const T> features,  // [batch, n_pairs, in_channels]
        size_t batch_size,
        size_t n_pairs) const;
};
```

### Architectures

DiagramDeepSet is permutation invariant, accepts variable-length input, and has complexity O(B * P * C * H). DiagramMultiHeadAttention is not permutation invariant, accepts variable-length input, and has complexity O(B * P^2 * d_model). GPUMultiScaleDiagramConv is not permutation invariant, requires fixed n_pairs, and has complexity O(B * P * S * C).


## Message passing formula

For each diagram point i, the message passing update is:

```
h_i' = sigma( W * sum_{j in N(i)} alpha(h_i, h_j) * h_j )
```

where the neighborhood N(i) includes points within scale distance in the birth-death plane:

```
N(i) = { j : ||(b_i, d_i) - (b_j, d_j)|| < scale }
```

### Multi-scale aggregation

```python
from pynerve.nn import DiagramMultiHeadAttention, DiagramDeepSet

# Multi-scale: aggregate at multiple neighborhood radii
layer = GPUMultiScaleDiagramConv(Config(
    in_channels=16,
    out_channels=64,
    scales=[0.1, 0.5, 1.0],  # three scales
    n_heads=4,
))

# Each scale produces features, concatenated along channel dim
output = layer.forward(diagram, features)
# output shape: [batch, n_pairs, 64 * len(scales)]
```

## Attention-based message passing

The `DiagramMultiHeadAttention` layer computes attention between all pairs of diagram points:

```python
attention = DiagramMultiHeadAttention(d_model=64, n_heads=4)
output = attention(features)

# Internally:
# Q = features @ W_Q  (queries)
# K = features @ W_K  (keys)
# V = features @ W_V  (values)
# attn = softmax(Q @ K^T / sqrt(d_k))
# output = attn @ V
```

## DeepSets for permutation invariance

The `DiagramDeepSet` layer processes diagram points as a set:

```python
deepset = DiagramDeepSet(in_channels=3, out_channels=64)
output = deepset(diagram)  # [batch, n_pairs, 64]

# Invariance: output is same regardless of pair order
# Proof: sum pooling over pairs is permutation-invariant
```

## Practical considerations

DeepSet is permutation invariant and accepts variable-length input, but uses sum-only pair interactions with O(P) memory, making it best for global features. Attention is not permutation invariant (positional encoding helps), accepts variable-length input, uses all-pairs interactions with O(P^2) memory, making it best for long-range dependencies. Multi-scale Conv is not permutation invariant, does not accept variable-length input (requires padding), uses local neighborhood interactions with O(P * S) memory, making it best for local patterns.

### Memory optimization for attention

```python
# For diagrams with >1000 pairs, use LSH attention
attention = DiagramMultiHeadAttention(
    d_model=64,
    n_heads=4,
    use_lsh=True,       # Locality-Sensitive Hashing
    lsh_num_hashes=8,   # reduces O(P^2) to O(P log P)
)
```


## FAQ

**Q: When should I use diagram message passing vs graph GNN?**
A: Diagram message passing operates on points in the birth-death plane, capturing topological relationships. Graph GNN operates on the underlying point cloud graph. Use diagram message passing when you want the model to learn from persistence structure; use graph GNN for the original spatial structure.

**Q: How do I handle diagrams with different numbers of pairs in a batch?**
A: Use zero-padding with a mask. The `DiagramMultiHeadAttention` and `GPUMultiScaleDiagramConv` layers accept a mask tensor indicating valid pairs. Masked positions are excluded from attention/aggregation.

**Q: Can I combine diagram message passing with image-based vectorization?**
A: Yes. Extract both message-passing features and persistence image features, then fuse them via concatenation or attention-based fusion. The `HybridEncoder` in `pynerve.encoders` supports this pattern.


### Cross-references

- `pynerve.nn`: Neural network layers
- `pynerve.ml`: ML module overview
- `pynerve.graphs.gnn`: Graph neural network layers
- `pynerve.encoders`: Encoder fusion of diagram features
