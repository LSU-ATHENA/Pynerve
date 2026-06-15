# Neural network

Convolutional and pooling layers for persistence diagrams, image-based diagram
representations, and SIMD/CUDA-accelerated activation functions. All layers
are PyTorch `nn.Module` subclasses.

```python
import torch
from pynerve.nn import (
    DiagramConv1D, DiagramConvNet, DiagramPooling,
    DiagramDeepSet, DiagramMultiHeadAttention,
    DiagramTransformerBlock,
)

# 1D convolution over sorted persistence pairs
diagrams = torch.randn(4, 100, 3)   # [batch, n_pairs, birth/death/dim]
features = torch.randn(4, 100, 1)   # [batch, n_pairs, in_channels]

conv = DiagramConv1D(in_channels=1, out_channels=16, kernel_size=5)
output = conv(diagrams, features)    # [batch, out_channels, n_pairs]

# Pooling along the diagram dimension
pool = DiagramPooling(pool_size=2, stride=2)
pooled = pool(output)                # [batch, out_channels, n_pairs/2]
```


## Python API (pynerve.nn)

### DiagramConv1D

1D convolution along the sorted-persistence axis of a diagram, with optional
persistence-based gating.

```python
conv = DiagramConv1D(
    in_channels=1,       # features per diagram point
    out_channels=16,
    kernel_size=5,       # convolution window
    stride=1,
    use_persistence_weighting=True,  # gate by (death - birth)
)

output = conv(diagram, features)  # [batch, out_channels, n_pairs]
```

### DiagramConvNet

Stack of multiple `DiagramConv1D` + `DiagramPooling` layers.

```python
convnet = DiagramConvNet(
    in_channels=1,
    hidden_channels=[16, 32, 64],
    kernel_sizes=[5, 3, 3],
    pool_sizes=[2, 2, 2],
)

output = convnet(diagram, features)  # [batch, 64, n_pairs/8]
```

### DiagramPooling

Max or average pooling along the diagram's pair dimension.

```python
pool = DiagramPooling(
    pool_size=2,
    stride=2,
    mode="max",           # "max" | "average"
)
```

### DiagramDeepSet

Permutation-invariant processing via sum-pooling over diagram points.

```python
deepset = DiagramDeepSet(
    in_channels=3,
    out_channels=64,
    hidden_channels=32,
)
output = deepset(diagram)  # [batch, 64]
```

### DiagramMultiHeadAttention

Multi-head self-attention over diagram points, using persistence as a bias.

```python
attn = DiagramMultiHeadAttention(
    d_model=64,
    n_heads=4,
    dropout=0.1,
)
output = attn(features)  # [batch, n_pairs, d_model]
```

### DiagramTransformerBlock

Full transformer block: self-attention + feed-forward + layer norm.

```python
block = DiagramTransformerBlock(
    d_model=64,
    n_heads=4,
    dim_feedforward=256,
    dropout=0.1,
)
output = block(features)  # [batch, n_pairs, d_model]
```


## C++ API

See [diagram_conv.md](diagram_conv.md) for DiagramConv1D and DiagramConv2D.
See [image_ops.md](image_ops.md) for PersistenceImageLayer, LandscapeLayer, and DiagramVectorizer.
See [simd.md](simd.md) for AVX-512 activation functions.
See [gpu.md](gpu.md) for CUDA activation kernels.


## Complexity

DiagramConv1D has time complexity O(B * P * C_out * C_in * K) per batch and space complexity O(B * C_out * P) per batch. DiagramConv2D has time complexity O(B * H * W * C_in * C_out * K_h * K_w) per batch and space complexity O(B * C_out * H' * W') per batch. DiagramPooling has time complexity O(B * C * P) per batch and space complexity O(B * C * P/stride) per batch. DiagramDeepSet has time complexity O(B * P * C * H) per batch and space complexity O(B * C) per batch. DiagramMultiHeadAttention has time complexity O(B * P^2 * d_model) per batch and space complexity O(B * P * d_model) per batch. SIMD activations have time complexity O(N / 8) per batch and O(1) in-place space complexity.

B = batch, P = pairs, C = channels, K = kernel size. H, W = image dimensions.


### Practical guidance

**When to use each layer:**
- `DiagramConv1D`: Variable-length diagrams, local patterns along sorted pairs
- `DiagramConvNet`: Hierarchical feature extraction
- `DiagramDeepSet`: Permutation-invariant, when pair order doesn't matter
- `DiagramMultiHeadAttention`: Capturing long-range dependencies between pairs
- `DiagramTransformerBlock`: Full transformer on diagram features

**Persistence weighting:** When enabled, each output channel is multiplied by
sigmoid(death - birth), focusing attention on long-lived topological features.



## Layer internals

### Persistence weighting mechanism

When `use_persistence_weighting=True`, each output channel is element-wise multiplied by:

```
weight = sigmoid((death - birth) / temperature)
```

where `temperature` is a learnable parameter (default: 1.0). This focuses the convolution on long-lived topological features.

```python
# Custom temperature
conv = DiagramConv1D(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    use_persistence_weighting=True,
    persistence_temperature=0.5,  # sharper gating
)
```

### DeepSet aggregation

```
output = sum_i phi(features_i)

where phi is an MLP: Linear -> ReLU -> Linear
Summation over pairs ensures permutation invariance.
```

### Padding strategies for variable-length diagrams

```python
from pynerve.nn import pad_diagrams, mask_diagrams

# Pad to max pairs in batch
padded = pad_diagrams(diagram_batch)  # adds zero-padding
mask = mask_diagrams(diagram_batch)   # boolean mask for valid pairs

# The mask is used internally to exclude padded positions
# from attention, pooling, and convolution operations
```

## Practical training tips

1. **Learning rate**: Start with 1e-3 for `DiagramDeepSet` and 1e-4 for `DiagramMultiHeadAttention`. The attention layer benefits from a lower LR due to the softmax normalization.
2. **Batch size**: Diagram convolutions benefit from larger batches (32-128) due to the parallel pair processing. Attention layers are memory-constrained; reduce batch size if OOM.
3. **Normalization**: Apply layer normalization after each DiagramConv1D or attention layer. The `DiagramTransformerBlock` includes built-in layer norm.
4. **Dropout**: Use 0.1-0.3 dropout in attention layers. Higher dropout for diagrams with <100 pairs.

## Performance benchmarks

```python
from pynerve.validation import benchmark_nn_layer

for n_pairs in [100, 500, 1000]:
    bm = benchmark_nn_layer(
        layer_type="diagram_conv1d",
        batch_size=32,
        n_pairs=n_pairs,
        in_channels=16,
        out_channels=64,
    )
    print(f"n_pairs={n_pairs}: {bm.mean_time_ms:.2f}ms "
          f"({bm.throughput:.0f} pairs/s)")
```


## FAQ

**Q: Can I use DiagramConv1D with 2D persistence images?**
A: DiagramConv1D operates on the sorted-persistence axis of diagrams. For persistence images, use DiagramConv2D which applies 2D convolution on the rasterized birth-death plane.

**Q: How does DiagramDeepSet handle variable-length diagrams in a batch?**
A: It pads all diagrams to the length of the largest diagram in the batch, then applies a mask to exclude padding from the sum pooling. The output is independent of the padding length.

**Q: What is the memory complexity of DiagramMultiHeadAttention?**
A: O(B * P^2 * d_model) for the attention matrix. For diagrams with P > 1000, the attention matrix becomes large (~1M elements for P=1000). Use `use_lsh=True` for LSH attention which reduces to O(B * P * log P * d_model).


### Cross-references

- `pynerve.torch`: Underlying PyTorch integration
- `pynerve.ml`: ML pipeline
- `pynerve.encoders`: Encoder architectures
- `pynerve.validation.benchmarks`: NN layer benchmarks
