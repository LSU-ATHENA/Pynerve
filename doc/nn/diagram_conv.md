## Diagram convolutional layers

### DiagramConv1D (templated float/double)

```cpp
#include <nerve/nn/diagram_conv.hpp>

namespace nerve::nn {

template <typename T = float>
class DiagramConv1D {
    struct Config {
        int in_channels = 1;
        int out_channels = 1;
        int kernel_size = 5;
        int stride = 1;
        bool use_persistence_weighting = true;
    };

    explicit DiagramConv1D(Config config);

    std::vector<T> forward(
        std::span<const T> diagram,    // [batch, n_pairs, 3]
        std::span<const T> features,   // [batch, n_pairs, in_channels]
        size_t batch_size, size_t n_pairs) const;

    void set_weights(std::span<const T> kernel, std::span<const T> bias);
    size_t output_size(size_t n_pairs) const;
};

}
```

Internally applies `conv1d` followed by `apply_persistence_gate` which
multiplies each output channel by a sigmoid of the persistence value,
focusing attention on long-lived features.

### DiagramConv2D

2D convolution on a rasterized birth-death plane.

```cpp
template <typename T = float>
class DiagramConv2D {
    struct Config {
        int in_channels = 1;
        int out_channels = 1;
        int kernel_h = 3, kernel_w = 3;
        int stride_h = 1, stride_w = 1;
    };

    std::vector<T> forward(
        std::span<const T> input,    // [batch, in_channels, height, width]
        size_t batch_size, size_t height, size_t width) const;
};
```

### DiagramConvNet

Stack of multiple `DiagramConv1D` + pooling layers.

```cpp
template <typename T = float>
class DiagramConvNet {
    struct Config {
        int in_channels = 1;
        std::vector<int> hidden_channels;
        std::vector<int> kernel_sizes;
        std::vector<int> pool_sizes;
    };

    std::vector<T> forward(
        std::span<const T> diagram,
        std::span<const T> features,
        size_t batch_size, size_t n_pairs) const;
};
```

### GPU-accelerated conv1d

CUDA kernel for batched conv1d on diagram features:

```cpp
template <typename T>
__global__ void diagram_conv1d_cuda_kernel(
    const T* diagram, const T* features,
    const T* kernel, const T* bias,
    T* output,
    int batch_size, int n_pairs,
    int in_channels, int out_channels,
    int kernel_size);
```


## Convolution kernel implementation

### DiagramConv1D forward pass

```cpp
template <typename T>
std::vector<T> DiagramConv1D<T>::forward(
    std::span<const T> diagram,
    std::span<const T> features,
    size_t batch_size, size_t n_pairs) const {

    // diagram: [batch, n_pairs, 3] = (birth, death, dim)
    // features: [batch, n_pairs, in_channels]
    // kernel: [out_channels, in_channels, kernel_size]

    size_t out_size = (n_pairs - config.kernel_size) / config.stride + 1;
    std::vector<T> output(batch_size * out_size * config.out_channels);

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < config.out_channels; o++) {
            for (size_t p = 0; p < out_size; p++) {
                T sum = bias[o];
                for (size_t k = 0; k < config.kernel_size; k++) {
                    size_t idx = p * config.stride + k;
                    for (size_t c = 0; c < config.in_channels; c++) {
                        sum += features[b * n_pairs * config.in_channels
                                       + idx * config.in_channels + c]
                             * kernel[o * config.in_channels * config.kernel_size
                                     + c * config.kernel_size + k];
                    }
                }
                // Apply persistence gate
                if (config.use_persistence_weighting) {
                    T persistence = diagram[b * n_pairs * 3 + p * 3 + 1]
                                  - diagram[b * n_pairs * 3 + p * 3];
                    T gate = T(1) / (T(1) + std::exp(-persistence));
                    sum *= gate;
                }
                output[b * out_size * config.out_channels
                      + p * config.out_channels + o] = sum;
            }
        }
    }
    return output;
}
```

### Persistence gate

The persistence gate multiplies each output position by sigmoid(death - birth):

```
gate = 1 / (1 + exp(-persistence))
```

This ensures that positions with high persistence (long-lived features) have a stronger influence on the output, while noise (short-lived features) is suppressed.

## Padding and stride behavior

A kernel size of 3 with stride 1 produces 98 output positions from 100 input pairs and 498 from 500. A kernel size of 5 with stride 1 produces 96 from 100 and 496 from 500. A kernel size of 5 with stride 2 produces 48 from 100 and 248 from 500. A kernel size of 7 with stride 2 produces 47 from 100 and 247 from 500. Use stride > 1 for aggressive downsampling of diagram length.


## DiagramConv2D details

```cpp
template <typename T>
std::vector<T> DiagramConv2D<T>::forward(
    std::span<const T> input,
    size_t batch_size, size_t height, size_t width) const {

    size_t out_h = (height - config.kernel_h) / config.stride_h + 1;
    size_t out_w = (width - config.kernel_w) / config.stride_w + 1;
    std::vector<T> output(
        batch_size * config.out_channels * out_h * out_w);

    // Standard 2D convolution loop
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < config.out_channels; o++) {
            for (size_t i = 0; i < out_h; i++) {
                for (size_t j = 0; j < out_w; j++) {
                    T sum = bias[o];
                    for (size_t c = 0; c < config.in_channels; c++) {
                        for (size_t ki = 0; ki < config.kernel_h; ki++) {
                            for (size_t kj = 0; kj < config.kernel_w; kj++) {
                                sum += input[/* ... */] * kernel[/* ... */];
                            }
                        }
                    }
                    output[/* ... */] = sum;
                }
            }
        }
    }
    return output;
}
```

## FAQ

**Q: How does persistence weighting interact with different kernel sizes?**
A: Larger kernel sizes smooth over more pairs, reducing the effect of per-point persistence weighting. For fine-grained gating, use kernel_size=3 or 5. For coarser aggregation with persistence emphasis, use kernel_size=7 or larger.

**Q: What happens when kernel_size exceeds n_pairs?**
A: The output size becomes zero or negative, and the layer returns an empty tensor. Always ensure n_pairs >= kernel_size, or handle zero-length outputs in downstream layers.

**Q: Can I use DiagramConv2D without rasterizing first?**
A: No. DiagramConv2D requires a rasterized birth-death plane (grid) as input. Use PersistenceImageLayer or a custom rasterization step to produce the grid before passing it to DiagramConv2D.


### Cross-references

- `pynerve.nn`: Neural network overview
- `pynerve.nn.gpu`: GPU kernel implementations
- `pynerve.nn.image_ops`: Image-based diagram representations
- `pynerve.algorithms.vectorization`: Diagram vectorization
