## GPU encoder kernels

`src/encoders/encoder_gpu_kernels.cu`:

```cpp
__global__ void encoder_forward_kernel(
    const float* input, float* output,
    const float* weights, const float* biases,
    int batch_size, int input_dim, int output_dim,
    ActivationType activation);

__global__ void encoder_backward_kernel(
    const float* grad_output, const float* input,
    float* grad_weights, float* grad_input,
    int batch_size, int input_dim, int output_dim);

__global__ void encoder_fused_kernel(
    const float* input, float* output,
    const float* weights, const float* biases,
    int batch_size, int input_dim, int output_dim,
    ActivationType activation);
```

### Tensor core support

```cpp
namespace nerve::encoders::tensorcore {

class TensorCoreMLPEncoder {
    TensorCoreMLPEncoder(int input_dim, const std::vector<int>& hidden_dims,
                          int output_dim);
    void encode(const std::vector<float>& input, std::vector<float>& output,
                int batch_size);
};

class CUDNNTopologicalEncoder {
    CUDNNTopologicalEncoder(int h, int w, int channels,
                            int filters, int filter_size);
    void encode(const std::vector<float>& input, std::vector<float>& output);
};

class MixedPrecisionEncoder {
    MixedPrecisionEncoder();
    float scaleLoss(float loss);
    void unscaleGradients(float* gradients, int n);
    bool checkForInfNan(const float* gradients, int n);
    void updateLossScale(bool found_inf);
};

}
```

### Benchmark utilities

```cpp
struct TensorCoreBenchmark {
    double fp32_time_ms, fp16_time_ms, speedup;
    int batch_size, input_dim, output_dim;
};
TensorCoreBenchmark benchmarkTensorCore(int batch_size, int input_dim, int output_dim);
```

### Usage

```cpp
#include <nerve/encoders/gpu_encoders.hpp>

TensorCoreMLPEncoder encoder(128, {256, 128}, 32);
std::vector<float> input(batch_size * 128);
std::vector<float> output(batch_size * 32);
encoder.encode(input, output, batch_size);
// Uses Tensor Core MMA instructions for matrix multiply-accumulate
```


## Fused kernel pipeline

The encoder fuses multiple operations into a single kernel to reduce launch overhead:

```cpp
// Single fused kernel: linear + batch_norm + ReLU
__global__ void encoder_fused_kernel(
    const float* input, float* output,
    const float* weights, const float* biases,
    const float* bn_mean, const float* bn_std,
    int batch_size, int input_dim, int output_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_dim) return;

    int b = idx / output_dim;
    int o = idx % output_dim;

    // Linear: sum_i input[b][i] * weights[o][i] + bias[o]
    float sum = biases[o];
    for (int i = 0; i < input_dim; i++) {
        sum += input[b * input_dim + i] * weights[o * input_dim + i];
    }

    // Batch norm: (sum - mean) / std
    sum = (sum - bn_mean[o]) * bn_std[o];

    // ReLU
    output[idx] = max(sum, 0.0f);
}
```

Fusion reduces kernel launch overhead by 3x (3 kernels -> 1 kernel) and improves L1/L2 cache utilization.

## Mixed precision training loop

```cpp
#include <nerve/encoders/gpu_encoders.hpp>

MixedPrecisionEncoder amp;
amp.updateLossScale(false);  // no inf/nan found, scale stays

float loss = compute_loss(model(input));
loss = amp.scaleLoss(loss);

loss.backward();
amp.unscaleGradients(model.gradients(), model.num_params());

if (amp.checkForInfNan(model.gradients(), model.num_params())) {
    amp.updateLossScale(true);  // found inf, reduce scale
    optimizer.zero_grad();
    // skip this batch
}

optimizer.step();
```

## Performance tuning

### Grid/block sizing

- **MLP forward**: block size 256, grid size ceil(n / 256), 1 thread per output element.
- **CNN forward**: block size 128, grid size ceil(n / 128), more shared memory per block.
- **Autoencoder**: block size 256, grid size ceil(batch * code_dim / 256), tensor core tiles 16x16.
- **Topological**: block size 64, grid size ceil(n_pairs / 64), memory-bound, smaller blocks.

### Tensor core tile layout

```cpp
// WMMA 16x16x16 tiles for matrix multiply
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                       half, nvcuda::wmma::row_major> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                       half, nvcuda::wmma::col_major> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                       float> c_frag;
```

Tile dimensions must be multiples of 16 for Tensor Core MMA instructions on H100+.

### Multi-GPU data parallelism

```python
from pynerve.encoders.gpu import MultiGPUEncoder

mgenc = MultiGPUEncoder(num_gpus=4)
mgenc.scatter_data(data)     # split batch across GPUs
mgenc.encode_parallel()      # all GPUs compute simultaneously
result = mgenc.gather_output()  # merge results
```


## Benchmark results

```python
from pynerve.encoders.gpu import benchmarkTensorCore

for batch in [64, 256, 1024]:
    bm = benchmarkTensorCore(batch, input_dim=512, output_dim=128)
    print(f"Batch {batch}: FP32={bm.fp32_time_ms:.2f}ms, "
          f"FP16={bm.fp16_time_ms:.2f}ms, "
          f"speedup={bm.speedup:.1f}x")
```

Expected speedups on H100:
- FP16 vs FP32: 2-3x for matrix-bound layers
- FP8 vs FP32: 4-6x with Blackwell TMA
- Tensor Core vs CUDA cores: 4-8x for matmul-dominated encoders


## Common pitfalls

1. **Tensor Core alignment**: Matrix dimensions must be multiples of 16 for WMMA. Pad input/output dimensions if necessary.
2. **FP8 gradient scaling**: FP8 gradients require loss scaling to avoid underflow. Use `MixedPrecisionEncoder::scaleLoss` and `unscaleGradients`.
3. **Multi-GPU load imbalance**: Uneven scatter creates stragglers. Use `num_gpus` that divides batch_size evenly, or enable dynamic load balancing.
4. **CPU-GPU sync**: Avoid `cudaDeviceSynchronize()` between encoder layers. Chain kernels with CUDA streams for async execution.

## FAQ

**Q: Do I need an H100 or newer GPU for Tensor Core support?**
A: Tensor Cores are available from Volta (V100) onward. H100 adds FP8 and higher-throughput MMA. Blackwell further improves FP8 TMA. The encoder falls back to CUDA cores on older hardware.

**Q: How do I choose between FP16 and FP32?**
A: Use FP16 for throughput-bound matmul-heavy encoders (MLP, Autoencoder) -- expect 2--3x speedup. Use FP32 for memory-bound or precision-sensitive encoders (Topological, small CNNs). The `MixedPrecisionEncoder` helper handles loss scaling and gradient unscaling automatically.

**Q: What batch size should I use for multi-GPU encoding?**
A: Choose `num_gpus` that divides `batch_size` evenly. Uneven splits create stragglers. For dynamic workloads, enable load balancing via `MultiGPUEncoder.set_dynamic_load_balancing(true)`.

**Q: How do I avoid CPU-GPU synchronization overhead?**
A: Chain encoder kernels on the same CUDA stream and avoid `cudaDeviceSynchronize()` between layers. Use CUDA events for timing instead of blocking synchronization.


### Cross-references

- `pynerve.encoders`: Encoders overview
- `pynerve.cuda`: CUDA infrastructure
- `pynerve.encoders.simd`: SIMD encoder operations
- `pynerve.optimization.gpu`: GPU optimizer for encoder training
