# Tensor Cores

Tensor Cores accelerate distance matrix construction with mixed-precision. FP32 is supported from Volta onward as the default, offering 1x baseline throughput. FP16 is supported from Turing onward with roughly 2x the throughput of FP32. BF16 is available from Ampere onward with roughly 2x throughput and better numerical range than FP16. FP8 (E4M3/E5M2) is supported from Hopper onward with roughly 4x throughput versus FP32. FP4 is available from Blackwell onward with roughly 8x throughput versus FP32.

Distance computation is delegated to TCgen05 matrix multiply-accumulate when the input dimension is a multiple of the Tensor Core tile size. Pynerve automatically selects the best precision based on GPU capability and input characteristics.

```python
# GPU dispatches distance computation to Tensor Cores when:
# - input dimension % tile_size == 0 (auto-padded otherwise)
# - dtype is float16/bfloat16/float32
# - GPU architecture supports it
result = pynerve.compute_persistence(points.half(), max_dim=2)
```

## Tensor Core distance pipeline

The Tensor Core distance pipeline in `src/persistence/cuda/tensor_core_cuda.cu`:

1. Points are loaded into `row_major` format
2. Matrix multiply `C = A * B^T` via TCgen05 or wgmma
3. Diagonal extraction: `dist[i,i] = A[i,:] * A[i,:]`
4. Distance assembly: `dist[i,j] = sqrt(C[i,j] - 2*C[i,j] + C[j,j])`
5. Filtration thresholding applied inline

For FP16/BF16, step 2 uses `__hmma` or `wmma` Tensor Core instructions. For FP8, step 2 uses `tcgen05` with per-tensor scaling factors.

## Auto-padding

When the input dimension is not a multiple of the Tensor Core tile size (16 for FP16, 8 for FP8), the input is zero-padded in a temporary buffer. Padding overhead is negligible for dim > 32.


<- [Back to GPU Acceleration index](index.md)
