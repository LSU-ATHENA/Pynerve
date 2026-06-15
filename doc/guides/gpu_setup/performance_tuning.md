# Performance tuning for GPU

## Launch configuration

Pynerve auto-tunes kernel launch parameters, but manual configuration is available:

```python
from pynerve import PersistenceOptions, PersistenceBackend

opts = PersistenceOptions()
opts.backend = PersistenceBackend.CUDA_HYBRID
# opts.gpu_block_size = 256       # override block size
# opts.gpu_use_tensor_cores = True # force Tensor Core path
# opts.gpu_precision = "fp16"     # force FP16 distance
```

## Common GPU performance issues

Low GPU utilization is typically caused by a block size that is too small and is fixed by increasing the block size (auto-tuned). Memory bandwidth bottlenecks occur when the distance kernel does not use Tensor Cores; using FP16 or BF16 input dtype addresses this. Kernel launch overhead from too many small kernel launches is mitigated by enabling CUDA graph capture (which is automatic). PCIe transfer bottlenecks from frequent host-to-device transfers are resolved by keeping data on GPU. Occupancy limited by register pressure in the reduction kernel can be addressed by reducing max_dim or using a simpler kernel.

## GPU profiler hints

```bash
# Profile with Nsight Systems
nsys profile -o nerve_profile python my_script.py

# Profile with Nsight Compute (kernel-level)
ncu --set full -o nerve_kernel python my_script.py

# Key metrics to check:
# - Achieved Occupancy: should be > 50%
# - SM Efficiency: should be > 80%
# - Memory Throughput: should be > 60% of peak
# - Tensor Core Utilization: should be > 50% for distance kernels
```

## FP16 vs FP32 trade-offs

For distance accuracy, FP32 serves as the 1x baseline. FP16 is good for dimensions under 100. BF16 is good for all dimensions. FP8 is acceptable for dimensions under 50. Memory for the distance matrix scales from 4 bytes per element for FP32 down to 2 bytes for FP16 and BF16, and 1 byte for FP8. Tensor Core throughput roughly doubles from FP32 to FP16 and BF16, and quadruples with FP8. Auto-padding is not needed for FP32, but FP16 and BF16 require it when the dimension is not a multiple of 16, while FP8 requires it when the dimension is not a multiple of 32. The fallback path for lower precisions converts up through FP32.

## Tensor Core precision auto-selection

Pynerve automatically selects the best Tensor Core precision based on:

1. GPU architecture (Volta/Ampere/Hopper/Blackwell)
2. Input dtype (float32/float16/bfloat16)
3. Input dimension (multiple of tile size check)
4. Available memory (FP8 uses 4x less memory)

```python
# Automatic precision selection
# On H100 with float32 input: uses FP8 Tensor Cores
# On A100 with float16 input: uses BF16 Tensor Cores
# On V100 with float32 input: uses FP32 Tensor Cores
result = pynerve.compute_persistence(points, max_dim=2)
```

## GPU kernel launch bounds

Each kernel type has tuned launch bounds. Matrix reduction uses a minimum grid of 1 and maximum of 65535, with a minimum block of 64 and maximum of 512, and shared memory up to tens of kilobytes. Distance kernels in FP32 and FP16 use the same grid range with block sizes from 64 to 256, and shared memory up to tens of kilobytes. The FP8 distance kernel uses similar grid and block bounds with shared memory up to hundreds of kilobytes. Apparent pairs use blocks from 64 to 256 with shared memory up to a few kilobytes. The persistence image kernel uses blocks from 64 to 256 with shared memory up to tens of kilobytes. Spectral kernels use blocks from 64 to 512 with tens of kilobytes of shared memory.

The auto-tuner selects (`grid`, `block`, `shared_mem`) within these bounds for each GPU architecture.

## GPU memory pool configuration

```python
from pynerve import PersistenceOptions, PersistenceBackend

opts = PersistenceOptions()
opts.backend = PersistenceBackend.CUDA_HYBRID
opts.device_memory_limit_mb = 8192  # limit to a few gigabytes
# opts.gpu_pool_initial_size_mb = 1024  # initial pool size of a few gigabytes
# opts.gpu_pool_max_size_mb = 16384    # max pool size of tens of gigabytes
```

When the memory budget is exceeded, columns overflow to pinned host memory automatically.

## GPU kernel launch configuration reference

```cpp
// Example: custom launch configuration for reduction kernels
// src/persistence/cuda/matrix_reduction_launch_cuda.cu

dim3 grid_dim(num_columns / block_size + 1);
dim3 block_dim(256);  // tuned per architecture

cudaMemPrefetchAsync(distance_matrix, bytes, device_id, compute_stream);

kernel_apparent_pairs<<<grid_dim, block_dim, shared_mem, compute_stream>>>(
    d_columns, d_pivots, n_columns
);
CUDA_CHECK(cudaGetLastError());

kernel_clearing<<<grid_dim, block_dim, 0, compute_stream>>>(
    d_columns, d_pairs, n_columns
);
CUDA_CHECK(cudaGetLastError());

kernel_reduction<<<grid_dim, block_dim, shared_mem, compute_stream>>>(
    d_columns, d_pivots, d_pairs, n_columns
);
CUDA_CHECK(cudaGetLastError());
```

## Custom CUDA stream configuration

For advanced users who want to control stream concurrency:

```python
from pynerve import PersistenceOptions, PersistenceBackend

opts = PersistenceOptions()
opts.backend = PersistenceBackend.CUDA_HYBRID
# opts.num_compute_streams = 2     # parallel reduction streams
# opts.transfer_stream_priority = -1  # higher priority for transfers
```


<- [Back to GPU Acceleration index](index.md)
