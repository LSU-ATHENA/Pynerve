# Memory management

## Device memory limit

Control GPU memory budget via `PersistenceOptions`:

```python
from pynerve import PersistenceOptions, PersistenceBackend

opts = PersistenceOptions()
opts.backend = PersistenceBackend.CUDA_HYBRID
# opts.device_memory_limit_mb = 4096  # limit to a few gigabytes
```

When the budget is exceeded, Pynerve falls back to host-pinned memory for overflow columns. The GPU memory pool uses chunked allocation -- columns are packed into pre-allocated buffers.

## P2P exchange

Cross-GPU column exchange uses `cudaMemcpyPeer` when NVLink P2P is available, avoiding host copies entirely. When P2P is unavailable, data is staged through pinned host memory.

## Unified memory

CuPy-backed operations (`pynerve.cupy_ops`) use CuPy's memory pool. The C++ backend uses `DeviceArray<T>` -- an RAII wrapper around `cudaMalloc`/`cudaFree` with automatic release.

## DeviceMemoryPool

```cpp
// src/gpu/gpu_memory_pool.hpp
class DeviceMemoryPool {
    CudaResult<void*> allocate(size_t bytes, size_t alignment = 256);
    template <typename T> CudaResult<T*> allocateTyped(size_t count);
    CudaResult<void> deallocate(void* ptr, size_t bytes);
    void reset();
    size_t bytesUsed() const;
    size_t bytesTotal() const;
    double utilization() const;
};
```

The pool is initialized with a few gigabytes by default and grows on demand. Allocations are aligned to 256 bytes for optimal memory access.


<- [Back to GPU Acceleration index](index.md)
