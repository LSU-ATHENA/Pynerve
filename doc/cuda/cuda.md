# CUDA

GPU-accelerated distance computation, matrix reduction, and kernel launch
optimization for persistence pipelines. Provides auto-tuning, CUDA graphs,
multi-stream execution, and occupancy tuning across NVIDIA architectures
from Turing through Blackwell.

```cpp
#include <nerve/gpu/cuda_dispatch.hpp>  // C++ internal header
```


## Pages

- [kernels.md](kernels.md) -- distance kernels, reduction kernels, kernel organization
- [streams.md](streams.md) -- CUDA streams, concurrent execution
- [graphs.md](graphs.md) -- CUDA graph capture and replay
- [determinism.md](determinism.md) -- fixed-tree reductions, no atomics, compiler flags


## Error handling

CUDA launch status is validated within 3 lines of every `<<<grid, block>>>`
invocation:

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    throw std::runtime_error(
        "Kernel launch failed: " + std::string(cudaGetErrorString(err)));
}
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    throw std::runtime_error(
        "Kernel execution failed: " + std::string(cudaGetErrorString(err)));
}
```

`cuda_dispatch.hpp` wraps this pattern into `CudaProfiler` and `StreamGuard`
utilities.


## Profiling

```cpp
class CudaProfiler {
public:
    static void beginRange(const char* name);
    static void endRange();
    static void markKernel(const char* name, cudaStream_t stream);
    static float getKernelOccupancy(const void* func, int blockSize,
                                    size_t dynamicSmemSize);
    static void printMetrics(cudaStream_t stream = 0);
};
```

Compatible with Nsight Compute ranges for profiling annotation.

```python
# Python profiling
from pynerve.cuda import CudaProfiler

CudaProfiler.begin_range("persistence_pipeline")
diagram = compute_persistence(points)
CudaProfiler.end_range()

# Get kernel occupancy
occ = CudaProfiler.get_kernel_occupancy("distance_kernel", 256, 0)
```


## DeviceArray

RAII wrapper around `cudaMalloc`/`cudaFree` used by all GPU kernels.

```cpp
template <typename T>
class DeviceArray {
    DeviceArray(size_t count);
    ~DeviceArray();
    T* get();
    size_t size() const;
    size_t bytes() const;
    void copyFromHost(const T* host_ptr, size_t count,
                      cudaStream_t stream = 0);
    void copyToHost(T* host_ptr, size_t count,
                    cudaStream_t stream = 0);
    void copyFromDevice(const DeviceArray& src, cudaStream_t stream = 0);
    void setZero(cudaStream_t stream = 0);
};
```

```python
from pynerve.cuda import DeviceArray

arr = DeviceArray('float32', count=1024)
arr.copy_from_host(numpy_array)
# ... kernel operates on arr ...
arr.copy_to_host(result_array)
```


## Architecture detection

`GPUArchitecture::detect()` queries `cudaGetDeviceProperties` and returns
architecture information:

Turing architecture (compute capability 7.5) features first-generation Tensor Cores. Ampere (8.0, 8.6) has third-generation Tensor Cores with FP16 and BF16 support. Hopper (9.0) has fourth-generation Tensor Cores with FP8 and TMA. Ada (8.9) has enhanced third-generation Tensor Cores. Blackwell (10.x) has fifth-generation Tensor Cores with FP4 support.


### Complexity

Distance matrix computation on GPU is O(n^2 * d / cores), achieving 20-50x speedup over CPU. Matrix reduction for 100k columns is O(r * c / blocks), achieving 10-30x speedup. Filtration construction is O(m * log m / cores), achieving 5-10x speedup. CUDA graph replay is O(1) per launch, eliminating launch overhead for roughly 100x improvement in that component.



## FAQ

**Which GPU architecture should I target?** Ampere (compute capability 8.x) is the recommended minimum for persistence pipelines, offering third-generation Tensor Cores and FP16/BF16 support. Hopper (9.0) adds FP8 and TMA for significant performance gains in distance computation. Turing (7.5) is supported but lacks key features for optimal performance.

**When should I use CUDA graphs?** Use CUDA graphs when the same computation repeats with identical grid and block shapes for more than 10 iterations. This captures the launch sequence and eliminates CPU launch overhead, providing up to 100x improvement in launch latency.

**How do I enable deterministic GPU computation?** Determinism is always enabled by default (`DeterminismLevel.STRICT` is the default). This uses fixed-tree reductions, no atomics, and compiler flags (`--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`). AUDIT mode adds checksums at roughly 20-30% overhead.


### Cross-references

- `pynerve.cuda.kernels`: Kernel implementations
- `pynerve.cuda.streams`: Stream management
- `pynerve.cuda.determinism`: Deterministic GPU computation
- `pynerve.cuda.graphs`: CUDA graph capture
- `pynerve.algebra.distance`: CPU distance computation (fallback)
