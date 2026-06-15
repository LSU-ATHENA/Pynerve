## Distance kernels (12 kernels)

```cpp
struct DistanceMatrixOptimizer {
    static int compute(const double* points, double* distances,
                       Size n_points, Size point_dim,
                       double max_radius, cudaStream_t stream = 0);

    static int computeBatch(const double* const* pointsBatch,
                            double** distancesBatch,
                            const Size* n_points, Size point_dim,
                            Size batch_size, cudaStream_t stream = 0);

    static int computeFP16(const float* points, float* distances,
                           Size n_points, Size point_dim,
                           float max_radius, cudaStream_t stream = 0);
};
```

Low-level pairwise distance kernels (in `src/cuda/kernels/distance_kernels.cu`):

```cpp
#include <nerve/gpu/distance_kernels.hpp>

namespace nerve::gpu {

cudaError_t launch_pairwise_distance_radius_f32(
    const float* d_points, int points_ld, float* d_out,
    int out_ld, int n_points, int dim, float max_radius,
    void* stream_handle = nullptr);

cudaError_t launch_pairwise_distance_radius_f64(
    const double* d_points, int points_ld, double* d_out,
    int out_ld, int n_points, int dim, double max_radius,
    void* stream_handle = nullptr);

}
```


### Kernel source files

`distance_kernels.cu` handles FP32/FP64 pairwise computation with leading-dimension awareness. `distance_kernels_ext.cu` provides extended batch modes. `distance_kernels_tensorcore.cuh` implements Tensor Core WGMMA matmul. `distance_fasted.cu` provides FastED join-based distance. `distance_tedjoin.cu` implements thresholded Euclidean distance join. `reduction_kernels.cu` handles core warp-shuffle reduction. `reduction_kernels_launcher.cpp` provides host-side launcher dispatch. `gpu_persistence_reduction.cu` is the full persistence reduction pipeline. `gpu_persistence_launcher.cu` handles kernel launch orchestration. `specseq_reduction.cu` implements spectral sequence reduction.


### Distance kernel (FP64, column-major)

```cpp
__global__ void pairwise_distance_f64_kernel(
    const double* __restrict__ points, int ld_points,
    double* __restrict__ distances, int ld_dist,
    int n_points, int dim, double max_radius) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_points || j >= n_points) return;

    double sum = 0.0;
    for (int d = 0; d < dim; ++d) {
        double diff = points[d * ld_points + i] - points[d * ld_points + j];
        sum += diff * diff;
    }
    double dist = sqrt(sum);
    distances[j * ld_dist + i] = dist;
}
```


## Reduction kernels

### MatrixReductionOptimizer

Warp-level optimized boundary matrix reduction with shared-memory caching
and persistent kernels.

```cpp
struct MatrixReductionOptimizer {
    static cudaError_t reduce(
        const uint64_t* boundaryMatrix, uint64_t* columns,
        int n_cols, int n_words_per_col,
        int* pivotColumn, uint64_t* reduced,
        cudaStream_t stream = 0);

    static cudaError_t reduceBatch(
        const uint64_t* const* boundaryMatrices,
        uint64_t** columnsArray,
        const int* n_cols_array, int n_words_per_col,
        int** pivotTables, int batchSize,
        cudaStream_t stream = 0);

    static cudaError_t applyClearing(
        const int2* pairs, int n_pairs,
        uint64_t* columns, int n_cols,
        int n_words_per_col, bool* cleared,
        cudaStream_t stream = 0);
};
```


### Shared-memory tree reduction

Cross-warp column accumulation uses a fixed-order shared-memory tree rather
than atomicCAS, ensuring bitwise reproducibility:

```
Phase 1: Each warp writes its partial result to shared memory

 Warp 0 ──-> [ shmem[0] ]
 Warp 1 ──-> [ shmem[1] ]
 Warp 2 ──-> [ shmem[2] ]
 Warp 3 ──-> [ shmem[3] ]
  ...           ...

Phase 2: Synchronize; tree-reduce pairs in shared memory

 shmem[0] ──╮
             ├──-> pair reduce ──╮
 shmem[1] ──╯                   │
                                ├──-> Final Reduction Result
 shmem[2] ──╮                   │
             ├──-> pair reduce ──╯
 shmem[3] ──╯

Bitwise reproducible  --  fixed reduction tree, no atomicCAS
```

### Warp-level shuffle

Inside each warp, columns are reduced using `__shfl_xor_sync` butterfly
exchanges -- no global-memory atomics at all.


## Occupancy tuning

### GPUArchitecture

Auto-detects GPU compute capability and recommends optimal launch parameters.

```cpp
struct GPUArchitecture {
    int major, minor;
    int computeCapability;
    int multiProcessorCount;
    size_t sharedMemPerBlock;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;

    enum class Family {
        Fermi, Kepler, Maxwell, Pascal, Volta,
        Turing, Ampere, Hopper, Ada, Blackwell, Unknown
    } family;

    static GPUArchitecture detect();

    bool supportsTensorCores() const;
    bool supportsAsyncCopy() const;
    bool supportsCooperativeGroups() const;
    bool supportsMultiInstanceGPU() const;

    int getOptimalTileSize() const;
    int getOptimalBlockSize() const;
    int getOptimalStreamCount() const;
};
```


### Advanced tuning (SM90+)

```cpp
namespace nerve::gpu::advanced {

struct GpuTuningHandle {
    int blockSize = 128;
    int numStages = 2;
    int tileSize = 32;
    bool useWGMMA = false;
    bool useTMA = false;
    bool usePTXOpts = true;
    float measuredTime = 0.0f;
};

GpuTuningHandle tuneForGpu(uint32_t nPoints, uint32_t pointDim, int deviceId);
std::vector<GpuTuningHandle> tuneForMultiGpu(
    uint32_t nPoints, uint32_t pointDim, const std::vector<int>& deviceIds);

struct TileKernelConfig {
    int tileSizeM = 64;
    int tileSizeN = 64;
    int tileSizeK = 16;
    int clusterSize = 1;
    bool useTMA = false;
    bool useTensorCores = false;
    std::string dataType = "fp32";
};

struct Tcgen05Config {
    int shapeM = 128, shapeN = 16, shapeK = 32;
    std::string aFormat = "e4m3", bFormat = "e4m3";
    std::string accumType = "fp32";
    int pipelineDepth = 3;
};

struct Cluster16Config {
    bool useNonPortable = true;
    bool useMulticast = true;
    bool useDistributedL2 = true;
    size_t sharedMemPerBlock = 128 * 1024;
};

}
```

### CudaAutoTuner

Benchmarks multiple kernel configurations and selects the fastest for the
current GPU.

```cpp
class CudaAutoTuner {
public:
    struct KernelConfig {
        dim3 block;
        dim3 grid;
        int sharedMemBytes;
        int tileSize;
        float measuredTimeMs;
    };

    static KernelConfig tuneDistanceMatrix(
        Size n_points, Size point_dim, int numTrials = 10);
    static KernelConfig tuneMatrixReduction(
        int n_cols, int n_words_per_col, int numTrials = 10);
};
```


### Cross-references

- `pynerve.cuda.cuda`: CUDA module overview
- `pynerve.cuda.determinism`: Deterministic reduction kernels
- `pynerve.cuda.streams`: Stream management for concurrent kernels
- `pynerve.cuda.graphs`: CUDA graph capture for repeatable launches
- `pynerve.algebra.distance`: CPU distance kernels
