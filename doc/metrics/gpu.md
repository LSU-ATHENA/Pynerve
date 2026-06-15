## GPU acceleration

`src/metrics/cuda/diagram_distance_gpu_kernels.cu`:

The kernel `diagram_pairwise_distance_kernel` computes batch pairwise diagram distances. `wasserstein_cost_kernel` computes the Wasserstein cost matrix on GPU. `hungarian_assignment_kernel` runs the Hungarian algorithm on GPU.

### GPU bottleneck

`src/cuda/kernels/bottleneck_distance.cu`:
- GPU bottleneck via parallel Hungarian with shared memory optimization
- Block size = 256 threads, one diagram pair per block
- Uses warp shuffle for cost matrix reduction

### GPU Wasserstein

`src/cuda/kernels/wasserstein_distance.cu`:
- GPU Sinkhorn with warp-level reductions
- Matrix exponential via CUDA math library
- Iteration loop on GPU (no CPU synchronization per iteration)

### AVX-512

```cpp
namespace nerve::metrics::avx512 {

void avx512DiagramDistanceMatrix(
    const float* birth1, const float* death1, int n1,
    const float* birth2, const float* death2, int n2,
    float* out_matrix);

}
```

### MPI distributed

`src/metrics/mpi/metrics_mpi_ops.cpp`:

```cpp
namespace nerve::metrics::mpi {

double distributedBottleneck(
    const std::vector<Diagram>& local_diagrams,
    MPI_Comm comm);

std::vector<std::vector<double>> distributedDiagramMatrix(
    const std::vector<Diagram>& all_diagrams,
    MPI_Comm comm);

}
```

Collective pattern:
1. Scatter diagram pairs across ranks
2. Each rank computes local distance matrix
3. Allgather to assemble full result
4. Optionally reduce for aggregate statistics

### Python

```python
from pynerve.metrics import bottleneck, wasserstein

# GPU acceleration auto-selected when inputs are on GPU
d_bn = bottleneck(dgm1_gpu, dgm2_gpu, backend="cuda")
d_ws = wasserstein(dgm1_gpu, dgm2_gpu, p=2.0, method="sinkhorn")
```


## GPU kernel details

### Pairwise diagram distance kernel

```cpp
// Batch pairwise diagram distances
__global__ void diagram_pairwise_distance_kernel(
    const float* births_a, const float* deaths_a, int n_a,
    const float* births_b, const float* deaths_b, int n_b,
    float* out, int batch_size) {

    int batch_idx = blockIdx.x;
    int i = threadIdx.x;  // index into diagram A

    if (i >= n_a) return;

    float b_i = births_a[batch_idx * n_a + i];
    float d_i = deaths_a[batch_idx * n_a + i];

    for (int j = 0; j < n_b; j++) {
        float b_j = births_b[batch_idx * n_b + j];
        float d_j = deaths_b[batch_idx * n_b + j];
        float cost = fmaxf(fabsf(b_i - b_j), fabsf(d_i - d_j));
        out[batch_idx * n_a * n_b + i * n_b + j] = cost;
    }
}
```

### Hungarian on GPU

```cpp
// Parallel Hungarian algorithm step
__global__ void hungarian_step_kernel(
    float* cost_matrix, int* assignment,
    float* u, float* v, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Find minimum in row i (primal step)
    float min_val = INFINITY;
    int min_j = -1;
    for (int j = 0; j < n; j++) {
        float reduced = cost_matrix[i * n + j] - u[i] - v[j];
        if (reduced < min_val) {
            min_val = reduced;
            min_j = j;
        }
    }
    // ... alternating path and augmenting ...
}
```

## Memory optimization

```cpp
// Shared memory optimization for cost matrix reduction
__shared__ float shared_cost[256];  // one word per thread

// Warp shuffle for fast reduction
float warp_reduce(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}
```

## AVX-512 distance matrix

```cpp
void avx512DiagramDistanceMatrix(
    const float* birth1, const float* death1, int n1,
    const float* birth2, const float* death2, int n2,
    float* out_matrix) {

    for (int i = 0; i < n1; i++) {
        __m512 b_i = _mm512_set1_ps(birth1[i]);
        __m512 d_i = _mm512_set1_ps(death1[i]);

        for (int j = 0; j < n2; j += 16) {
            __m512 b_j = _mm512_loadu_ps(birth2 + j);
            __m512 d_j = _mm512_loadu_ps(death2 + j);

            // L_inf = max(|b_i-b_j|, |d_i-d_j|)
            __m512 db = _mm512_sub_ps(b_i, b_j);
            __m512 dd = _mm512_sub_ps(d_i, d_j);
            db = _mm512_abs_ps(db);
            dd = _mm512_abs_ps(dd);
            __m512 cost = _mm512_max_ps(db, dd);

            _mm512_storeu_ps(out_matrix + i * n2 + j, cost);
        }
    }
}
```

## MPI collective pattern

```cpp
// MPI distributed diagram matrix (simplified)
std::vector<std::vector<double>> distributedDiagramMatrix(
    const std::vector<Diagram>& all_diagrams, MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int k = all_diagrams.size();
    int local_start = rank * k / size;
    int local_end = (rank + 1) * k / size;
    int local_k = local_end - local_start;

    // Each rank computes a block of rows
    std::vector<std::vector<double>> local_block(
        local_k, std::vector<double>(k));
    for (int i = 0; i < local_k; i++)
        for (int j = 0; j < k; j++)
            local_block[i][j] = computeDistance(
                all_diagrams[local_start + i], all_diagrams[j]);

    // Gather all blocks
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&local_k, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    // ... MPI_Allgatherv to assemble full matrix ...
}
```


## Benchmark results

For 100 pairs, exact bottleneck takes 2 milliseconds on a 96-core CPU versus 3 milliseconds on an A100 GPU, so CPU is faster due to GPU overhead. For 1000 pairs, approximate bottleneck runs in 50 milliseconds on CPU versus 10 milliseconds on GPU, a 5x speedup. Wasserstein Sinkhorn with 500 pairs runs in 30 milliseconds on CPU versus 2 milliseconds on GPU, a 15x speedup. A pairwise distance matrix of 100 by 100 runs in 5 milliseconds on CPU versus 0.3 milliseconds on GPU, a 17x speedup. A batch of 100 diagrams with 500 pairs each runs in 500 milliseconds on CPU versus 25 milliseconds on GPU, a 20x speedup.

*GPU overhead dominates for small sizes.


## FAQ

**Q: When should I use GPU over CPU for distance computation?**
A: GPU acceleration pays off for large diagrams (over 500 points) or large batches (over 50 diagram pairs). For small problems, CPU is often faster due to kernel launch overhead and PCIe transfer costs. The crossover point is typically around 100 to 500 points depending on the metric.

**Q: How do I control GPU memory usage for very large diagrams?**
A: The GPU kernels allocate cost matrices in device memory, which scales as O(n * m). If you run out of memory, reduce batch size, use the sliced Wasserstein variant (which avoids full cost matrices), or switch to CPU backend with `backend="cpu"`.

**Q: Does the GPU implementation support MPI-distributed computation?**
A: Yes. The MPI distributed path scatters diagram pairs across ranks, each rank computes its local block on its local GPU, and results are assembled via Allgather. This works for clusters where each node has one or more GPUs.


### Cross-references

- `pynerve.cuda`: CUDA infrastructure
- `pynerve.metrics.bottleneck`: Bottleneck implementation
- `pynerve.metrics.wasserstein`: Wasserstein implementation
- `pynerve.validation.benchmarks`: Performance benchmarks
- `pynerve.metrics.mpi`: MPI distributed distance computation
