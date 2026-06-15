## GPU autodiff kernels

CUDA kernels in `src/autodiff/autodiff_gpu_kernels.cu`:

- `elementwise_gradient_kernel` -- gradient for pointwise ops
- `topological_gradient_kernel` -- gradient propagation through merge tree
- `reduction_gradient_kernel` -- gradient through column reduction

Enabled when `ENABLE_CUDA=ON` at build time. The autograd functions
automatically transfer tensors to GPU and launch CUDA kernels.


### Merge-tree gradient kernel (H0)

The merge-tree gradient kernel walks the union-find hierarchy in parallel
over batches of persistence pairs. Each thread block handles one diagram;
warps within the block process individual pairs.

```cpp
__global__ void topological_gradient_kernel(
    const float* birth_grad,    // [num_pairs] gradient w.r.t. birth
    const float* death_grad,    // [num_pairs] gradient w.r.t. death
    const int* edge_parents,    // [num_edges] union-find parent pointers
    float* point_grad,          // [num_points, dim] output gradient
    int num_pairs,
    int num_points,
    int dim);
```

**Algorithm:**

For each persistence pair (birth_simplex, death_simplex):
1. The death gradient contributes to the death simplex's filtration value
2. The death simplex filtration value depends on the distance between two
   points (for edge = death in H0)
3. Backprop through the distance computation distributes gradient to the
   two points' coordinates


### Column reduction gradient kernel (H1+)

For higher-dimensional persistence, gradients flow through the reduced
boundary matrix. The reduction gradient kernel solves:

```text
For each pair (row, col):
  For each column operation: col += pivot_col
  grad_input[row] += grad_output[col]
  (gradient of addition is sum of gradients)
```

```cpp
__global__ void reduction_gradient_kernel(
    const float* pair_grad,     // [num_pairs] gradient on pairs
    const int* pivot_rows,      // [num_cols] pivot row per column
    const int* reduction_order, // [num_ops] sequence of column ops
    float* simplex_grad,        // [num_simplices] output gradient
    int num_pairs,
    int num_cols);
```


### Elementwise gradient kernel

Computes gradients for pointwise operations (distance computations):

```cpp
__global__ void elementwise_gradient_kernel(
    const float* a,             // first point coordinates
    const float* b,             // second point coordinates
    const float* grad_output,   // incoming gradient
    float* grad_a,              // gradient for point a
    float* grad_b,              // gradient for point b
    int num_pairs,
    int dim);
```

**Mathematical form:**

For Euclidean distance d = sqrt(sum((a_i - b_i)^2)):
- d(d)/d(a_i) = (a_i - b_i) / d
- d(d)/d(b_i) = (b_i - a_i) / d

These are computed elementwise with fused multiply-add.


### Kernel launch pattern

```cpp
void launch_topological_gradient(
    const Tensor& birth_grad,
    const Tensor& death_grad,
    const Tensor& edge_parents,
    Tensor& point_grad,
    int num_pairs, int num_points, int dim,
    cudaStream_t stream) {

    dim3 block(256);  // 256 threads per block
    dim3 grid((num_pairs + 255) / 256);

    topological_gradient_kernel<<<grid, block, 0, stream>>>(
        birth_grad.data_ptr<float>(),
        death_grad.data_ptr<float>(),
        edge_parents.data_ptr<int>(),
        point_grad.data_ptr<float>(),
        num_pairs, num_points, dim
    );

    CHECK_CUDA(cudaGetLastError());
}
```


### Performance characteristics

- **topological_gradient**: 75% occupancy, no shared memory, 32 registers
- **reduction_gradient**: 60% occupancy, 1024 bytes shared memory, 48 registers
- **elementwise_gradient**: 85% occupancy, no shared memory, 16 registers


### Common pitfalls

1. **Coalesced access**: Point coordinates are stored as [N, D]; ensure
   the stride pattern allows coalesced loads.

2. **Bank conflicts**: Shared memory should be padded to avoid bank
   conflicts in warp-level reductions.

3. **Concurrent kernel execution**: Different diagrams within a batch
   can be processed concurrently on different streams.

4. **Mixed precision**: While the kernels support FP16 inputs,
   gradient accumulation should be done in FP32 to avoid underflow.


## FAQ

**What GPU architectures are supported?**

Any CUDA-capable GPU with compute capability 6.0+ (Pascal and later). The kernels are compiled just-in-time for the target architecture when `ENABLE_CUDA=ON`.

**When should I use mixed precision (FP16)?**

FP16 inputs reduce memory bandwidth and can improve throughput. However, gradient accumulation should always be done in FP32 to prevent underflow, especially for persistence pairs with small death values.

**Can multiple diagrams be processed concurrently?**

Yes. Different diagrams within a batch are independent and can be dispatched on separate CUDA streams for concurrent kernel execution.


### Cross-references

- `pynerve.autodiff.gradients`: CPU-side autograd persistence
- `pynerve.cuda.kernels`: General CUDA kernels
- `pynerve.torch.autograd`: PyTorch autograd integration
