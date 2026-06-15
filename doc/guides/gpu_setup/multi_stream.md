# Multi-stream execution

Pynerve maintains a pool of CUDA streams for concurrent kernel execution:

Compute stream 0 handles reduction kernels. Compute stream 1 handles distance matrix computation, overlapping with reduction. The data transfer stream manages host-to-device transfers. The collective stream handles NCCL operations. The filtration stream handles filtration sorting and edge extraction.

Concurrent execution allows distance computation for the next filtration level to overlap with current-level reduction.

| Stream | Role | Typical kernel |
|--------|------|---------------|
| Stream 2 (data transfer) | Host-to-device copies | `cudaMemcpyAsync` for next-level points |
| Stream 1 (compute distance) | Distance matrix computation | `compute_distance(level_i)` |
| Stream 0 (reduce) | Matrix reduction | `reduce(level_i)` via warp-shuffle or shared-memory tree |

A barrier synchronizes all streams before the next iteration. This pipelining allows data transfer for level i+1 to overlap with distance computation for level i, and distance for level i to overlap with reduction for level i−1.

## Stream synchronization

```cpp
cudaStreamWaitEvent(compute_stream, distance_done_event, 0);
// Guarantees distance matrix is ready before reduction starts
```

Events are recorded after each kernel launch in the dependency chain. The CUDA graph mode captures the entire DAG and replays it without host-side re-scheduling.


<- [Back to GPU Acceleration index](index.md)
