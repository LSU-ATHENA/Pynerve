# CUDA Graph Capture

Pynerve uses CUDA graphs to reduce kernel launch overhead for repeated computations:

```cpp
// src/gpu/cuda_graph_capture.cpp
cudaGraph_t graph;
cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal);

// Launch dependent kernels
kernel_apparent_pairs<<<grid, block, 0, compute_stream>>>(...);
kernel_clearing<<<grid, block, 0, compute_stream>>>(...);
kernel_reduction<<<grid, block, 0, compute_stream>>>(...);

cudaStreamEndCapture(compute_stream, &graph);

// Instantiate and replay
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, compute_stream);

// Replay with updated parameters
cudaGraphExecUpdate(instance, graph, &update_info, &update_error);
```

Graph capture is used for:
- Distance computation (tiled + threshold)
- Matrix reduction (apparent pairs + reduction)
- NCCL collective encapsulation

Graph replay reduces launch latency by 30-70% for repeated operations.


[Back to Architecture Index](index.md)
