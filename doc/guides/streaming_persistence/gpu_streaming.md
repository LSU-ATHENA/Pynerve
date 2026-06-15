# GPU streaming scatter/gather

Multi-GPU point scatter/gather distributes streaming persistence across devices:

```python
# Each GPU processes a separate window stream
# NCCL collectives merge results periodically
sp = StreamingPersistence(
    chunk_size=1000,
    use_gpu=True,  # distributes across available GPUs
    max_dim=2,
)
```

GPU streaming implementation:
- `src/streaming/gpu/streaming_persistence_cuda.cu` -- GPU kernel for windowed reduction
- `src/streaming/gpu/windowed_ph_cuda.cu` -- sliding window on GPU
- `src/streaming/gpu/gpu_multi_stream_ops.cpp` -- multi-GPU coordination
- Multi-stream CUDA execution overlaps data transfer with reduction

### GPU scatter/gather protocol

```
Scatter phase:
1. Input chunk is divided into sub-chunks (one per GPU)
2. Each sub-chunk is transferred to its assigned GPU via cudaMemcpy
3. Each GPU computes persistence independently on its sub-chunk
4. NCCL AllGather shares results across GPUs

Gather phase:
5. Persistence diagrams from all GPUs are collected on the host
6. Diagrams are merged: pairs are sorted by (dimension, birth, death)
7. Merged result is yielded to the async iterator

Overlap:
- While GPU N computes, data for GPU N+1 is being transferred
- NCCL communication overlaps with GPU kernel execution
```

### CUDA stream management for streaming

```cpp
// src/streaming/gpu/gpu_multi_stream_ops.cpp
struct StreamingStreams {
    cudaStream_t transfer_stream;   // Host <-> Device transfers
    cudaStream_t compute_stream;    // Persistence kernels
    cudaStream_t collect_stream;    // NCCL collectives

    cudaEvent_t transfer_done;
    cudaEvent_t compute_done;
};

// Execution timeline for one chunk:
// transfer: H2D copy of chunk N+1   ||||||
// compute:  persistence of chunk N      ||||||
// collect:  NCCL gather of chunk N-1         |||||
```

[Back to index](index.md)
