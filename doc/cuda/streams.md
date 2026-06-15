## CUDA streams

`StreamPool` manages multiple streams for overlapping data transfers and
kernel execution.

```cpp
class StreamPool {
public:
    explicit StreamPool(int numStreams = 0);  // 0 = auto-detect

    cudaStream_t getComputeStream(int index = 0);
    cudaStream_t getTransferStream(int index = 0);
    cudaStream_t getH2DStream(int index = 0);
    cudaStream_t getD2HStream(int index = 0);

    void synchronizeAll();
    int getNumStreams() const;
    void waitForComputeTransferPair(int index);
};
```

`numStreams = 0` auto-selects based on GPU architecture:

On Turing (7.5), the recommendation is 2 compute plus 2 transfer streams. On Ampere (8.x), 3 compute plus 2 transfer streams. On Hopper (9.0), 4 compute plus 3 transfer streams. On Blackwell (10.x), 4 compute plus 4 transfer streams.


### Stream assignment

Compute stream 0 handles primary persistence reduction. Compute stream 1 computes the distance matrix for the next filtration level. The H2D stream manages host-to-device transfers. The D2H stream manages device-to-host transfers.

```cpp
// Overlap compute and transfer across streams
StreamPool pool(4);
cudaStream_t compute = pool.getComputeStream(0);
cudaStream_t h2d = pool.getH2DStream(0);

// Ensure compute and transfer streams are ready
pool.waitForComputeTransferPair(0);

// Kernel on compute stream runs concurrently with H2D copy
computeKernel<<<grid, block, 0, compute>>>(d_data);
cudaMemcpyAsync(d_next, h_next, size, cudaMemcpyHostToDevice, h2d);
```


### Stream synchronization patterns

**Double buffering:**
```cpp
for (int batch = 0; batch < num_batches; ++batch) {
    // Transfer next batch
    cudaMemcpyAsync(d_buffer[batch % 2], h_buffer[batch], size,
                    cudaMemcpyHostToDevice, h2d);

    // Compute on previous batch (if available)
    if (batch > 0) {
        computeKernel<<<grid, block, 0, compute>>>(
            d_buffer[(batch - 1) % 2], d_output + (batch - 1) * out_stride);
    }
}
```

**Pipeline overlap:**
```cpp
// Stream 0: compute distance matrix for current frame
// Stream 1: build filtration for next frame
auto s0 = pool.getComputeStream(0);
auto s1 = pool.getComputeStream(1);

for (int frame = 0; frame < num_frames; ++frame) {
    if (frame > 0) cudaStreamWaitEvent(s1, event[frame - 1]);

    distanceKernel<<<grid, block, 0, s0>>>(points[frame], dist[frame]);
    filtrationKernel<<<grid, block, 0, s1>>>(points[frame + 1], filt[frame + 1]);

    cudaEventRecord(event[frame], s0);
}
```


### Python usage

```python
from pynerve.cuda import StreamPool

pool = StreamPool(num_streams=0)  # auto-detect

compute = pool.get_compute_stream(0)
h2d = pool.get_h2d_stream(0)

# Launch kernel on compute stream
kernel<<<grid, block, stream=compute>>>(d_data)

# Concurrent H2D transfer
cudaMemcpyAsync(d_next, h_next, size, h2d)
```


### Performance tips

1. **Minimize stream synchronization**: `cudaStreamSynchronize` is expensive.
   Use events for fine-grained synchronization.

2. **Pinned memory**: Host memory for async transfers must be pinned
   (`cudaMallocHost`). Use the `pinned_memory` allocator.

3. **Stream priorities**: Compute streams should have higher priority
   than transfer streams.

4. **MPS**: On multi-process systems, use MPS to share streams across
   processes.


### Cross-references

- `pynerve.cuda.cuda`: CUDA module overview
- `pynerve.cuda.graphs`: CUDA graphs with stream capture
- `pynerve.cuda.kernels`: Kernels launched on streams
