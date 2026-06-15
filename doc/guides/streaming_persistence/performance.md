# Performance tuning for streaming

### Chunk size selection

The chunk size affects memory, accuracy, and overlap cost. A chunk of 100 points uses tens of kilobytes per chunk with low accuracy and high overlap cost, best for real-time sensor data. A chunk of 500 points uses a few megabytes per chunk with medium accuracy and medium overlap cost, best for interactive visualization. A chunk of 1000 points uses several megabytes per chunk with good accuracy and low overlap cost, best for batch processing. A chunk of 5000 points uses hundreds of megabytes per chunk with excellent accuracy and very low overlap cost, best for offline analysis.

### Memory vs accuracy trade-off

```
Larger chunks = better persistence accuracy but more memory
Smaller chunks = lower memory but features may span multiple chunks

Rule of thumb: chunk_size should be >= 10x the expected feature scale
(in terms of number of points needed to represent the largest feature).
```

### Overlap handling choice

The overlap handling mode trades off memory, smoothness, and continuity. `concat` has no memory overhead, low temporal smoothness, and produces independent windows. `mean` has 2x memory overhead for the last window, high temporal smoothness, and produces smooth transitions. `max` has 2x memory overhead for the last window, medium temporal smoothness, and produces conservative estimates.

### GPU streaming throughput

Measured on NVIDIA H100 with FP16 distance (Tensor Core), throughput varies by chunk size. With a chunk size of 100, throughput is 50,000 points/sec on a single GPU and 350,000 points/sec across 8 GPUs. With a chunk size of 500, throughput is 200,000 points/sec on a single GPU and 1,400,000 points/sec across 8 GPUs. With a chunk size of 1000, throughput is 350,000 points/sec on a single GPU and 2,400,000 points/sec across 8 GPUs. With a chunk size of 5000, throughput is 500,000 points/sec on a single GPU and 3,500,000 points/sec across 8 GPUs.

### I/O overlap

When reading from HDF5, I/O and computation overlap:

When reading from HDF5, I/O (reading the next chunk from disk) overlaps with computation (persistence homology of the current chunk). In a timeline view, Read chunk 0 (0-1 s) is followed by Read chunk 1 (1-2 s) and Compute chunk 0 (0-1.3 s) running in parallel during the 1-1.3 s window. Similarly, Read chunk 2 and Compute chunk 1 overlap. This pipelining achieves up to 1.5x effective throughput versus sequential read-then-compute.

Effective throughput with I/O overlap is up to 1.5x vs sequential read+compute.

### Producer-consumer tuning

Key tuning parameters for producer-consumer streaming. `max_buffered_chunks`: higher values use more memory but reduce producer blocking; 3-5 is typically optimal. `num_workers` (lockfree): higher values increase GPU concurrency; set equal to available GPUs. Async queue size: higher values provide a smoother pipeline; should be at least 2x `max_buffered_chunks`.

### Error handling in streaming

```python
from pynerve import StreamingPersistence, NerveError

sp = StreamingPersistence(chunk_size=1000, max_dim=2)

async for result in sp.stream_compute("data.h5"):
    try:
        betti = result.betti_numbers
        # Process result
    except NerveError as e:
        # Log error and continue with next chunk
        print(f"Chunk failed: {e}")
        continue
```

Streaming errors and their recovery. `NerveIOError` indicates a file read failure; check the file path and permissions. `NerveMemoryError` means the chunk is too large for the memory budget; reduce chunk_size. `GPUMemoryError` indicates GPU out-of-memory on a chunk; set use_gpu=False or reduce chunk_size. `ConvergenceError` means the PH engine timed out; increase max_iterations.

### Streaming with MPI

For distributed streaming across cluster nodes:

```bash
# Each node processes independent chunks
# MPI Allgather merges results at sync points
mpirun -np 16 --hostfile hosts.txt \
    python stream_persistence_mpi.py
```

```python
# stream_persistence_mpi.py
from pynerve import StreamingPersistence

sp = StreamingPersistence(chunk_size=1000, use_gpu=True, max_dim=2)

# MPI rank determines file offset
# rank 0: chunks 0-99, rank 1: chunks 100-199, etc.
async for result in sp.stream_compute(
    "shared_data.h5",
    mpi_rank=rank,
    mpi_world_size=world_size,
):
    # Results are already MPI-merged
    print(result.betti_numbers)
```

[Back to index](index.md)
