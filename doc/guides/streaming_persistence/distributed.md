# Distributed streaming

MPI-based multi-node streaming distributes the stream across cluster nodes:

```cpp
// src/streaming/lockfree/streaming_mpi_ops.cpp
// src/streaming/mpi_cuda/  (MPI+CUDA hybrid)
```

Each rank processes a subset of the stream. Results are merged via MPI Allgather at synchronization points.

```bash
mpirun -np 4 python stream_worker.py
```

### MPI distributed streaming protocol

```
Setup:
1. MPI_Init with MPI_THREAD_MULTIPLE
2. Each rank opens the same data file (shared filesystem or MPI-IO)
3. Compute per-rank chunk range:
   total_chunks = ceil(total_points / chunk_size)
   chunks_per_rank = total_chunks / world_size
   my_start = rank * chunks_per_rank
   my_end = (rank == world_size - 1) ? total_chunks : my_start + chunks_per_rank

Execution:
4. Each rank processes chunks [my_start, my_end):
   - Read chunk from file
   - Compute persistence
   - Store local pairs in per-rank buffer

Synchronization:
5. After every N chunks (default 10), ranks synchronize:
   - MPI_Allgather(local_pair_counts, 1, MPI_INT,
                    all_counts, 1, MPI_INT, comm)
   - Compute total_pairs = sum(all_counts)
   - MPI_Allgatherv(local_pairs, local_count, pair_type,
                    global_pairs, all_counts, offsets, pair_type, comm)
   - Rank 0 merges and yields global diagram

Finalize:
6. MPI_Barrier
7. MPI_Finalize
```

[Back to index](index.md)
