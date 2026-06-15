# Performance tuning for distributed execution

### When to use distributed mode

The choice of execution mode depends on the problem size and available hardware. For fewer than 100K points on a single node with no GPU, use CPU single-node (1x speedup). For 100K to 1M points on a single node with 1-8 GPUs, use Multi-GPU with CUDA_HYBRID (2-8x speedup). For 1M to 10M points across 2-4 nodes with 1-4 GPUs per node, use GPU+MPI (10-40x speedup). For 10M to 100M points across 4-16 nodes with 1-8 GPUs per node, use GPU+MPI with sparse mode (40-320x speedup). For over 100M points across 16 or more nodes with 8 GPUs each, use distributed with streaming (100x+ speedup).

### Communication overhead breakdown

Measured on 2x AMD EPYC 7763 nodes, 4x H100 per node, NVLink + InfiniBand HDR, the communication overhead breaks down as follows. Allgatherv for point coordinates takes 50 ms (2% of total). Local reduction takes 2 s (80% of total). Pivot exchange via MPI P2P takes 300 ms (12% of total). Cross-node allreduce takes 100 ms (4% of total). Checkpoint I/O takes 50 ms (2% of total).

### Scaling efficiency

Weak scaling (fixed n per node) achieves 1.00 on 1 node (baseline), 1.95 on 2 nodes, 3.80 on 4 nodes, 7.20 on 8 nodes, and 13.50 on 16 nodes. Strong scaling (fixed total n) achieves 1.00 on 1 node, 1.85 on 2 nodes, 3.40 on 4 nodes, 5.80 on 8 nodes, and 8.50 on 16 nodes. Strong scaling degrades when per-rank columns become too few for efficient GPU utilization.

### Memory considerations for distributed mode

Each rank's peak memory consumption depends on the component. Point coordinates use O(n/p * dim) memory, which for n=100K, p=4 amounts to under a megabyte. The dense distance matrix requires O(n^2 / p) memory, reaching tens of gigabytes for n=100K, p=4. The sparse distance matrix (at 1% density) uses O(n * k / p), which is hundreds of megabytes for n=100K, p=4. The boundary matrix uses O(simplices / p), around one gigabyte for d=2. Column reduction buffers need O(largest_column / p), roughly hundreds of megabytes. Per-rank memory decreases linearly with p for dense mode and sub-linearly for sparse mode (landmark overlap).

### Recommended MPI configuration

```bash
# Optimal for HPC cluster with InfiniBand + NVLink
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_ALGO=Ring

mpirun -np 16 \
    --hostfile hosts.txt \
    --map-by ppr:8:node:pe=8 \
    --bind-to core \
    --mca btl_tcp_if_include eth0 \
    python distributed_script.py
```

### MPI message size tuning

The following message types are used in MPI communication. Pivot requests consist of 3 uint64 values using MPI_UNSIGNED_LONG_LONG datatype with tag TAG_BOUNDARY_CHUNK=1. Column data contains 10K entries of MPI_UNSIGNED_LONG_LONG with the same tag. Reduction results contain 1K pairs using a custom struct with tag TAG_REDUCTION_RESULT=2. Work steal requests are a single MPI_INT with tag TAG_WORK_STEAL=3. Checkpoint data holds 10K columns of serialized bytes with tag TAG_CHECKPOINT=4.

### Scaling limits

Distributed persistence scaling is limited by:

1. **Communication overhead**: Pivot exchange requires point-to-point MPI. At > 64 ranks, the number of pivot requests grows as O(p^2).
2. **Load imbalance**: Column sizes vary by dimension. H0 columns are large; H2 columns are small. Round-robin distribution mitigates this.
3. **GPU memory**: Each GPU must hold its shard of the distance matrix. For dense n=500K, each of 8 GPUs needs roughly a few tens of gigabytes.

Recommended maximum: 128 ranks for n < 10M, 256 ranks for n < 100M.

### Distributed checkpointing for long-running jobs

For computations that run for hours:

```python
result = pynerve.distributed_persistence(
    points,
    max_dim=2,
    checkpoint_path="/scratch/nerve_ckpt/",
    checkpoint_interval_sec=600,  # every 10 minutes
)

# On restart:
# Pynerve checks checkpoint_path/rank_{id}/ for existing state
# Resumes from last consistent checkpoint
```

### Network topology considerations

The network topology affects performance based on latency and bandwidth. NVLink for GPU-GPU communication offers 0.5 us latency and hundreds of gigabytes per second bandwidth, best for within-node GPU exchange. InfiniBand HDR provides 1 us latency and hundreds of gigabytes per second bandwidth, best for inter-node MPI communication. InfiniBand EDR delivers 1.5 us latency and hundreds of gigabytes per second bandwidth, suitable for legacy inter-node setups. Ethernet 100 GbE has 5 us latency and tens of gigabytes per second bandwidth, good for low-cost clusters. Ethernet 1 GbE has 100 us latency and hundreds of megabytes per second bandwidth, for development only.

### Handling heterogeneous clusters

When nodes have different GPU counts or capabilities, use `Communicator splitting`:

```python
# Each rank detects its own capability
if gpu_count >= 4:
    # Fast nodes: compute more columns
    rank_weight = 2.0
else:
    # Slow nodes: compute fewer columns
    rank_weight = 0.5

result = pynerve.distributed_persistence(
    points,
    max_dim=2,
    # weight adjusts column distribution
    rank_weight=rank_weight,
)
```

### Fault tolerance tips

1. **Checkpoint frequently**: Set `checkpoint_interval_sec` to 1-5 minutes for long jobs
2. **Use shared filesystem**: Checkpoints must be visible to all ranks on restart
3. **Test restart**: Run a short job with forced kill to validate restart path
4. **Monitor NCCL timeouts**: Set `NCCL_TIMEOUT` for long-running reductions
5. **Pin MPI ranks to NUMA domains**: Use `--map-by socket` for best memory locality

<- [Distributed Computing Overview](index.md)
