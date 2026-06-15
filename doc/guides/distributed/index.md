# Distributed Computation

Scale persistence to multi-node clusters using MPI for rank coordination, NCCL for GPU collectives, and NVSHMEM for direct GPU-GPU communication.

```python
# Single-node multi-GPU (auto-distributed via CUDA_HYBRID backend)
import pynerve
import numpy as np

points = np.random.randn(500000, 3)
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    backend=pynerve.PersistenceBackend.CUDA_HYBRID,
)
```

For multi-node MPI execution:

```python
# Distributed persistence -- wrap in MPI launcher
# $ mpirun -np 8 --hostfile hosts.txt python script.py
import pynerve

points = load_my_data()  # each rank loads its shard
result = pynerve.distributed_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
)
```

## Sections

| Section | Description |
|---------|-------------|
| [MPI initialization and rank discovery](mpi_init.md) | Lazy MPI init, communicator splitting, rank discovery |
| [Distributed distance matrix](distance_matrix.md) | Allgatherv-based distribution, column partitioning |
| [Column distribution for parallel reduction](column_distribution.md) | Sharded boundary matrix, distribution strategies |
| [Pivot exchange across ranks](pivot_exchange.md) | Point-to-point MPI, non-blocking prefetch |
| [CUDA-aware MPI](cuda_aware_mpi.md) | Zero-copy GPU buffers, auto fallback |
| [Multi-GPU + MPI hybrid](multi_gpu_mpi.md) | NCCL intra-node, MPI inter-node |
| [NVSHMEM bridge](nvshmem.md) | One-sided GPU-GPU communication |
| [Work-stealing scheduler](work_stealing.md) | Dynamic load balancing, backoff strategy |
| [Launch](launch.md) | MPI launch commands, environment variables |
| [Checkpoint/restart](checkpoint.md) | Fault tolerance, checkpoint format |
| [API reference](api_reference.md) | Function signatures and parameters |
| [Performance tuning](performance_tuning.md) | Scaling, memory, topology, fault tolerance |
| [FAQ](faq.md) | Frequently asked questions |
