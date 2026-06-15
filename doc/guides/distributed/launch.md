# Launch

```bash
# Multi-node, multi-GPU
mpirun -np 16 \
    --hostfile hosts.txt \
    --map-by node:PE=8 \
    --bind-to core \
    python -c "
import pynerve
import numpy as np

# Each rank loads its shard
local_points = np.random.randn(50000, 3)

# Distributed persistence
result = pynerve.distributed_persistence(
    local_points,
    max_dim=2,
    max_radius=2.0,
)

if rank == 0:
    print(f'Found {len(result[\"pairs\"])} pairs')
"
```

### Environment variables

The following environment variables configure MPI and NCCL behavior. `OMPI_MCA_btl` controls the MPI transport (default `^openib` to disable IB for GPU). `NCCL_IB_DISABLE` disables InfiniBand for NCCL (default `0`). `NCCL_P2P_DISABLE` disables P2P for NCCL (default `0`). `NCCL_NET_GDR_LEVEL` sets the GPU Direct RDMA level (default `5`). `NCCL_SHM_DISABLE` disables shared memory for NCCL (default `0`). `MPICH_GPU_SUPPORT_ENABLED` enables CUDA-aware MPI for MPICH (default `0`).

<- [Distributed Computing Overview](index.md)
