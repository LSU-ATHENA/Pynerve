# FAQ

**Q: How does Pynerve handle node failures during distributed computation?**
A: Nerve supports checkpointing via the sharded boundary matrix. Each rank checkpoints its local column state to a shared filesystem at configurable intervals. On restart, `restore()` reloads the state and resumes from the last consistent global state.

**Q: What is the recommended number of ranks for a given dataset size?**
A: For datasets under 10 million points, up to 128 ranks is recommended. For datasets under 100 million points, up to 256 ranks is suitable. Beyond that, scaling efficiency is limited by pivot exchange communication overhead, which grows as O(p^2).

**Q: How does Pynerve choose between round-robin, chunked, and work-stealing column distribution?**
A: Round-robin is the default and works well for balanced load with uniform column sizes. Chunked is preferred when locality-sensitive reduction improves performance. Work-stealing activates automatically when load imbalance is detected, with exponential backoff to avoid thundering-herd problems.

**Q: Can I run distributed persistence without InfiniBand or high-speed interconnects?**
A: Yes, but performance will be limited by network bandwidth. Ethernet 100 GbE is functional for development and small-scale runs. Ethernet 1 GbE is suitable only for development and testing. CUDA-aware MPI and NVSHMEM require high-speed interconnects for effective GPU-GPU communication.

**Q: How does checkpointing work across ranks for fault tolerance?**
A: Each rank independently checkpoints its local columns to `checkpoint_path/rank_{id}/`. On restart, rank 0 broadcasts the highest consistent checkpoint ID, and all ranks with that ID load their state. Ranks missing the checkpoint recompute from scratch, then a global barrier synchronizes the restore.

**Q: What is the memory footprint per rank for a typical distributed run?**
A: Memory per rank depends on the component. Point coordinates use under a megabyte. The distance matrix (dense) can reach tens of gigabytes, while the sparse version uses hundreds of megabytes. The boundary matrix requires around one gigabyte for typical parameters. Per-rank memory decreases linearly with the number of ranks for dense mode.

<- [Distributed Computing Overview](index.md)
