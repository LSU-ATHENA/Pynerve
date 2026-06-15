# Deterministic execution overhead

Pynerve is deterministic by default. The overhead of determinism depends on the mechanism. Fixed-tree GPU reduction has no overhead (0%) on single GPU as no atomics are used, making it the same cost as non-deterministic approaches. RFA (reproducible floating-point accumulation) adds 20-30% overhead for cross-GPU reproducibility via sorted global accumulation order. MPI binned accumulation adds 1-12% overhead on multi-node systems by binning columns by pivot before reduction.

### Fixed-tree GPU reduction

GPU reduction uses warp-shuffle (`__shfl_xor_sync`) with a fixed butterfly pattern and shared-memory tree for cross-warp accumulation. No atomic operations are used -- the reduction order is identical every run. **Zero performance overhead** vs non-deterministic atomics.

### RFA overhead

RFA guarantees bitwise reproducibility across different GPU counts and topologies. The overhead comes from:
1. Sorting partial accumulations by global column index
2. Multi-pass reduction to avoid precision loss
3. Synchronization barriers

RFA is enabled through compile-time flags; it is the default in the standard build configuration. To disable RFA for performance, rebuild with determinism disabled.

### MPI binned accumulation

In distributed mode, columns are binned by pivot before cross-rank reduction. This ensures deterministic results even when work is stolen. Overhead ranges from 1% (balanced load) to 12% (highly skewed).

[Back to index](index.md)
