# GPU cohomology reduction

The GPU reduction pipeline is **deterministic by default** -- no atomic operations are used.

## Warp shuffle reductions

Columns are reduced inside warps using `__shfl_xor_sync` butterfly exchanges:

```
thread 0: pivot column baseline
thread 1: xor with thread 0 -> reduced
thread 2: xor with thread 0 -> reduced
...
```

This avoids global-memory atomics entirely. Each warp independently reduces a distinct set of columns.

## Shared memory tree reduction

For cross-warp column accumulation, a shared-memory tree reduction is used instead of atomicCAS:

```
warp 0 -> shmem[0]   warp 4 -> shmem[4]
warp 1 -> shmem[1]   warp 5 -> shmem[5]
...                  sync; then tree-reduce shmem pairs
```

Fixed reduction order means **bitwise reproducibility** across runs and GPU architectures.

## Apparent pairs

Apparent pairs are detected before reduction via `kernel_apparent_pairs_cuda.cu`, bypassing the full reduction for trivial 0-dimensional pairs. This provides a constant-factor speedup proportional to the number of apparent births.

## Clearing optimization

The clearing optimization in `kernel_clearing_cuda.cu` eliminates columns that are guaranteed to reduce to zero based on pairing information from lower dimensions. This reduces the effective number of columns by 30-70% in practice.

## Hypha scan

`kernel_hypha_scan_cuda.cu` implements a column-skipping optimization: when the remaining unpaired columns are sparse and non-interacting, the kernel scans ahead and skips redundant reduction iterations.


<- [Back to GPU Acceleration index](index.md)
