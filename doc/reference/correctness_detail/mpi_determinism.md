# MPI Determinism

For a fixed MPI process count, `MPI_SUM` is used with zero overhead, suitable when the same process count is used across runs. For cross-count reproducibility, binned accumulation quantizes values at O(log n) overhead, allowing consistent results regardless of how many ranks participate.

### Fixed-Process MPI

```cpp
// Deterministic for fixed MPI_COMM_WORLD size
nerve::determinism::deterministic_reduce(send, recv, n, root, comm);
nerve::determinism::deterministic_allreduce(send, recv, n, comm);
```

Internally uses `MPI_SUM` which is deterministic when the set of participants is stable.

### Cross-Count MPI

For reproducibility across different process counts:

```cpp
// Values are quantized into bins before accumulation
// Result is stable regardless of how many ranks participate
nerve::determinism::deterministic_allreduce_binned(send, recv, n, comm);
```

This partitions each floating-point value into a fixed set of quantization bins. The bin boundaries are derived from the seed, ensuring the same result regardless of how the values are distributed across ranks.

### MPI determinism protocol

```
1. Each rank computes local partial sums independently
2. For fixed-count determinism:
   a. MPI_Allreduce with MPI_SUM
   b. Zero overhead, fully deterministic for same process count
3. For cross-count determinism:
   a. Each value is quantized into 2^k bins (k = 8 by default)
   b. Bin index = floor(value * 2^k / range) where range = [min, max]
   c. MPI_Allreduce on bin counts instead of raw values
   d. Result reconstructed from bin centroids
   e. Quantization error: < range / 2^k
4. Binned accumulation overhead: 1-12% depending on data skew
```


[Back to Correctness Index](index.md)
