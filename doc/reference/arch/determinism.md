# Determinism System

### Contract Architecture

The determinism system ensures bitwise reproducibility through:

1. **Seeded RNG** (`DeterministicRNG`): `std::mt19937_64` seeded from a thread-local seed, producing deterministic sequences for shuffling, sampling, and tie-breaking
2. **Canonical ordering**: Simplices are sorted by (filtration value, dimension, vertex list) before any reduction
3. **GPU fixed-tree reductions**: No atomics in reduction kernels -- uses warp shuffle + shared memory reduction trees
4. **Binned MPI accumulators**: Deterministic across process counts via quantized accumulation

### Determinism Levels

At `NONE`, no determinism guarantees are made and the library may use atomics and floating-point reassociation. `BASIC` provides thread seeding, canonical filtration ordering, and deterministic CPU reduction, though the GPU path may still use atomics. `STRICT` ensures full bitwise reproducibility with fixed-tree GPU reductions, no fast-math compilation flags, and deterministic MPI reduction. `AUDIT` includes everything in STRICT plus checksum validation, intermediate result recording, and a fail-on-non-deterministic mode.

### Setting the Seed

```cpp
#include <nerve/determinism.hpp>

nerve::determinism::seed(42);           // set thread-local seed
uint64_t s = nerve::determinism::get_seed();
uint64_t next = nerve::determinism::next_seed();  // MT19937-64 value
```

### DeterminismContract (Core API)

```cpp
#include <nerve/core/determinism_contract.hpp>

auto contract = nerve::core::DeterminismEnforcer::createContract(
    nerve::core::DeterminismLevel::STRICT,
    "my_computation"
);
contract.setRngSeed(42);

// RAII context
{
    nerve::core::DeterminismContext ctx(contract);
    // All operations within this scope are deterministic
}
```

### GPU Determinism

```cpp
// Device-side: fixed-tree reduction (no atomics)
__device__ double val = /* ... */;
double sum = nerve::determinism::blockReduceSum<256>(val);
```

- Compile flag `--fmad=false` disables fused multiply-add reassociation
- Compile flag `--prec-div=true` enables precise division
- Compile flag `--prec-sqrt=true` enables precise square root
- Compile flag `--ftz=false` flushes denormals to zero (disabled for reproducibility)
- Define `NERVE_GPU_DETERMINISM=1` to enforce RFA (Reduction Fusion Algorithm) for GPU-to-GPU reproducibility

### MPI Determinism

```cpp
double sendbuf[256], recvbuf[256];
nerve::determinism::deterministic_reduce(sendbuf, recvbuf, 256, /* root */ 0, MPI_COMM_WORLD);
nerve::determinism::deterministic_allreduce(sendbuf, recvbuf, 256, MPI_COMM_WORLD);
```

- `MPI_SUM` for fixed process count: zero overhead, fully deterministic
- Binned accumulation for cross-count reproducibility: partitions values into quantization bins before reduction


[Back to Architecture Index](index.md)
