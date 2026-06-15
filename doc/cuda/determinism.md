## Determinism

- **Fixed-tree reductions**: cross-warp accumulation follows a deterministic
  shared-memory tree with compile-time ordering.
- **No atomics**: all reductions use warp shuffles and shared-memory trees.
- **Compiler flags**: `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`,
  `--ftz=false` enforced via CMake.
- **Warp shuffle**: `__shfl_xor_sync` butterfly pattern with fixed XOR masks.


### Deterministic butterfly reduction inside each warp

```cpp
unsigned mask = 0xffffffff;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(mask, val, offset, 32);
}
```

Results are bitwise identical across runs and GPU architectures for the same
topology.


### Cross-warp tree reduction with compile-time ordering

```cpp
// Cross-warp tree reduction with compile-time ordering
__shared__ uint64_t shmem[32];
int warp_id = threadIdx.x / 32;
int lane    = threadIdx.x % 32;

// Each warp writes its partial result
if (lane == 0) shmem[warp_id] = warp_partial;

__syncthreads();

// Tree reduce in warp 0
if (warp_id == 0) {
    uint64_t val = shmem[lane];
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, 32);
    }
    if (lane == 0) output[col] = val;
}
```


### Compiler flags

Set via CMake for all CUDA compilation:

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --prec-div=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --prec-sqrt=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ftz=false")
```

These flags disable:
- Contracted FMA (fused multiply-add) -- ensures separate multiply and add
  preserve IEEE 754 precision.
- Fast division -- uses IEEE 754 compliant division.
- Fast sqrt -- uses IEEE 754 compliant sqrt.
- Flush-to-zero -- preserves subnormal numbers.


### Determinism contract

```python
from pynerve.validation import DeterminismLevel, DeterminismContract

# Enforce deterministic GPU computation
contract = DeterminismContract(
    level=DeterminismLevel.STRICT,
    seed=42,
    require_bitwise_reproducible=True,
)

result_1 = compute_persistence(points)
result_2 = compute_persistence(points)

assert result_1 == result_2  # bitwise identical
```


### Performance impact

Setting `--fmad=false` makes kernels 10-20% slower than default. `--prec-div=true` adds 5-10% overhead. `--prec-sqrt=true` adds 5-10% overhead. `--ftz=false` has less than 1% impact. Fixed-tree reduction is under 5% slower. Eliminating atomics costs 0-10% depending on the kernel.

Total: approximately 20-30% slower than non-deterministic mode.


### Cross-references

- `pynerve.validation.determinism`: Validation framework
- `pynerve.cuda.kernels`: Reduction kernel implementations
- `pynerve.core.rng`: CPU-side deterministic RNG
