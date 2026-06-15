# Floating-Point Assumptions & Numerical Stability

## Floating-Point Assumptions

Pynerve assumes IEEE 754 floating-point arithmetic is used on all platforms.

### Floating-point compilation flags

The following flags are **enforced at compile time** in Release builds. They are NOT optional -- CMake configuration fails if these cannot be satisfied.

### C++ Flags

```
-fno-fast-math
-fno-associative-math
-fno-unsafe-math-optimizations
-ffp-contract=off
```

### CUDA Flags

```
--fmad=false        # Disable fused multiply-add (prevents reassociation)
--prec-div=true     # Precise division
--prec-sqrt=true    # Precise square root
--ftz=false         # Do NOT flush denormals to zero (preserves subnormals)
```

These are set in the root `CMakeLists.txt` (lines 108-124) and `src/cuda/kernels/CMakeLists.txt` (lines 40-42):

```cmake
# CMakeLists.txt (Release build)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -fno-fast-math ... -ffp-contract=off")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} --fmad=false --prec-div=true --prec-sqrt=true")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} --ftz=false")
add_compile_definitions(NERVE_GPU_DETERMINISM=1)
```

### Why no fast-math

`-ffast-math` (and equivalents) enable:
- Reassociation: `(a + b) + c -> a + (b + c)` -- changes results for floating-point
- Reciprocal approximation: `a / b -> a * (1/b)` -- less precise
- Contract operations: FMA contraction -- fused operations may differ from separate

Since persistent homology depends on the ordering of filtration values and matrix reduction involves repeated floating-point comparisons, any reassociation could change the resulting pairing. Pynerve therefore disables ALL fast-math optimizations.

### IEEE 754 compliance

The rounding mode is round-to-nearest-even, the IEEE 754 default, and is never modified. Denormals are preserved via the `--ftz=false` compilation flag, which is critical for bitwise reproducibility. NaN inputs produce an `E20_NUM_NAN` error and are rejected at input validation. Infinite distances are ignored in filtration but are allowed in precomputed matrices. Both positive and negative zero are preserved and are not modified.


## Numerical Stability

- **Internal precision**: All distance and persistence computations use `double` (64-bit) as the canonical type.
- **Float support**: `float` (32-bit) distance path available via explicit SIMD dispatch for performance; results are cast to `double` before reduction.
- **Tolerance**: Default numerical tolerance is `1e-12` for double, `1e-6f` for float (see `config.hpp.in`).
- **Subnormal support**: Denormals are preserved (`--ftz=false`). This is critical for bitwise reproducibility across GPU architectures.

### Numerical tolerance

The `numerical_tolerance` config parameter controls:

1. **Zero detection**: A floating-point value with absolute value < tolerance is treated as zero
2. **Persistence threshold**: Pairs with death - birth < tolerance are filtered out
3. **Convergence**: Iterative algorithms stop when change < tolerance

Default: `1e-12` for double precision, `1e-6f` for single precision.

### Potential numerical issues

Catastrophic cancellation can cause large errors in distance for near-identical points; this is mitigated by using `max_radius` to ignore near-zero edges. Sorting instability may arise when ties in filtration values produce different pairings -- the deterministic sort uses (value, dimension, vertices) as a tiebreaker to prevent this. GPU denormal performance can cause a significant slowdown on denormal-heavy inputs, though this is very rare in practice; `--ftz=false` preserves denormals for correctness. MPI summation order can produce different results with different rank counts, mitigated by binned accumulation for cross-count reproducibility.


[Back to Correctness Index](index.md)
