# CPU

## Quick start

```python
import pynerve.cpu as cpu

# Detect available CPU features
feats = cpu.features()
# feats.avx512 -> True (if AVX-512F supported)
# feats.avx2   -> True
# feats.fma    -> True
# feats.sse41  -> True

# Check specific feature
if cpu.avx512_enabled():
    # Use AVX-512 optimized path
    diagrams = cpu.compute_persistence_avx512(points, max_dim=2)
else:
    diagrams = compute_persistence(points, max_dim=2)
```

AVX-512/AVX2/FMA/SSE4.1 optimized persistent homology operations with runtime
CPU feature detection via CPUID. The dispatch layer selects the fastest kernel
for the current hardware.


## API

```cpp
#include <nerve/cpu/x86_intrinsics.hpp>
#include <nerve/cpu/simd.hpp>

namespace nerve::cpu::simd {

// CPU feature detection
class CPUFeatureDetector {
    static bool hasAVX512F();
    static bool hasAVX512VL();
    static bool hasAVX512BW();
    static bool hasAVX2();
    static bool hasFMA();
    static int getMaxSIMDWidth();
    static string getCPUModel();
    static int getNumCores();
    static int getNumThreads();
};

// Free functions
bool hasAVX512F();
bool hasAVX2();
int getMaxSIMDWidth();
string getCPUModel();
int getNumCores();
int getNumThreads();

// Dispatch to best SIMD path
PersistenceDiagram computeCPUOptimized(
    span<const double> points, size_t n_points,
    size_t point_dim, double max_distance);

string getCPUOptimizationReport();

}
```

```python
# Python bindings (via pynerve.cpu)
cpu.features()             # -> CPUFeatureInfo with boolean flags
cpu.avx512_enabled()       # -> bool
cpu.simd_width()           # -> int (128, 256, or 512)
cpu.get_cpu_model()        # -> str
cpu.get_num_cores()        # -> int
```

### SIMD dispatch

SIMD dispatch operates at three feature levels. SSE4.1 provides 128-bit width for basic distance and reduction operations as a fallback. AVX2 combined with FMA provides 256-bit width for distance matrix and matrix reduction. AVX-512F with VL and BW extensions provides 512-bit width for distance matrix, reduction, and clearing operations.

The implementation in `src/cpu/avx512_ph_ops.cpp` provides AVX-512 kernels for
boundary matrix reduction, column clearing, and distance computation. Runtime
dispatch falls back automatically.


## FAQ

**Q: How do I check if AVX-512 is available on my system?**
A: Use `cpu.avx512_enabled()` or check `cpu.features().avx512`. Pynerve detects CPU features at runtime using the CPUID instruction.

**Q: What happens if no SIMD extensions are available?**
A: Pynerve falls back to scalar implementations for all operations. Performance is still competitive due to cache-aware data structures and Numba JIT compilation.

**Q: Which CPU features give the biggest speedup for persistence computation?**
A: AVX-512 provides the largest gains, especially for distance matrix computation (up to 8x over scalar) and matrix reduction. AVX2 with FMA is the next best option. SSE4.1 is the minimum baseline for any SIMD acceleration.
