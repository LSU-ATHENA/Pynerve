# `nerve::cpu::simd` -- SIMD Detection

```cpp
#include <nerve/cpu/simd.hpp>

namespace nerve::cpu::simd;

class CPUFeatureDetector {
public:
    static bool hasAVX512F();       // AVX-512 Foundation
    static bool hasAVX512VL();      // AVX-512 Vector Length
    static bool hasAVX512BW();      // AVX-512 Byte/Word
    static bool hasAVX2();          // AVX2
    static bool hasFMA();           // FMA3
    static int getMaxSIMDWidth();   // 512, 256, or 128
    static std::string getCPUModel();
    static int getNumCores();
    static int getNumThreads();
};

// Convenience free functions
bool hasAVX512F();
bool hasAVX2();
int getMaxSIMDWidth();
std::string getCPUOptimizationReport();
```

**Cost:** O(1) after initial CPUID call (results cached).

<- [C++ API Overview](index.md)
