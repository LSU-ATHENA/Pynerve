#include "nerve/platform.hpp"
#include "nerve/simd/simd_base.hpp"

#include <cstdlib>
#include <cstring>

namespace nerve::simd
{

// Global dispatch table
SimdDispatchTable SIMD = {};

// Forward declarations for architecture assignment functions
// Each is implemented in the corresponding src/simd/simd_*.cpp file.
extern "C" void nerve_simd_assign_scalar(SimdDispatchTable *);
#if defined(__SSE4_1__)
extern "C" void nerve_simd_assign_sse(SimdDispatchTable *);
#endif
#if defined(__AVX2__)
extern "C" void nerve_simd_assign_avx2(SimdDispatchTable *);
#endif
#if defined(__AVX512F__)
extern "C" void nerve_simd_assign_avx512(SimdDispatchTable *);
#endif
#if defined(NERVE_HAS_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
extern "C" void nerve_simd_assign_neon(SimdDispatchTable *);
#endif
#if defined(NERVE_HAS_SVE) || defined(__ARM_FEATURE_SVE)
extern "C" void nerve_simd_assign_sve(SimdDispatchTable *);
#endif

// CPU feature detection

// NERVE_CAN_USE_CPUID: defined when the CPUID instruction is available
// via either GCC/Clang builtins or MSVC __cpuidex.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define NERVE_CAN_USE_CPUID 1
#else
#define NERVE_CAN_USE_CPUID 0
#endif

#if NERVE_CAN_USE_CPUID
static const auto s_cpu_flags = nerve::cpu::CpuFeatureFlags::detect();
#endif

static bool cpu_has_avx512f()
{
#if NERVE_CAN_USE_CPUID
    return s_cpu_flags.has_avx512f;
#elif defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}

static bool cpu_has_avx2()
{
#if NERVE_CAN_USE_CPUID
    return s_cpu_flags.has_avx2;
#elif defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

static bool cpu_has_sse41()
{
#if NERVE_CAN_USE_CPUID
    return s_cpu_flags.has_sse41;
#elif defined(__SSE4_1__)
    return true;
#else
    return false;
#endif
}

static bool cpu_has_neon()
{
#if defined(NERVE_HAS_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

static bool cpu_has_sve()
{
#if defined(NERVE_HAS_SVE) || defined(__ARM_FEATURE_SVE)
    return true;
#else
    return false;
#endif
}

SimdArch detect_simd_arch()
{
#if defined(NERVE_SIMD_FORCE_SCALAR)
    return SimdArch::SCALAR;
#elif defined(NERVE_SIMD_FORCE_SSE41)
    return SimdArch::SSE41;
#elif defined(NERVE_SIMD_FORCE_AVX2)
    return SimdArch::AVX2;
#elif defined(NERVE_SIMD_FORCE_AVX512)
    return SimdArch::AVX512;
#elif defined(NERVE_SIMD_FORCE_NEON)
    return SimdArch::NEON;
#elif defined(NERVE_SIMD_FORCE_SVE)
    return SimdArch::SVE;
#else
    if (cpu_has_avx512f())
        return SimdArch::AVX512;
    if (cpu_has_avx2())
        return SimdArch::AVX2;
    if (cpu_has_sse41())
        return SimdArch::SSE41;
    if (cpu_has_neon())
        return SimdArch::NEON;
    if (cpu_has_sve())
        return SimdArch::SVE;
    return SimdArch::SCALAR;
#endif
}

const char *simd_arch_name(SimdArch arch)
{
    switch (arch)
    {
        case SimdArch::SCALAR:
            return "scalar";
        case SimdArch::SSE41:
            return "SSE4.1";
        case SimdArch::AVX2:
            return "AVX2";
        case SimdArch::AVX512:
            return "AVX-512";
        case SimdArch::NEON:
            return "NEON";
        case SimdArch::SVE:
            return "SVE";
        default:
            return "unknown";
    }
}

void simd_init()
{
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

    SimdArch arch = detect_simd_arch();

    switch (arch)
    {
#if defined(__AVX512F__)
        case SimdArch::AVX512:
            if (cpu_has_avx512f())
            {
                nerve_simd_assign_avx512(&SIMD);
                break;
            }
            // Fall through if runtime check fails despite compile-time support
#endif
#if defined(__AVX2__)
        case SimdArch::AVX2:
            nerve_simd_assign_avx2(&SIMD);
            break;
#endif
#if defined(__SSE4_1__)
        case SimdArch::SSE41:
            nerve_simd_assign_sse(&SIMD);
            break;
#endif
#if defined(NERVE_HAS_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        case SimdArch::NEON:
            nerve_simd_assign_neon(&SIMD);
            break;
#endif
#if defined(NERVE_HAS_SVE) || defined(__ARM_FEATURE_SVE)
        case SimdArch::SVE:
            nerve_simd_assign_sve(&SIMD);
            break;
#endif
        default:
        case SimdArch::SCALAR:
            nerve_simd_assign_scalar(&SIMD);
            break;
    }
}

} // namespace nerve::simd
