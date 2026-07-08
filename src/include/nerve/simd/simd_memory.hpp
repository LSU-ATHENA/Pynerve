#pragma once
#include "nerve/simd/simd_base.hpp"

namespace nerve::simd
{

// simd_memcpy / simd_memset are defined in simd_base.hpp

// Aligned variant using non-temporal stores when possible
inline void simd_memcpy_aligned(void *dst, const void *src, std::size_t bytes)
{
    // Fallback to regular memcpy for now; AVX2/AVX-512 backends
    // will overload this with streaming stores when alignment holds.
    simd_memcpy(dst, src, bytes);
}

} // namespace nerve::simd
