#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_memory.hpp"
#include "nerve/simd/simd_reduce.hpp"

#include <cmath>
#include <cstring>

namespace nerve::core
{

void simdMemcpy(void *dst, const void *src, std::size_t bytes)
{
    nerve::simd::simd_memcpy(dst, src, bytes);
}

void simdMemset(void *dst, int value, std::size_t bytes)
{
    nerve::simd::simd_memset(dst, value, bytes);
}

void simdMemcpyAligned(void *dst, const void *src, std::size_t bytes)
{
    // Both aligned and unaligned paths currently delegate to the same
    // simd_memcpy primitive.  The aligned path exists as an extension point
    // for future streaming-store optimisations.
    nerve::simd::simd_memcpy(dst, src, static_cast<std::size_t>(bytes));
}

double simdReduceSum(const double *data, std::size_t n)
{
    return nerve::simd::simd_reduce_sum(data, n);
}

} // namespace nerve::core
