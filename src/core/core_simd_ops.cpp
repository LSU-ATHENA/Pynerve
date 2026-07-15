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
    // Check 64-byte alignment for potential streaming stores
    if (reinterpret_cast<uintptr_t>(dst) % 64 == 0 && reinterpret_cast<uintptr_t>(src) % 64 == 0)
    {
        // Currently delegates to simd_memcpy; aligned streaming store
        // optimization can be added to the SIMD primitive layer later.
        nerve::simd::simd_memcpy(dst, src, static_cast<std::size_t>(bytes));
    }
    else
    {
        nerve::simd::simd_memcpy(dst, src, static_cast<std::size_t>(bytes));
    }
}

double simdReduceSum(const double *data, std::size_t n)
{
    return nerve::simd::simd_reduce_sum(data, n);
}

} // namespace nerve::core
