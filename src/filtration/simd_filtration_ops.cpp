#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"
#include "nerve/filtration/simd_filtration.hpp"

#include <algorithm>
#include <cmath>

namespace nerve::filtration
{

void simdBatchFilterValues(double *values, Size n, Size start_dim, Size end_dim, double threshold)
{
    (void)start_dim;
    (void)end_dim;
#if defined(__AVX512F__)
    if (cpu::CPUFeatureDetector::instance().hasAVX512F())
    {
        __m512d v_thresh = _mm512_set1_pd(threshold);
        Size i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(values + i);
            __mmask8 mask = _mm512_cmp_pd_mask(v, v_thresh, _CMP_GE_OQ);
            _mm512_mask_storeu_pd(values + i, mask, v);
        }
        for (; i < n; ++i)
            if (values[i] < threshold)
                values[i] = 0.0;
    }
    else
#elif defined(__AVX2__)
    if (cpu::CPUFeatureDetector::instance().hasAVX2())
#endif
    {
        for (Size i = 0; i < n; ++i)
            if (values[i] < threshold)
                values[i] = 0.0;
    }
}

void simdSortPairsByBirth(Pair *pairs, Size n)
{
    std::sort(pairs, pairs + n, [](const Pair &a, const Pair &b) { return a.birth < b.birth; });
}

} // namespace nerve::filtration
