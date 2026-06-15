#pragma once

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define NERVE_HAS_NEON 1
#else
#define NERVE_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define NERVE_HAS_SVE 1
#else
#define NERVE_HAS_SVE 0
#endif

#if defined(__ARM_FEATURE_SVE2)
#define NERVE_HAS_SVE2 1
#else
#define NERVE_HAS_SVE2 0
#endif

namespace nerve::cpu::arm
{

#if NERVE_HAS_NEON

// NEON float32: 4 floats per vector
inline float reduce_sum_f32(float32x4_t v)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return vaddvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum = vadd_f32(lo, hi);
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}

// NEON float64: 2 doubles per vector
inline double reduce_sum_f64(float64x2_t v)
{
    double result[2];
    vst1q_f64(result, v);
    return result[0] + result[1];
}

#endif // NERVE_HAS_NEON

// SVE float32 horizontal sum
#if NERVE_HAS_SVE
inline float sve_reduce_sum_f32(svfloat32_t v)
{
    return svaddv_f32(svptrue_b32(), v);
}
inline double sve_reduce_sum_f64(svfloat64_t v)
{
    return svaddv_f64(svptrue_b64(), v);
}
#endif

} // namespace nerve::cpu::arm

// Euclidean distance squared accumulation helpers
namespace nerve::cpu::arm::detail
{

#if NERVE_HAS_NEON

// NEON: accumulate squared difference for 4 floats at a time
inline float32x4_t neon_sqdiff_f32(float32x4_t a, float32x4_t b, float32x4_t acc)
{
    float32x4_t diff = vsubq_f32(a, b);
    return vmlaq_f32(acc, diff, diff);
}

// NEON: accumulate squared difference for 2 doubles at a time
inline float64x2_t neon_sqdiff_f64(float64x2_t a, float64x2_t b, float64x2_t acc)
{
    float64x2_t diff = vsubq_f64(a, b);
    return vfmaq_f64(acc, diff, diff);
}

// NEON: accumulate absolute difference for Manhattan
inline float32x4_t neon_absdiff_f32(float32x4_t a, float32x4_t b, float32x4_t acc)
{
    float32x4_t diff = vsubq_f32(a, b);
    return vaddq_f32(acc, vabsq_f32(diff));
}

#endif // NERVE_HAS_NEON

// SVE: accumulate squared difference (vector-length agnostic)
#if NERVE_HAS_SVE
inline svfloat32_t sve_sqdiff_f32(svfloat32_t a, svfloat32_t b, svfloat32_t acc, svbool_t mask)
{
    svfloat32_t diff = svsub_f32_m(mask, a, b);
    return svmla_f32_m(mask, acc, diff, diff);
}

inline svfloat64_t sve_sqdiff_f64(svfloat64_t a, svfloat64_t b, svfloat64_t acc, svbool_t mask)
{
    svfloat64_t diff = svsub_f64_m(mask, a, b);
    return svmla_f64_m(mask, acc, diff, diff);
}
#endif

} // namespace nerve::cpu::arm::detail
