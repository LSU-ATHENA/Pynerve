#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nerve::simd
{

// Float16 type -- mapped to compiler-supported _Float16
using half = _Float16;

// Helper: convert float to half using round-to-nearest-even
inline float half_to_float(half v)
{
    return static_cast<float>(v);
}
inline half float_to_half(float v)
{
    return static_cast<half>(v);
}

// Constants for float16
constexpr half kHalfZero = static_cast<half>(0.0f);
constexpr half kHalfOne = static_cast<half>(1.0f);

// Alignment
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kAvx512Alignment = 64;
constexpr std::size_t kAvxAlignment = 32;
constexpr std::size_t kSseAlignment = 16;
constexpr std::size_t kNeonAlignment = 16;

// Lane counts

constexpr int kAvx512DoubleLanes = 8;
constexpr int kAvx512FloatLanes = 16;
constexpr int kAvx2DoubleLanes = 4;
constexpr int kAvx2FloatLanes = 8;
constexpr int kSseDoubleLanes = 2;
constexpr int kSseFloatLanes = 4;
constexpr int kNeonDoubleLanes = 2;
constexpr int kNeonFloatLanes = 4;

// Dispatch table

struct SimdDispatchTable
{
    // Memory
    void (*memcpy)(void *dst, const void *src, std::size_t bytes);
    void (*memset)(void *dst, int value, std::size_t bytes);

    // Arithmetic (float64)
    void (*add)(double *a, const double *b, std::size_t n);
    void (*sub)(double *a, const double *b, std::size_t n);
    void (*mul)(double *a, const double *b, std::size_t n);
    void (*scale)(double *a, double alpha, std::size_t n);
    void (*axpy)(double alpha, const double *x, double *y, std::size_t n);
    void (*fmad)(const double *a, const double *b, double *c, std::size_t n);

    // Reductions
    double (*reduce_sum)(const double *data, std::size_t n);
    double (*reduce_max)(const double *data, std::size_t n);
    double (*reduce_min)(const double *data, std::size_t n);
    double (*dot)(const double *a, const double *b, std::size_t n);
    double (*norm2)(const double *vec, std::size_t n);
    double (*sqdiff_sum)(const double *a, const double *b, std::size_t n);

    // Element-wise unary
    void (*abs)(double *data, std::size_t n);
    void (*neg)(double *data, std::size_t n);
    void (*sqrt)(double *data, std::size_t n);
    void (*exp)(double *data, std::size_t n);
    void (*log)(double *data, std::size_t n);
    void (*relu)(double *data, std::size_t n);
    void (*sigmoid)(double *data, std::size_t n);
    void (*tanh)(double *data, std::size_t n);

    // Element-wise binary
    void (*min)(double *a, const double *b, std::size_t n);
    void (*max)(double *a, const double *b, std::size_t n);
    void (*clamp)(double *data, double lo, double hi, std::size_t n);

    // Float32 arithmetic
    void (*fmad_f32)(const float *a, const float *b, float *c, std::size_t n);

    // Float32 primitives
    void (*add_f32)(float *a, const float *b, std::size_t n);
    void (*sub_f32)(float *a, const float *b, std::size_t n);
    void (*mul_f32)(float *a, const float *b, std::size_t n);
    void (*scale_f32)(float *a, float alpha, std::size_t n);
    void (*axpy_f32)(float alpha, const float *x, float *y, std::size_t n);
    float (*reduce_sum_f32)(const float *data, std::size_t n);
    float (*reduce_max_f32)(const float *data, std::size_t n);
    float (*reduce_min_f32)(const float *data, std::size_t n);
    float (*dot_f32)(const float *a, const float *b, std::size_t n);
    float (*norm2_f32)(const float *vec, std::size_t n);

    // Float32 element-wise unary
    void (*neg_f32)(float *data, std::size_t n);
    void (*sqrt_f32)(float *data, std::size_t n);
    void (*exp_f32)(float *data, std::size_t n);
    void (*log_f32)(float *data, std::size_t n);
    void (*sigmoid_f32)(float *data, std::size_t n);
    void (*tanh_f32)(float *data, std::size_t n);

    // Float32 element-wise
    void (*abs_f32)(float *data, std::size_t n);
    void (*relu_f32)(float *data, std::size_t n);
    void (*min_f32)(float *a, const float *b, std::size_t n);
    void (*max_f32)(float *a, const float *b, std::size_t n);
    void (*clamp_f32)(float *data, float lo, float hi, std::size_t n);

    // Float32 distance primitives
    float (*sqdiff_sum_f32)(const float *a, const float *b, std::size_t n);
    float (*euclidean_f32)(const float *a, const float *b, std::size_t n);
    float (*cosine_f32)(const float *a, const float *b, std::size_t n);

    // Float32 BLAS level 1/2
    void (*gemv_f32)(float alpha, const float *A, const float *x, float beta, float *y,
                     std::size_t m, std::size_t n);
    void (*ger_f32)(float alpha, const float *x, const float *y, float *A, std::size_t m,
                    std::size_t n);

    // Float16 primitives
    void (*add_f16)(half *a, const half *b, std::size_t n);
    void (*sub_f16)(half *a, const half *b, std::size_t n);
    void (*mul_f16)(half *a, const half *b, std::size_t n);
    void (*scale_f16)(half *a, half alpha, std::size_t n);
    void (*axpy_f16)(half alpha, const half *x, half *y, std::size_t n);
    void (*fmad_f16)(const half *a, const half *b, half *c, std::size_t n);
    float (*reduce_sum_f16)(const half *data, std::size_t n);
    float (*reduce_max_f16)(const half *data, std::size_t n);
    float (*reduce_min_f16)(const half *data, std::size_t n);
    float (*dot_f16)(const half *a, const half *b, std::size_t n);
    float (*norm2_f16)(const half *vec, std::size_t n);
    void (*neg_f16)(half *data, std::size_t n);
    void (*sqrt_f16)(half *data, std::size_t n);
    void (*exp_f16)(half *data, std::size_t n);
    void (*log_f16)(half *data, std::size_t n);
    void (*relu_f16)(half *data, std::size_t n);
    void (*sigmoid_f16)(half *data, std::size_t n);
    void (*tanh_f16)(half *data, std::size_t n);
    void (*abs_f16)(half *data, std::size_t n);
    void (*min_f16)(half *a, const half *b, std::size_t n);
    void (*max_f16)(half *a, const half *b, std::size_t n);
    void (*clamp_f16)(half *data, half lo, half hi, std::size_t n);
    float (*sqdiff_sum_f16)(const half *a, const half *b, std::size_t n);

    // Float16 BLAS level 1/2
    void (*gemv_f16)(half alpha, const half *A, const half *x, half beta, half *y, std::size_t m,
                     std::size_t n);
    void (*ger_f16)(half alpha, const half *x, const half *y, half *A, std::size_t m,
                    std::size_t n);

    // Float16 quantize / dequantize
    void (*quantize_f16)(const half *input, std::size_t n, int bits, std::uint8_t *output);
    void (*dequantize_f16)(const std::uint8_t *input, std::size_t n, int bits, half *output);

    // BLAS level 1/2
    void (*gemv)(double alpha, const double *A, const double *x, double beta, double *y,
                 std::size_t m, std::size_t n);
    void (*ger)(double alpha, const double *x, const double *y, double *A, std::size_t m,
                std::size_t n);
};

extern SimdDispatchTable SIMD;

// Runtime architecture detection

enum class SimdArch
{
    SCALAR,
    SSE41,
    AVX2,
    AVX512,
    NEON,
    SVE
};

SimdArch detect_simd_arch();
const char *simd_arch_name(SimdArch arch);

void simd_init();

// Convenience wrappers

inline void simd_memcpy(void *dst, const void *src, std::size_t bytes)
{
    simd_init();
    SIMD.memcpy(dst, src, bytes);
}

inline void simd_memset(void *dst, int value, std::size_t bytes)
{
    simd_init();
    SIMD.memset(dst, value, bytes);
}

inline void simd_add(double *a, const double *b, std::size_t n)
{
    simd_init();
    SIMD.add(a, b, n);
}

inline void simd_sub(double *a, const double *b, std::size_t n)
{
    simd_init();
    SIMD.sub(a, b, n);
}

inline void simd_mul(double *a, const double *b, std::size_t n)
{
    simd_init();
    SIMD.mul(a, b, n);
}

inline void simd_scale(double *a, double alpha, std::size_t n)
{
    simd_init();
    SIMD.scale(a, alpha, n);
}

inline void simd_axpy(double alpha, const double *x, double *y, std::size_t n)
{
    simd_init();
    SIMD.axpy(alpha, x, y, n);
}

inline void simd_fmad(const double *a, const double *b, double *c, std::size_t n)
{
    simd_init();
    SIMD.fmad(a, b, c, n);
}

inline double simd_reduce_sum(const double *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_sum(data, n);
}

inline double simd_reduce_max(const double *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_max(data, n);
}

inline double simd_reduce_min(const double *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_min(data, n);
}

inline double simd_dot(const double *a, const double *b, std::size_t n)
{
    simd_init();
    return SIMD.dot(a, b, n);
}

inline double simd_norm2(const double *vec, std::size_t n)
{
    simd_init();
    return SIMD.norm2(vec, n);
}

inline double simd_sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    simd_init();
    return SIMD.sqdiff_sum(a, b, n);
}

inline void simd_abs(double *data, std::size_t n)
{
    simd_init();
    SIMD.abs(data, n);
}

inline void simd_neg(double *data, std::size_t n)
{
    simd_init();
    SIMD.neg(data, n);
}

inline void simd_sqrt(double *data, std::size_t n)
{
    simd_init();
    SIMD.sqrt(data, n);
}

inline void simd_exp(double *data, std::size_t n)
{
    simd_init();
    SIMD.exp(data, n);
}

inline void simd_log(double *data, std::size_t n)
{
    simd_init();
    SIMD.log(data, n);
}

inline void simd_relu(double *data, std::size_t n)
{
    simd_init();
    SIMD.relu(data, n);
}

inline void simd_sigmoid(double *data, std::size_t n)
{
    simd_init();
    SIMD.sigmoid(data, n);
}

inline void simd_tanh(double *data, std::size_t n)
{
    simd_init();
    SIMD.tanh(data, n);
}

inline void simd_min(double *a, const double *b, std::size_t n)
{
    simd_init();
    SIMD.min(a, b, n);
}

inline void simd_max(double *a, const double *b, std::size_t n)
{
    simd_init();
    SIMD.max(a, b, n);
}

inline void simd_clamp(double *data, double lo, double hi, std::size_t n)
{
    simd_init();
    SIMD.clamp(data, lo, hi, n);
}

// Float32 convenience wrappers

inline void simd_fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    simd_init();
    SIMD.fmad_f32(a, b, c, n);
}

inline void simd_add_f32(float *a, const float *b, std::size_t n)
{
    simd_init();
    SIMD.add_f32(a, b, n);
}

inline float simd_reduce_sum_f32(const float *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_sum_f32(data, n);
}

inline float simd_dot_f32(const float *a, const float *b, std::size_t n)
{
    simd_init();
    return SIMD.dot_f32(a, b, n);
}

inline void simd_sub_f32(float *a, const float *b, std::size_t n)
{
    simd_init();
    SIMD.sub_f32(a, b, n);
}

inline void simd_mul_f32(float *a, const float *b, std::size_t n)
{
    simd_init();
    SIMD.mul_f32(a, b, n);
}

inline void simd_scale_f32(float *a, float alpha, std::size_t n)
{
    simd_init();
    SIMD.scale_f32(a, alpha, n);
}

inline void simd_axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    simd_init();
    SIMD.axpy_f32(alpha, x, y, n);
}

inline float simd_reduce_max_f32(const float *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_max_f32(data, n);
}

inline float simd_reduce_min_f32(const float *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_min_f32(data, n);
}

inline float simd_norm2_f32(const float *vec, std::size_t n)
{
    simd_init();
    return SIMD.norm2_f32(vec, n);
}

inline void simd_neg_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.neg_f32(data, n);
}

inline void simd_sqrt_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.sqrt_f32(data, n);
}

inline void simd_exp_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.exp_f32(data, n);
}

inline void simd_log_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.log_f32(data, n);
}

inline void simd_sigmoid_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.sigmoid_f32(data, n);
}

inline void simd_tanh_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.tanh_f32(data, n);
}

inline float simd_sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    simd_init();
    return SIMD.sqdiff_sum_f32(a, b, n);
}

inline float simd_euclidean_f32(const float *a, const float *b, std::size_t n)
{
    simd_init();
    return std::sqrt(SIMD.sqdiff_sum_f32(a, b, n));
}

inline float simd_cosine_f32(const float *a, const float *b, std::size_t n)
{
    simd_init();
    float dot_val = SIMD.dot_f32(a, b, n);
    float na = SIMD.norm2_f32(a, n);
    float nb = SIMD.norm2_f32(b, n);
    if (na == 0.0f || nb == 0.0f)
        return 1.0f;
    float cos_sim = dot_val / (na * nb);
    if (cos_sim < -1.0f)
        cos_sim = -1.0f;
    if (cos_sim > 1.0f)
        cos_sim = 1.0f;
    return 1.0f - cos_sim;
}

inline void simd_gemv_f32(float alpha, const float *A, const float *x, float beta, float *y,
                          std::size_t m, std::size_t n)
{
    simd_init();
    SIMD.gemv_f32(alpha, A, x, beta, y, m, n);
}

inline void simd_ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m,
                         std::size_t n)
{
    simd_init();
    SIMD.ger_f32(alpha, x, y, A, m, n);
}

inline void simd_abs_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.abs_f32(data, n);
}

inline void simd_relu_f32(float *data, std::size_t n)
{
    simd_init();
    SIMD.relu_f32(data, n);
}

inline void simd_min_f32(float *a, const float *b, std::size_t n)
{
    simd_init();
    SIMD.min_f32(a, b, n);
}

inline void simd_max_f32(float *a, const float *b, std::size_t n)
{
    simd_init();
    SIMD.max_f32(a, b, n);
}

inline void simd_clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    simd_init();
    SIMD.clamp_f32(data, lo, hi, n);
}

// Float16 convenience wrappers

inline void simd_add_f16(half *a, const half *b, std::size_t n)
{
    simd_init();
    SIMD.add_f16(a, b, n);
}

inline void simd_sub_f16(half *a, const half *b, std::size_t n)
{
    simd_init();
    SIMD.sub_f16(a, b, n);
}

inline void simd_mul_f16(half *a, const half *b, std::size_t n)
{
    simd_init();
    SIMD.mul_f16(a, b, n);
}

inline void simd_scale_f16(half *a, half alpha, std::size_t n)
{
    simd_init();
    SIMD.scale_f16(a, alpha, n);
}

inline void simd_axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    simd_init();
    SIMD.axpy_f16(alpha, x, y, n);
}

inline void simd_fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    simd_init();
    SIMD.fmad_f16(a, b, c, n);
}

inline float simd_reduce_sum_f16(const half *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_sum_f16(data, n);
}

inline float simd_reduce_max_f16(const half *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_max_f16(data, n);
}

inline float simd_reduce_min_f16(const half *data, std::size_t n)
{
    simd_init();
    return SIMD.reduce_min_f16(data, n);
}

inline float simd_dot_f16(const half *a, const half *b, std::size_t n)
{
    simd_init();
    return SIMD.dot_f16(a, b, n);
}

inline float simd_norm2_f16(const half *vec, std::size_t n)
{
    simd_init();
    return SIMD.norm2_f16(vec, n);
}

inline void simd_neg_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.neg_f16(data, n);
}

inline void simd_sqrt_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.sqrt_f16(data, n);
}

inline void simd_exp_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.exp_f16(data, n);
}

inline void simd_log_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.log_f16(data, n);
}

inline void simd_relu_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.relu_f16(data, n);
}

inline void simd_sigmoid_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.sigmoid_f16(data, n);
}

inline void simd_tanh_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.tanh_f16(data, n);
}

inline void simd_abs_f16(half *data, std::size_t n)
{
    simd_init();
    SIMD.abs_f16(data, n);
}

inline void simd_min_f16(half *a, const half *b, std::size_t n)
{
    simd_init();
    SIMD.min_f16(a, b, n);
}

inline void simd_max_f16(half *a, const half *b, std::size_t n)
{
    simd_init();
    SIMD.max_f16(a, b, n);
}

inline void simd_clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    simd_init();
    SIMD.clamp_f16(data, lo, hi, n);
}

inline float simd_sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    simd_init();
    return SIMD.sqdiff_sum_f16(a, b, n);
}

inline void simd_gemv_f16(half alpha, const half *A, const half *x, half beta, half *y,
                          std::size_t m, std::size_t n)
{
    simd_init();
    SIMD.gemv_f16(alpha, A, x, beta, y, m, n);
}

inline void simd_ger_f16(half alpha, const half *x, const half *y, half *A, std::size_t m,
                         std::size_t n)
{
    simd_init();
    SIMD.ger_f16(alpha, x, y, A, m, n);
}

inline void simd_quantize_f16(const half *input, std::size_t n, int bits, std::uint8_t *output)
{
    simd_init();
    SIMD.quantize_f16(input, n, bits, output);
}

inline void simd_dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, half *output)
{
    simd_init();
    SIMD.dequantize_f16(input, n, bits, output);
}

inline void simd_gemv(double alpha, const double *A, const double *x, double beta, double *y,
                      std::size_t m, std::size_t n)
{
    simd_init();
    SIMD.gemv(alpha, A, x, beta, y, m, n);
}

inline void simd_ger(double alpha, const double *x, const double *y, double *A, std::size_t m,
                     std::size_t n)
{
    simd_init();
    SIMD.ger(alpha, x, y, A, m, n);
}

} // namespace nerve::simd
