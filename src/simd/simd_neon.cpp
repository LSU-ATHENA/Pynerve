#include "nerve/simd/simd_base.hpp"
#include "nerve/cpu/arm_simd.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace nerve::simd::neon_impl
{

// NEON implementations delegate to the functions in arm_simd.hpp
// which already provide optimized NEON primitives.

void memcpy(void *dst, const void *src, std::size_t bytes)
{
    nerve::simd::neon::memcpy(dst, src, bytes);
}

void memset(void *dst, int value, std::size_t bytes)
{
    nerve::simd::neon::memset(dst, value, bytes);
}

void add(double *a, const double *b, std::size_t n)
{
    nerve::simd::neon::add(a, b, n);
}

void sub(double *a, const double *b, std::size_t n)
{
    nerve::simd::neon::sub(a, b, n);
}

void mul(double *a, const double *b, std::size_t n)
{
    nerve::simd::neon::mul(a, b, n);
}

void scale(double *a, double alpha, std::size_t n)
{
    nerve::simd::neon::scale(a, alpha, n);
}

void axpy(double alpha, const double *x, double *y, std::size_t n)
{
    nerve::simd::neon::axpy(alpha, x, y, n);
}

void fmad(const double *a, const double *b, double *c, std::size_t n)
{
    nerve::simd::neon::fmad(a, b, c, n);
}

double reduce_sum(const double *data, std::size_t n)
{
    return nerve::simd::neon::reduce_sum(data, n);
}

double reduce_max(const double *data, std::size_t n)
{
    return nerve::simd::neon::reduce_max(data, n);
}

double reduce_min(const double *data, std::size_t n)
{
    if (n == 0) return 0.0;
    // NEON doesn't have a dedicated reduce_min, use scalar
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m) m = data[i];
    return m;
}

double dot(const double *a, const double *b, std::size_t n)
{
    return nerve::simd::neon::dot(a, b, n);
}

double norm2(const double *vec, std::size_t n)
{
    return nerve::simd::neon::norm2(vec, n);
}

double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    return nerve::simd::neon::sqdiff_sum(a, b, n);
}

void abs(double *data, std::size_t n)
{
    // NEON has vabsq_f64 for float64
    std::size_t i = 0;
#if NERVE_HAS_NEON
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vst1q_f64(data + i, vabsq_f64(v));
    }
#endif
    for (; i < n; ++i) data[i] = std::abs(data[i]);
}

void neg(double *data, std::size_t n)
{
    std::size_t i = 0;
#if NERVE_HAS_NEON
    float64x2_t vzero = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vst1q_f64(data + i, vsubq_f64(vzero, v));
    }
#endif
    for (; i < n; ++i) data[i] = -data[i];
}

void sqrt(double *data, std::size_t n)
{
    std::size_t i = 0;
#if NERVE_HAS_NEON
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(data + i);
        vst1q_f64(data + i, vsqrtq_f64(v));
    }
#endif
    for (; i < n; ++i) data[i] = std::sqrt(data[i]);
}

void exp(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) data[i] = std::exp(data[i]);
}

void log(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) data[i] = std::log(data[i]);
}

void relu(double *data, std::size_t n)
{
    nerve::simd::neon::relu(data, n);
}

void sigmoid(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
}

void tanh(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) data[i] = std::tanh(data[i]);
}

void min(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
#if NERVE_HAS_NEON
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vminq_f64(va, vb));
    }
#endif
    for (; i < n; ++i) a[i] = a[i] < b[i] ? a[i] : b[i];
}

void max(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
#if NERVE_HAS_NEON
    for (; i + 2 <= n; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vmaxq_f64(va, vb));
    }
#endif
    for (; i < n; ++i) a[i] = a[i] > b[i] ? a[i] : b[i];
}

void clamp(double *data, double lo, double hi, std::size_t n)
{
    nerve::simd::neon::clamp(data, lo, hi, n);
}

void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    nerve::simd::neon::fmad_f32(a, b, c, n);
}

void add_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::neon::add_f32(a, b, n);
}

void sub_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::neon::sub_f32(a, b, n);
}

void mul_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::neon::mul_f32(a, b, n);
}

void scale_f32(float *a, float alpha, std::size_t n)
{
    nerve::simd::neon::scale_f32(a, alpha, n);
}

void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    nerve::simd::neon::axpy_f32(alpha, x, y, n);
}

float reduce_sum_f32(const float *data, std::size_t n)
{
    return nerve::simd::neon::reduce_sum_f32(data, n);
}

float reduce_max_f32(const float *data, std::size_t n)
{
    return nerve::simd::neon::reduce_max_f32(data, n);
}

float reduce_min_f32(const float *data, std::size_t n)
{
    return nerve::simd::neon::reduce_min_f32(data, n);
}

float dot_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::neon::dot_f32(a, b, n);
}

float norm2_f32(const float *vec, std::size_t n)
{
    return nerve::simd::neon::norm2_f32(vec, n);
}

float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::neon::sqdiff_sum_f32(a, b, n);
}

float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::neon::euclidean_f32(a, b, n);
}

float cosine_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::neon::cosine_f32(a, b, n);
}

void neg_f32(float *data, std::size_t n)
{
    nerve::simd::neon::neg_f32(data, n);
}

void sqrt_f32(float *data, std::size_t n)
{
    nerve::simd::neon::sqrt_f32(data, n);
}

void exp_f32(float *data, std::size_t n)
{
    nerve::simd::neon::exp_f32(data, n);
}

void log_f32(float *data, std::size_t n)
{
    nerve::simd::neon::log_f32(data, n);
}

void sigmoid_f32(float *data, std::size_t n)
{
    nerve::simd::neon::sigmoid_f32(data, n);
}

void tanh_f32(float *data, std::size_t n)
{
    nerve::simd::neon::tanh_f32(data, n);
}

void abs_f32(float *data, std::size_t n)
{
    nerve::simd::neon::abs_f32(data, n);
}

void relu_f32(float *data, std::size_t n)
{
    nerve::simd::neon::relu_f32(data, n);
}

void min_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::neon::min_f32(a, b, n);
}

void max_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::neon::max_f32(a, b, n);
}

void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    nerve::simd::neon::clamp_f32(data, lo, hi, n);
}

void add_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::neon::add_f16(a, b, n);
}

void sub_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::neon::sub_f16(a, b, n);
}

void mul_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::neon::mul_f16(a, b, n);
}

void scale_f16(half *a, half alpha, std::size_t n)
{
    nerve::simd::neon::scale_f16(a, alpha, n);
}

void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    nerve::simd::neon::axpy_f16(alpha, x, y, n);
}

void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    nerve::simd::neon::fmad_f16(a, b, c, n);
}

float reduce_sum_f16(const half *data, std::size_t n)
{
    return nerve::simd::neon::reduce_sum_f16(data, n);
}

float reduce_max_f16(const half *data, std::size_t n)
{
    return nerve::simd::neon::reduce_max_f16(data, n);
}

float reduce_min_f16(const half *data, std::size_t n)
{
    return nerve::simd::neon::reduce_min_f16(data, n);
}

float dot_f16(const half *a, const half *b, std::size_t n)
{
    return nerve::simd::neon::dot_f16(a, b, n);
}

float norm2_f16(const half *vec, std::size_t n)
{
    return nerve::simd::neon::norm2_f16(vec, n);
}

void neg_f16(half *data, std::size_t n)
{
    nerve::simd::neon::neg_f16(data, n);
}

void sqrt_f16(half *data, std::size_t n)
{
    nerve::simd::neon::sqrt_f16(data, n);
}

void exp_f16(half *data, std::size_t n)
{
    nerve::simd::neon::exp_f16(data, n);
}

void log_f16(half *data, std::size_t n)
{
    nerve::simd::neon::log_f16(data, n);
}

void relu_f16(half *data, std::size_t n)
{
    nerve::simd::neon::relu_f16(data, n);
}

void sigmoid_f16(half *data, std::size_t n)
{
    nerve::simd::neon::sigmoid_f16(data, n);
}

void tanh_f16(half *data, std::size_t n)
{
    nerve::simd::neon::tanh_f16(data, n);
}

void abs_f16(half *data, std::size_t n)
{
    nerve::simd::neon::abs_f16(data, n);
}

void min_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::neon::min_f16(a, b, n);
}

void max_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::neon::max_f16(a, b, n);
}

void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    nerve::simd::neon::clamp_f16(data, lo, hi, n);
}

float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    return nerve::simd::neon::sqdiff_sum_f16(a, b, n);
}


void gemv_f16(half alpha, const half *A, const half *x,
              half beta, half *y, std::size_t m, std::size_t n)
{
    float fa = half_to_float(alpha);
    float fb = half_to_float(beta);
    if (fb != 1.0f)
        for (std::size_t i = 0; i < m; ++i) y[i] = float_to_half(half_to_float(y[i]) * fb);
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += half_to_float(A[i * n + j]) * half_to_float(x[j]);
        y[i] = float_to_half(half_to_float(y[i]) + fa * sum);
    }
}

void ger_f16(half alpha, const half *x, const half *y,
             half *A, std::size_t m, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < m; ++i)
    {
        float xi = half_to_float(x[i]);
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] = float_to_half(half_to_float(A[i * n + j]) + fa * xi * half_to_float(y[j]));
    }
}

void quantize_f16(const half *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (n == 0) return;
    float scale = static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(input[i]);
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        output[i] = static_cast<std::uint8_t>(v * scale + 0.5f);
    }
}

void dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, half *output)
{
    if (n == 0) return;
    float inv = 1.0f / static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = float_to_half(static_cast<float>(input[i]) * inv);
}

void gemv_f32(float alpha, const float *A, const float *x,
              float beta, float *y, std::size_t m, std::size_t n)
{
    if (beta != 1.0f)
        for (std::size_t i = 0; i < m; ++i) y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = nerve::simd::neon::dot_f32(A + i * n, x, n);
        y[i] += alpha * sum;
    }
}

void ger_f32(float alpha, const float *x, const float *y,
             float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        float axi = alpha * x[i];
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += axi * y[j];
    }
}

void gemv(double alpha, const double *A, const double *x,
          double beta, double *y, std::size_t m, std::size_t n)
{
    if (beta != 1.0)
        for (std::size_t i = 0; i < m; ++i) y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        double sum = nerve::simd::neon::dot(A + i * n, x, n);
        y[i] += alpha * sum;
    }
}

void ger(double alpha, const double *x, const double *y,
         double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        double axi = alpha * x[i];
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += axi * y[j];
    }
}

} // namespace nerve::simd::neon_impl

extern "C" void nerve_simd_assign_neon(nerve::simd::SimdDispatchTable *table)
{
    using namespace nerve::simd::neon_impl;
    table->memcpy        = memcpy;
    table->memset        = memset;
    table->add           = add;
    table->sub           = sub;
    table->mul           = mul;
    table->scale         = scale;
    table->axpy          = axpy;
    table->fmad          = fmad;
    table->reduce_sum    = reduce_sum;
    table->reduce_max    = reduce_max;
    table->reduce_min    = reduce_min;
    table->dot           = dot;
    table->norm2         = norm2;
    table->sqdiff_sum    = sqdiff_sum;
    table->abs           = abs;
    table->neg           = neg;
    table->sqrt          = sqrt;
    table->exp           = exp;
    table->log           = log;
    table->relu          = relu;
    table->sigmoid       = sigmoid;
    table->tanh          = tanh;
    table->min           = min;
    table->max           = max;
    table->clamp         = clamp;
    table->fmad_f32      = fmad_f32;
    table->add_f32       = add_f32;
    table->sub_f32       = sub_f32;
    table->mul_f32       = mul_f32;
    table->scale_f32     = scale_f32;
    table->axpy_f32      = axpy_f32;
    table->reduce_sum_f32 = reduce_sum_f32;
    table->reduce_max_f32 = reduce_max_f32;
    table->reduce_min_f32 = reduce_min_f32;
    table->dot_f32       = dot_f32;
    table->norm2_f32     = norm2_f32;
    table->neg_f32       = neg_f32;
    table->sqrt_f32      = sqrt_f32;
    table->exp_f32       = exp_f32;
    table->log_f32       = log_f32;
    table->sigmoid_f32   = sigmoid_f32;
    table->tanh_f32      = tanh_f32;
    table->sqdiff_sum_f32 = sqdiff_sum_f32;
    table->euclidean_f32  = euclidean_f32;
    table->cosine_f32     = cosine_f32;
    table->abs_f32       = abs_f32;
    table->relu_f32      = relu_f32;
    table->min_f32       = min_f32;
    table->max_f32       = max_f32;
    table->clamp_f32     = clamp_f32;
    table->gemv_f32      = gemv_f32;
    table->ger_f32       = ger_f32;
    table->gemv          = gemv;
    table->ger           = ger;
    table->add_f16       = add_f16;
    table->sub_f16       = sub_f16;
    table->mul_f16       = mul_f16;
    table->scale_f16     = scale_f16;
    table->axpy_f16      = axpy_f16;
    table->fmad_f16      = fmad_f16;
    table->reduce_sum_f16 = reduce_sum_f16;
    table->reduce_max_f16 = reduce_max_f16;
    table->reduce_min_f16 = reduce_min_f16;
    table->dot_f16       = dot_f16;
    table->norm2_f16     = norm2_f16;
    table->neg_f16       = neg_f16;
    table->sqrt_f16      = sqrt_f16;
    table->exp_f16       = exp_f16;
    table->log_f16       = log_f16;
    table->relu_f16      = relu_f16;
    table->sigmoid_f16   = sigmoid_f16;
    table->tanh_f16      = tanh_f16;
    table->abs_f16       = abs_f16;
    table->min_f16       = min_f16;
    table->max_f16       = max_f16;
    table->clamp_f16     = clamp_f16;
    table->sqdiff_sum_f16 = sqdiff_sum_f16;
    table->gemv_f16       = gemv_f16;
    table->ger_f16        = ger_f16;
    table->quantize_f16   = quantize_f16;
    table->dequantize_f16 = dequantize_f16;
}
