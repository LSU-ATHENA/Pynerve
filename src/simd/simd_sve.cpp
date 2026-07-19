#include "nerve/cpu/arm_simd.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#if defined(NERVE_HAS_SVE) || defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace nerve::simd::sve_impl
{

#if defined(NERVE_HAS_SVE) || defined(__ARM_FEATURE_SVE)

void memcpy(void *dst, const void *src, std::size_t bytes)
{
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    const auto *s = static_cast<const std::uint8_t *>(src);
    // SVE uses predicate-driven loop -- process in variable-length chunks
    svbool_t pg = svwhilelt_b8(i, bytes);
    while (svptest_any(svptrue_b8(), pg))
    {
        svuint8_t v = svld1_u8(pg, s + i);
        svst1_u8(pg, d + i, v);
        i += svcntb();
        pg = svwhilelt_b8(i, bytes);
    }
}

void memset(void *dst, int value, std::size_t bytes)
{
    std::size_t i = 0;
    auto *d = static_cast<std::uint8_t *>(dst);
    svuint8_t v = svdup_n_u8(static_cast<std::uint8_t>(value));
    svbool_t pg = svwhilelt_b8(i, bytes);
    while (svptest_any(svptrue_b8(), pg))
    {
        svst1_u8(pg, d + i, v);
        i += svcntb();
        pg = svwhilelt_b8(i, bytes);
    }
}

void add(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svadd_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void sub(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svsub_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void mul(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svmul_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void scale(double *a, double alpha, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t va = svdup_n_f64(alpha);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, a + i);
        svst1_f64(pg, a + i, svmul_f64_m(pg, v, va));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void axpy(double alpha, const double *x, double *y, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t va = svdup_n_f64(alpha);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t vx = svld1_f64(pg, x + i);
        svfloat64_t vy = svld1_f64(pg, y + i);
        svst1_f64(pg, y + i, svmla_f64_m(pg, vy, va, vx));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void fmad(const double *a, const double *b, double *c, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svfloat64_t vc = svld1_f64(pg, c + i);
        svst1_f64(pg, c + i, svmla_f64_m(pg, vc, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

double reduce_sum(const double *data, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        vacc = svadd_f64_m(pg, vacc, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svaddv_f64(svptrue_b64(), vacc);
}

double reduce_max(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    std::size_t i = 0;
    svfloat64_t vmax = svdup_n_f64(-std::numeric_limits<double>::infinity());
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        vmax = svmax_f64_m(pg, vmax, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svmaxv_f64(svptrue_b64(), vmax);
}

double reduce_min(const double *data, std::size_t n)
{
    if (n == 0)
        return 0.0;
    std::size_t i = 0;
    svfloat64_t vmin = svdup_n_f64(std::numeric_limits<double>::infinity());
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        vmin = svmin_f64_m(pg, vmin, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svminv_f64(svptrue_b64(), vmin);
}

double dot(const double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        vacc = svmla_f64_m(pg, vacc, va, vb);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    return svaddv_f64(svptrue_b64(), vacc);
}

double norm2(const double *vec, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, vec + i);
        vacc = svmla_f64_m(pg, vacc, v, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    double sum = svaddv_f64(svptrue_b64(), vacc);
    return std::sqrt(sum);
}

double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vacc = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svfloat64_t diff = svsub_f64_m(pg, va, vb);
        vacc = svmla_f64_m(pg, vacc, diff, diff);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
    double sum = svaddv_f64(svptrue_b64(), vacc);
    return sum;
}

void abs(double *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        svst1_f64(pg, data + i, svabs_f64_m(pg, v, v));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void neg(double *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        svst1_f64(pg, data + i, svneg_f64_m(pg, v, v));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void sqrt(double *data, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        svst1_f64(pg, data + i, svsqrt_f64_m(pg, v, v));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void exp(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}

void log(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}

void relu(double *data, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vzero = svdup_n_f64(0.0);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        svst1_f64(pg, data + i, svmax_f64_m(pg, v, vzero));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void sigmoid(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
}

void tanh(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}

void min(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svmin_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void max(double *a, const double *b, std::size_t n)
{
    std::size_t i = 0;
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t va = svld1_f64(pg, a + i);
        svfloat64_t vb = svld1_f64(pg, b + i);
        svst1_f64(pg, a + i, svmax_f64_m(pg, va, vb));
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void clamp(double *data, double lo, double hi, std::size_t n)
{
    std::size_t i = 0;
    svfloat64_t vlo = svdup_n_f64(lo);
    svfloat64_t vhi = svdup_n_f64(hi);
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {
        svfloat64_t v = svld1_f64(pg, data + i);
        v = svmax_f64_m(pg, v, vlo);
        v = svmin_f64_m(pg, v, vhi);
        svst1_f64(pg, data + i, v);
        i += svcntd();
        pg = svwhilelt_b64(i, n);
    }
}

void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    nerve::simd::sve::fmad_f32(a, b, c, n);
}

void add_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::sve::add_f32(a, b, n);
}

void sub_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::sve::sub_f32(a, b, n);
}

void mul_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::sve::mul_f32(a, b, n);
}

void scale_f32(float *a, float alpha, std::size_t n)
{
    nerve::simd::sve::scale_f32(a, alpha, n);
}

void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    nerve::simd::sve::axpy_f32(alpha, x, y, n);
}

float reduce_sum_f32(const float *data, std::size_t n)
{
    return nerve::simd::sve::reduce_sum_f32(data, n);
}

float reduce_max_f32(const float *data, std::size_t n)
{
    return nerve::simd::sve::reduce_max_f32(data, n);
}

float reduce_min_f32(const float *data, std::size_t n)
{
    return nerve::simd::sve::reduce_min_f32(data, n);
}

float dot_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::sve::dot_f32(a, b, n);
}

float norm2_f32(const float *vec, std::size_t n)
{
    return nerve::simd::sve::norm2_f32(vec, n);
}

float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::sve::sqdiff_sum_f32(a, b, n);
}

float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::sve::euclidean_f32(a, b, n);
}

float cosine_f32(const float *a, const float *b, std::size_t n)
{
    return nerve::simd::sve::cosine_f32(a, b, n);
}

void neg_f32(float *data, std::size_t n)
{
    nerve::simd::sve::neg_f32(data, n);
}

void sqrt_f32(float *data, std::size_t n)
{
    nerve::simd::sve::sqrt_f32(data, n);
}

void exp_f32(float *data, std::size_t n)
{
    nerve::simd::sve::exp_f32(data, n);
}

void log_f32(float *data, std::size_t n)
{
    nerve::simd::sve::log_f32(data, n);
}

void sigmoid_f32(float *data, std::size_t n)
{
    nerve::simd::sve::sigmoid_f32(data, n);
}

void tanh_f32(float *data, std::size_t n)
{
    nerve::simd::sve::tanh_f32(data, n);
}

void abs_f32(float *data, std::size_t n)
{
    nerve::simd::sve::abs_f32(data, n);
}

void relu_f32(float *data, std::size_t n)
{
    nerve::simd::sve::relu_f32(data, n);
}

void min_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::sve::min_f32(a, b, n);
}

void max_f32(float *a, const float *b, std::size_t n)
{
    nerve::simd::sve::max_f32(a, b, n);
}

void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    nerve::simd::sve::clamp_f32(data, lo, hi, n);
}

void add_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::sve::add_f16(a, b, n);
}

void sub_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::sve::sub_f16(a, b, n);
}

void mul_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::sve::mul_f16(a, b, n);
}

void scale_f16(half *a, half alpha, std::size_t n)
{
    nerve::simd::sve::scale_f16(a, alpha, n);
}

void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    nerve::simd::sve::axpy_f16(alpha, x, y, n);
}

void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    nerve::simd::sve::fmad_f16(a, b, c, n);
}

float reduce_sum_f16(const half *data, std::size_t n)
{
    return nerve::simd::sve::reduce_sum_f16(data, n);
}

float reduce_max_f16(const half *data, std::size_t n)
{
    return nerve::simd::sve::reduce_max_f16(data, n);
}

float reduce_min_f16(const half *data, std::size_t n)
{
    return nerve::simd::sve::reduce_min_f16(data, n);
}

float dot_f16(const half *a, const half *b, std::size_t n)
{
    return nerve::simd::sve::dot_f16(a, b, n);
}

float norm2_f16(const half *vec, std::size_t n)
{
    return nerve::simd::sve::norm2_f16(vec, n);
}

void neg_f16(half *data, std::size_t n)
{
    nerve::simd::sve::neg_f16(data, n);
}

void sqrt_f16(half *data, std::size_t n)
{
    nerve::simd::sve::sqrt_f16(data, n);
}

void exp_f16(half *data, std::size_t n)
{
    nerve::simd::sve::exp_f16(data, n);
}

void log_f16(half *data, std::size_t n)
{
    nerve::simd::sve::log_f16(data, n);
}

void relu_f16(half *data, std::size_t n)
{
    nerve::simd::sve::relu_f16(data, n);
}

void sigmoid_f16(half *data, std::size_t n)
{
    nerve::simd::sve::sigmoid_f16(data, n);
}

void tanh_f16(half *data, std::size_t n)
{
    nerve::simd::sve::tanh_f16(data, n);
}

void abs_f16(half *data, std::size_t n)
{
    nerve::simd::sve::abs_f16(data, n);
}

void min_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::sve::min_f16(a, b, n);
}

void max_f16(half *a, const half *b, std::size_t n)
{
    nerve::simd::sve::max_f16(a, b, n);
}

void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    nerve::simd::sve::clamp_f16(data, lo, hi, n);
}

float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    return nerve::simd::sve::sqdiff_sum_f16(a, b, n);
}

void gemv_f16(half alpha, const half *A, const half *x, half beta, half *y, std::size_t m,
              std::size_t n)
{
    float fa = half_to_float(alpha);
    float fb = half_to_float(beta);
    if (fb != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] = float_to_half(half_to_float(y[i]) * fb);
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += half_to_float(A[i * n + j]) * half_to_float(x[j]);
        y[i] = float_to_half(half_to_float(y[i]) + fa * sum);
    }
}

void ger_f16(half alpha, const half *x, const half *y, half *A, std::size_t m, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < m; ++i)
    {
        float xi = half_to_float(x[i]);
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] =
                float_to_half(half_to_float(A[i * n + j]) + fa * xi * half_to_float(y[j]));
    }
}

void quantize_f16(const half *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (n == 0)
        return;
    float scale = static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(input[i]);
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        output[i] = static_cast<std::uint8_t>(v * scale + 0.5f);
    }
}

void dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, half *output)
{
    if (n == 0)
        return;
    float inv = 1.0f / static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = float_to_half(static_cast<float>(input[i]) * inv);
}

void gemv_f32(float alpha, const float *A, const float *x, float beta, float *y, std::size_t m,
              std::size_t n)
{
    if (beta != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = nerve::simd::sve::dot_f32(A + i * n, x, n);
        y[i] += alpha * sum;
    }
}

void ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        float axi = alpha * x[i];
        std::size_t j = 0;
        svfloat32_t vaxi = svdup_n_f32(axi);
        svbool_t pg = svwhilelt_b32(j, n);
        while (svptest_any(svptrue_b32(), pg))
        {
            svfloat32_t vy = svld1_f32(pg, y + j);
            svfloat32_t va = svld1_f32(pg, A + i * n + j);
            svst1_f32(pg, A + i * n + j, svmla_f32_m(pg, va, vaxi, vy));
            j += svcntw();
            pg = svwhilelt_b32(j, n);
        }
    }
}

void gemv(double alpha, const double *A, const double *x, double beta, double *y, std::size_t m,
          std::size_t n)
{
    if (beta != 1.0)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        double sum = dot(A + i * n, x, n);
        y[i] += alpha * sum;
    }
}

void ger(double alpha, const double *x, const double *y, double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        double axi = alpha * x[i];
        std::size_t j = 0;
        svfloat64_t vaxi = svdup_n_f64(axi);
        svbool_t pg = svwhilelt_b64(j, n);
        while (svptest_any(svptrue_b64(), pg))
        {
            svfloat64_t vy = svld1_f64(pg, y + j);
            svfloat64_t va = svld1_f64(pg, A + i * n + j);
            svst1_f64(pg, A + i * n + j, svmla_f64_m(pg, va, vaxi, vy));
            j += svcntd();
            pg = svwhilelt_b64(j, n);
        }
    }
}

#else
// Stub -- SVE not available at compile time
void memcpy(void *dst, const void *src, std::size_t bytes)
{
    std::memcpy(dst, src, bytes);
}
void memset(void *dst, int value, std::size_t bytes)
{
    std::memset(dst, value, bytes);
}
void add(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
}
void sub(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
}
void mul(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
}
void scale(double *a, double alpha, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
}
void axpy(double alpha, const double *x, double *y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}
void fmad(const double *a, const double *b, double *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
}
double reduce_sum(const double *data, std::size_t n)
{
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += data[i];
    return s;
}
double reduce_max(const double *data, std::size_t n)
{
    if (n == 0)
        return 0;
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}
double reduce_min(const double *data, std::size_t n)
{
    if (n == 0)
        return 0;
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}
double dot(const double *a, const double *b, std::size_t n)
{
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
double norm2(const double *vec, std::size_t n)
{
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += vec[i] * vec[i];
    return std::sqrt(s);
}
double sqdiff_sum(const double *a, const double *b, std::size_t n)
{
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
void abs(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
}
void neg(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
}
void sqrt(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::sqrt(data[i]);
}
void exp(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}
void log(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}
void relu(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0)
            data[i] = 0;
}
void sigmoid(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
}
void tanh(double *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}
void min(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}
void max(double *a, const double *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}
void clamp(double *data, double lo, double hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}
void fmad_f32(const float *a, const float *b, float *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] += a[i] * b[i];
}
void add_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] += b[i];
}
void sub_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] -= b[i];
}
void mul_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= b[i];
}
void scale_f32(float *a, float alpha, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] *= alpha;
}
void axpy_f32(float alpha, const float *x, float *y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}
float reduce_sum_f32(const float *data, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += data[i];
    return s;
}
float reduce_max_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0;
    float m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] > m)
            m = data[i];
    return m;
}
float reduce_min_f32(const float *data, std::size_t n)
{
    if (n == 0)
        return 0;
    float m = data[0];
    for (std::size_t i = 1; i < n; ++i)
        if (data[i] < m)
            m = data[i];
    return m;
}
float dot_f32(const float *a, const float *b, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
float norm2_f32(const float *vec, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += vec[i] * vec[i];
    return std::sqrt(s);
}
void neg_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = -data[i];
}
void sqrt_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::sqrt(data[i]);
}
void exp_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::exp(data[i]);
}
void log_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::log(data[i]);
}
void sigmoid_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
}
void tanh_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::tanh(data[i]);
}
float sqdiff_sum_f32(const float *a, const float *b, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
float euclidean_f32(const float *a, const float *b, std::size_t n)
{
    return std::sqrt(sqdiff_sum_f32(a, b, n));
}
float cosine_f32(const float *a, const float *b, std::size_t n)
{
    float d = dot_f32(a, b, n);
    float na = norm2_f32(a, n);
    float nb = norm2_f32(b, n);
    if (na == 0 || nb == 0)
        return 1.0f;
    float cs = d / (na * nb);
    if (cs < -1.0f)
        cs = -1.0f;
    if (cs > 1.0f)
        cs = 1.0f;
    return 1.0f - cs;
}
void abs_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = std::abs(data[i]);
}
void relu_f32(float *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0)
            data[i] = 0;
}
void min_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] < b[i] ? a[i] : b[i];
}
void max_f32(float *a, const float *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = a[i] > b[i] ? a[i] : b[i];
}
void clamp_f32(float *data, float lo, float hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (data[i] < lo)
            data[i] = lo;
        if (data[i] > hi)
            data[i] = hi;
    }
}
void add_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) + half_to_float(b[i]));
}
void sub_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) - half_to_float(b[i]));
}
void mul_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * half_to_float(b[i]));
}
void scale_f16(half *a, half alpha, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < n; ++i)
        a[i] = float_to_half(half_to_float(a[i]) * fa);
}
void axpy_f16(half alpha, const half *x, half *y, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = float_to_half(half_to_float(y[i]) + fa * half_to_float(x[i]));
}
void fmad_f16(const half *a, const half *b, half *c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        c[i] = float_to_half(half_to_float(c[i]) + half_to_float(a[i]) * half_to_float(b[i]));
}
float reduce_sum_f16(const half *data, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += half_to_float(data[i]);
    return s;
}
float reduce_max_f16(const half *data, std::size_t n)
{
    if (n == 0)
        return 0;
    float m = half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v > m)
            m = v;
    }
    return m;
}
float reduce_min_f16(const half *data, std::size_t n)
{
    if (n == 0)
        return 0;
    float m = half_to_float(data[0]);
    for (std::size_t i = 1; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < m)
            m = v;
    }
    return m;
}
float dot_f16(const half *a, const half *b, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += half_to_float(a[i]) * half_to_float(b[i]);
    return s;
}
float norm2_f16(const half *vec, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(vec[i]);
        s += v * v;
    }
    return std::sqrt(s);
}
void neg_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(-half_to_float(data[i]));
}
void sqrt_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::sqrt(half_to_float(data[i])));
}
void exp_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::exp(half_to_float(data[i])));
}
void log_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::log(half_to_float(data[i])));
}
void relu_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(v < 0 ? 0 : v);
    }
}
void sigmoid_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        data[i] = float_to_half(1.0f / (1.0f + std::exp(-v)));
    }
}
void tanh_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::tanh(half_to_float(data[i])));
}
void abs_f16(half *data, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        data[i] = float_to_half(std::abs(half_to_float(data[i])));
}
void min_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa < fb ? fa : fb);
    }
}
void max_f16(half *a, const half *b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float fa = half_to_float(a[i]);
        float fb = half_to_float(b[i]);
        a[i] = float_to_half(fa > fb ? fa : fb);
    }
}
void clamp_f16(half *data, half lo, half hi, std::size_t n)
{
    float flo = half_to_float(lo);
    float fhi = half_to_float(hi);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(data[i]);
        if (v < flo)
            v = flo;
        if (v > fhi)
            v = fhi;
        data[i] = float_to_half(v);
    }
}
float sqdiff_sum_f16(const half *a, const half *b, std::size_t n)
{
    float s = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        float d = half_to_float(a[i]) - half_to_float(b[i]);
        s += d * d;
    }
    return s;
}
void gemv_f32(float alpha, const float *A, const float *x, float beta, float *y, std::size_t m,
              std::size_t n)
{
    if (beta != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        float s = 0;
        for (std::size_t j = 0; j < n; ++j)
            s += A[i * n + j] * x[j];
        y[i] += alpha * s;
    }
}
void ger_f32(float alpha, const float *x, const float *y, float *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}
void gemv_f16(half alpha, const half *A, const half *x, half beta, half *y, std::size_t m,
              std::size_t n)
{
    float fa = half_to_float(alpha);
    float fb = half_to_float(beta);
    if (fb != 1.0f)
        for (std::size_t i = 0; i < m; ++i)
            y[i] = float_to_half(half_to_float(y[i]) * fb);
    for (std::size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            sum += half_to_float(A[i * n + j]) * half_to_float(x[j]);
        y[i] = float_to_half(half_to_float(y[i]) + fa * sum);
    }
}
void ger_f16(half alpha, const half *x, const half *y, half *A, std::size_t m, std::size_t n)
{
    float fa = half_to_float(alpha);
    for (std::size_t i = 0; i < m; ++i)
    {
        float xi = half_to_float(x[i]);
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] =
                float_to_half(half_to_float(A[i * n + j]) + fa * xi * half_to_float(y[j]));
    }
}
void quantize_f16(const half *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (n == 0)
        return;
    float scale = static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = half_to_float(input[i]);
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        output[i] = static_cast<std::uint8_t>(v * scale + 0.5f);
    }
}
void dequantize_f16(const std::uint8_t *input, std::size_t n, int bits, half *output)
{
    if (n == 0)
        return;
    float inv = 1.0f / static_cast<float>((1 << bits) - 1);
    for (std::size_t i = 0; i < n; ++i)
        output[i] = float_to_half(static_cast<float>(input[i]) * inv);
}
void gemv(double alpha, const double *A, const double *x, double beta, double *y, std::size_t m,
          std::size_t n)
{
    if (beta != 1.0)
        for (std::size_t i = 0; i < m; ++i)
            y[i] *= beta;
    for (std::size_t i = 0; i < m; ++i)
    {
        double s = 0;
        for (std::size_t j = 0; j < n; ++j)
            s += A[i * n + j] * x[j];
        y[i] += alpha * s;
    }
}
void ger(double alpha, const double *x, const double *y, double *A, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}
#endif

} // namespace nerve::simd::sve_impl

extern "C" void nerve_simd_assign_sve(nerve::simd::SimdDispatchTable *table)
{
    using namespace nerve::simd::sve_impl;
    table->memcpy = memcpy;
    table->memset = memset;
    table->add = add;
    table->sub = sub;
    table->mul = mul;
    table->scale = scale;
    table->axpy = axpy;
    table->fmad = fmad;
    table->reduce_sum = reduce_sum;
    table->reduce_max = reduce_max;
    table->reduce_min = reduce_min;
    table->dot = dot;
    table->norm2 = norm2;
    table->sqdiff_sum = sqdiff_sum;
    table->abs = abs;
    table->neg = neg;
    table->sqrt = sqrt;
    table->exp = exp;
    table->log = log;
    table->relu = relu;
    table->sigmoid = sigmoid;
    table->tanh = tanh;
    table->min = min;
    table->max = max;
    table->clamp = clamp;
    table->fmad_f32 = fmad_f32;
    table->add_f32 = add_f32;
    table->sub_f32 = sub_f32;
    table->mul_f32 = mul_f32;
    table->scale_f32 = scale_f32;
    table->axpy_f32 = axpy_f32;
    table->reduce_sum_f32 = reduce_sum_f32;
    table->reduce_max_f32 = reduce_max_f32;
    table->reduce_min_f32 = reduce_min_f32;
    table->dot_f32 = dot_f32;
    table->norm2_f32 = norm2_f32;
    table->neg_f32 = neg_f32;
    table->sqrt_f32 = sqrt_f32;
    table->exp_f32 = exp_f32;
    table->log_f32 = log_f32;
    table->sigmoid_f32 = sigmoid_f32;
    table->tanh_f32 = tanh_f32;
    table->sqdiff_sum_f32 = sqdiff_sum_f32;
    table->euclidean_f32 = euclidean_f32;
    table->cosine_f32 = cosine_f32;
    table->abs_f32 = abs_f32;
    table->relu_f32 = relu_f32;
    table->min_f32 = min_f32;
    table->max_f32 = max_f32;
    table->clamp_f32 = clamp_f32;
    table->gemv_f32 = gemv_f32;
    table->ger_f32 = ger_f32;
    table->gemv = gemv;
    table->ger = ger;
    table->add_f16 = add_f16;
    table->sub_f16 = sub_f16;
    table->mul_f16 = mul_f16;
    table->scale_f16 = scale_f16;
    table->axpy_f16 = axpy_f16;
    table->fmad_f16 = fmad_f16;
    table->reduce_sum_f16 = reduce_sum_f16;
    table->reduce_max_f16 = reduce_max_f16;
    table->reduce_min_f16 = reduce_min_f16;
    table->dot_f16 = dot_f16;
    table->norm2_f16 = norm2_f16;
    table->neg_f16 = neg_f16;
    table->sqrt_f16 = sqrt_f16;
    table->exp_f16 = exp_f16;
    table->log_f16 = log_f16;
    table->relu_f16 = relu_f16;
    table->sigmoid_f16 = sigmoid_f16;
    table->tanh_f16 = tanh_f16;
    table->abs_f16 = abs_f16;
    table->min_f16 = min_f16;
    table->max_f16 = max_f16;
    table->clamp_f16 = clamp_f16;
    table->sqdiff_sum_f16 = sqdiff_sum_f16;
    table->gemv_f16 = gemv_f16;
    table->ger_f16 = ger_f16;
    table->quantize_f16 = quantize_f16;
    table->dequantize_f16 = dequantize_f16;
}
