#include "nerve/algorithms/persistence_vectorization.hpp"
#include "nerve/cpu/arm_simd.hpp"
#include "nerve/cpu/simd.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nerve::algorithms
{

namespace
{

#if defined(__AVX2__) || defined(__AVX512F__)
inline __m256d simd_exp_pd_avx(__m256d x)
{
    const __m256d max_input = _mm256_set1_pd(709.0);
    const __m256d min_input = _mm256_set1_pd(-709.0);
    x = _mm256_min_pd(x, max_input);
    x = _mm256_max_pd(x, min_input);

    const __m256d inv_ln2 = _mm256_set1_pd(1.44269504088896340736);

    __m256d n = _mm256_floor_pd(_mm256_add_pd(_mm256_mul_pd(x, inv_ln2), _mm256_set1_pd(0.5)));
    __m256d r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0.6931471805599453), x);

    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d r3 = _mm256_mul_pd(r2, r);
    __m256d r4 = _mm256_mul_pd(r2, r2);
    __m256d r5 = _mm256_mul_pd(r3, r2);

    const __m256d c1 = _mm256_set1_pd(1.0);
    const __m256d c2 = _mm256_set1_pd(1.0 / 2.0);
    const __m256d c3 = _mm256_set1_pd(1.0 / 6.0);
    const __m256d c4 = _mm256_set1_pd(1.0 / 24.0);
    const __m256d c5 = _mm256_set1_pd(1.0 / 120.0);

    __m256d poly = _mm256_fmadd_pd(
        c5, r5,
        _mm256_fmadd_pd(
            c4, r4, _mm256_fmadd_pd(c3, r3, _mm256_fmadd_pd(c2, r2, _mm256_fmadd_pd(c1, r, c1)))));

    alignas(32) double n_arr[4];
    alignas(32) double p_arr[4];
    _mm256_store_pd(n_arr, n);
    _mm256_store_pd(p_arr, poly);

    for (int i = 0; i < 4; ++i)
    {
        int64_t ni = static_cast<int64_t>(n_arr[i]);
        int64_t exp_bits = (ni + 1023) << 52;
        int64_t bits;
        std::memcpy(&bits, &p_arr[i], sizeof(double));
        bits += exp_bits;
        std::memcpy(&p_arr[i], &bits, sizeof(double));
    }

    return _mm256_load_pd(p_arr);
}
#endif

#if defined(__AVX512F__)
inline __m512d simd_exp_pd_avx512(__m512d x)
{
    const __m512d max_input = _mm512_set1_pd(709.0);
    const __m512d min_input = _mm512_set1_pd(-709.0);
    x = _mm512_min_pd(x, max_input);
    x = _mm512_max_pd(x, min_input);

    const __m512d inv_ln2 = _mm512_set1_pd(1.44269504088896340736);

    __m512d n = _mm512_floor_pd(_mm512_add_pd(_mm512_mul_pd(x, inv_ln2), _mm512_set1_pd(0.5)));
    __m512d r = _mm512_fnmadd_pd(n, _mm512_set1_pd(0.6931471805599453), x);

    __m512d r2 = _mm512_mul_pd(r, r);
    __m512d r3 = _mm512_mul_pd(r2, r);
    __m512d r4 = _mm512_mul_pd(r2, r2);
    __m512d r5 = _mm512_mul_pd(r3, r2);

    const __m512d c1 = _mm512_set1_pd(1.0);
    const __m512d c2 = _mm512_set1_pd(1.0 / 2.0);
    const __m512d c3 = _mm512_set1_pd(1.0 / 6.0);
    const __m512d c4 = _mm512_set1_pd(1.0 / 24.0);
    const __m512d c5 = _mm512_set1_pd(1.0 / 120.0);

    __m512d poly = _mm512_fmadd_pd(
        c5, r5,
        _mm512_fmadd_pd(
            c4, r4, _mm512_fmadd_pd(c3, r3, _mm512_fmadd_pd(c2, r2, _mm512_fmadd_pd(c1, r, c1)))));

    alignas(64) double n_arr[8];
    alignas(64) double p_arr[8];
    _mm512_store_pd(n_arr, n);
    _mm512_store_pd(p_arr, poly);

    for (int i = 0; i < 8; ++i)
    {
        int64_t ni = static_cast<int64_t>(n_arr[i]);
        int64_t exp_bits = (ni + 1023) << 52;
        int64_t bits;
        std::memcpy(&bits, &p_arr[i], sizeof(double));
        bits += exp_bits;
        std::memcpy(&p_arr[i], &bits, sizeof(double));
    }

    return _mm512_load_pd(p_arr);
}
#endif

} // namespace

template <typename T>
PersistenceLandscape compute_landscape(std::span<const T> diagram, size_t num_pairs, int num_levels,
                                       double resolution)
{
    PersistenceLandscape result;
    result.num_levels = num_levels;

    T b_min = std::numeric_limits<T>::max();
    T d_max = std::numeric_limits<T>::lowest();
    std::vector<std::pair<T, T>> finite_pairs;
    for (size_t i = 0; i < num_pairs; ++i)
    {
        T b = diagram[i * 2];
        T d = diagram[i * 2 + 1];
        if (std::isfinite(d) && d > b)
        {
            b_min = std::min(b_min, b);
            d_max = std::max(d_max, d);
            finite_pairs.emplace_back(b, d);
        }
    }

    if (finite_pairs.empty())
    {
        result.x_min = 0;
        result.x_max = 1;
        result.landscape_levels.resize(num_levels);
        return result;
    }

    result.x_min = static_cast<double>(b_min);
    result.x_max = static_cast<double>(d_max);
    double range = result.x_max - result.x_min;
    if (range <= 0)
        range = 1.0;

    int num_points = std::max(100, static_cast<int>(range / resolution));
    double dx = range / static_cast<double>(num_points - 1);

    std::vector<std::vector<double>> tent_values(
        finite_pairs.size(), std::vector<double>(static_cast<size_t>(num_points), 0.0));

    for (size_t p = 0; p < finite_pairs.size(); ++p)
    {
        double b = static_cast<double>(finite_pairs[p].first);
        double d = static_cast<double>(finite_pairs[p].second);
        double mid = (b + d) / 2.0;
#if defined(NERVE_USE_SIMD) && defined(__AVX2__)
        {
            __m256d b_vec = _mm256_set1_pd(b);
            __m256d d_vec = _mm256_set1_pd(d);
            __m256d mid_vec = _mm256_set1_pd(mid);
            __m256d zero_vec = _mm256_set1_pd(0.0);
            __m256d dx_offsets = _mm256_set_pd(3.0 * dx, 2.0 * dx, dx, 0.0);

            int i = 0;
            for (; i + 4 <= num_points; i += 4)
            {
                double x0 = result.x_min + static_cast<double>(i) * dx;
                __m256d x_vec = _mm256_add_pd(_mm256_set1_pd(x0), dx_offsets);

                __m256d mask1 = _mm256_and_pd(_mm256_cmp_pd(x_vec, b_vec, _CMP_GE_OQ),
                                              _mm256_cmp_pd(x_vec, mid_vec, _CMP_LE_OQ));
                __m256d val1 = _mm256_sub_pd(x_vec, b_vec);

                __m256d mask2 = _mm256_and_pd(_mm256_cmp_pd(x_vec, mid_vec, _CMP_GT_OQ),
                                              _mm256_cmp_pd(x_vec, d_vec, _CMP_LE_OQ));
                __m256d val2 = _mm256_sub_pd(d_vec, x_vec);

                __m256d result_vec = _mm256_blendv_pd(zero_vec, val1, mask1);
                result_vec = _mm256_blendv_pd(result_vec, val2, mask2);

                alignas(32) double res[4];
                _mm256_store_pd(res, result_vec);
                for (int k = 0; k < 4; ++k)
                {
                    tent_values[p][static_cast<size_t>(i + k)] = res[k];
                }
            }
            for (; i < num_points; ++i)
            {
                double x = result.x_min + static_cast<double>(i) * dx;
                if (x >= b && x <= mid)
                {
                    tent_values[p][static_cast<size_t>(i)] = x - b;
                }
                else if (x > mid && x <= d)
                {
                    tent_values[p][static_cast<size_t>(i)] = d - x;
                }
            }
        }
#else
        for (int i = 0; i < num_points; ++i)
        {
            double x = result.x_min + static_cast<double>(i) * dx;
            if (x >= b && x <= mid)
            {
                tent_values[p][static_cast<size_t>(i)] = x - b;
            }
            else if (x > mid && x <= d)
            {
                tent_values[p][static_cast<size_t>(i)] = d - x;
            }
        }
#endif
    }

    result.landscape_levels.resize(static_cast<size_t>(num_levels));
    for (int k = 0; k < num_levels; ++k)
    {
        result.landscape_levels[static_cast<size_t>(k)].resize(static_cast<size_t>(num_points),
                                                               0.0);
        for (int i = 0; i < num_points; ++i)
        {
            std::vector<double> vals;
            vals.reserve(finite_pairs.size());
            for (size_t p = 0; p < finite_pairs.size(); ++p)
            {
                if (tent_values[p][static_cast<size_t>(i)] > 0)
                {
                    vals.push_back(tent_values[p][static_cast<size_t>(i)]);
                }
            }
            if (!vals.empty())
            {
                std::sort(vals.begin(), vals.end(), std::greater<>());
                if (static_cast<size_t>(k) < vals.size())
                {
                    result.landscape_levels[static_cast<size_t>(k)][static_cast<size_t>(i)] =
                        vals[static_cast<size_t>(k)];
                }
            }
        }
    }

    return result;
}

template <typename T>
PersistenceImage compute_persistence_image(std::span<const T> diagram, size_t num_pairs,
                                           int resolution, double sigma)
{
    PersistenceImage result;
    result.resolution = resolution;
    result.sigma = sigma;

    T b_min = std::numeric_limits<T>::max(), b_max = std::numeric_limits<T>::lowest();
    T p_min = std::numeric_limits<T>::max(), p_max = std::numeric_limits<T>::lowest();

    std::vector<std::pair<T, T>> finite_pairs;
    for (size_t i = 0; i < num_pairs; ++i)
    {
        T b = diagram[i * 2];
        T d = diagram[i * 2 + 1];
        if (std::isfinite(d) && d > b)
        {
            T p = d - b;
            b_min = std::min(b_min, b);
            b_max = std::max(b_max, b);
            p_min = std::min(p_min, p);
            p_max = std::max(p_max, p);
            finite_pairs.emplace_back(b, d);
        }
    }

    if (finite_pairs.empty())
    {
        result.image =
            std::vector<std::vector<double>>(resolution, std::vector<double>(resolution, 0.0));
        return result;
    }

    result.birth_min = static_cast<double>(b_min);
    result.birth_max = static_cast<double>(b_max);
    result.persistence_min = static_cast<double>(p_min);
    result.persistence_max = static_cast<double>(p_max);

    double b_range = result.birth_max - result.birth_min;
    double p_range = result.persistence_max - result.persistence_min;
    if (b_range <= 0)
        b_range = 1.0;
    if (p_range <= 0)
        p_range = 1.0;

    result.image =
        std::vector<std::vector<double>>(resolution, std::vector<double>(resolution, 0.0));
    double sigma_sq2 = 2.0 * sigma * sigma;
    double neg_inv_sigma_sq2 = -1.0 / sigma_sq2;

    for (const auto &[b, d] : finite_pairs)
    {
        double birth = static_cast<double>(b);
        double pers = static_cast<double>(d - b);
        double bx = (birth - result.birth_min) / b_range * (resolution - 1);
        double py = (pers - result.persistence_min) / p_range * (resolution - 1);

        int x0 = std::max(0, static_cast<int>(bx - 3 * sigma));
        int x1 = std::min(resolution - 1, static_cast<int>(bx + 3 * sigma));
        int y0 = std::max(0, static_cast<int>(py - 3 * sigma));
        int y1 = std::min(resolution - 1, static_cast<int>(py + 3 * sigma));

#if defined(NERVE_USE_SIMD) && defined(__AVX512F__)
        {
            for (int x = x0; x <= x1; ++x)
            {
                double dx = static_cast<double>(x) - bx;
                __m512d dx2_broadcast = _mm512_set1_pd(dx * dx);
                __m512d neg_inv_sig = _mm512_set1_pd(neg_inv_sigma_sq2);
                __m512d py_broadcast = _mm512_set1_pd(py);
                __m512d y_offsets = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

                int y = y0;
                for (; y + 8 <= y1 + 1; y += 8)
                {
                    __m512d y_base = _mm512_set1_pd(static_cast<double>(y));
                    __m512d y_vals = _mm512_add_pd(y_base, y_offsets);
                    __m512d dy = _mm512_sub_pd(y_vals, py_broadcast);
                    __m512d dy2 = _mm512_mul_pd(dy, dy);
                    __m512d dist2 = _mm512_add_pd(dx2_broadcast, dy2);
                    __m512d exp_arg = _mm512_mul_pd(dist2, neg_inv_sig);
                    __m512d weight_vec = simd_exp_pd_avx512(exp_arg);

                    alignas(64) double weights[8];
                    _mm512_store_pd(weights, weight_vec);
                    for (int k = 0; k < 8; ++k)
                    {
                        result.image[static_cast<size_t>(y + k)][static_cast<size_t>(x)] +=
                            weights[k];
                    }
                }
                for (; y <= y1; ++y)
                {
                    double dy = static_cast<double>(y) - py;
                    double weight = std::exp(-(dx * dx + dy * dy) / sigma_sq2);
                    result.image[static_cast<size_t>(y)][static_cast<size_t>(x)] += weight;
                }
            }
        }
#elif defined(NERVE_USE_SIMD) && defined(__AVX2__)
        {
            for (int x = x0; x <= x1; ++x)
            {
                double dx = static_cast<double>(x) - bx;
                __m256d dx2_broadcast = _mm256_set1_pd(dx * dx);
                __m256d neg_inv_sig = _mm256_set1_pd(neg_inv_sigma_sq2);
                __m256d py_broadcast = _mm256_set1_pd(py);
                __m256d y_offsets = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

                int y = y0;
                for (; y + 4 <= y1 + 1; y += 4)
                {
                    __m256d y_base = _mm256_set1_pd(static_cast<double>(y));
                    __m256d y_vals = _mm256_add_pd(y_base, y_offsets);
                    __m256d dy = _mm256_sub_pd(y_vals, py_broadcast);
                    __m256d dy2 = _mm256_mul_pd(dy, dy);
                    __m256d dist2 = _mm256_add_pd(dx2_broadcast, dy2);
                    __m256d exp_arg = _mm256_mul_pd(dist2, neg_inv_sig);
                    __m256d weight_vec = simd_exp_pd_avx(exp_arg);

                    alignas(32) double weights[4];
                    _mm256_store_pd(weights, weight_vec);
                    for (int k = 0; k < 4; ++k)
                    {
                        result.image[static_cast<size_t>(y + k)][static_cast<size_t>(x)] +=
                            weights[k];
                    }
                }
                for (; y <= y1; ++y)
                {
                    double dy = static_cast<double>(y) - py;
                    double weight = std::exp(-(dx * dx + dy * dy) / sigma_sq2);
                    result.image[static_cast<size_t>(y)][static_cast<size_t>(x)] += weight;
                }
            }
        }
#elif defined(NERVE_USE_SIMD) && (NERVE_HAS_SVE || NERVE_HAS_NEON)
        {
            for (int x = x0; x <= x1; ++x)
            {
                double dx = static_cast<double>(x) - bx;
                for (int y = y0; y <= y1; ++y)
                {
                    double dy = static_cast<double>(y) - py;
                    double weight = std::exp(-(dx * dx + dy * dy) / sigma_sq2);
                    result.image[static_cast<size_t>(y)][static_cast<size_t>(x)] += weight;
                }
            }
        }
#else
        for (int x = x0; x <= x1; ++x)
        {
            double dx = static_cast<double>(x) - bx;
            for (int y = y0; y <= y1; ++y)
            {
                double dy = static_cast<double>(y) - py;
                double weight = std::exp(-(dx * dx + dy * dy) / sigma_sq2);
                result.image[static_cast<size_t>(y)][static_cast<size_t>(x)] += weight;
            }
        }
#endif
    }

    return result;
}

template <typename T>
std::vector<std::pair<double, int>> compute_betti_curve(std::span<const T> diagram,
                                                        size_t num_pairs, int max_dim)
{
    std::vector<std::pair<double, int>> events;

    for (size_t i = 0; i < num_pairs; ++i)
    {
        T b = diagram[i * 2];
        T d = diagram[i * 2 + 1];
        events.emplace_back(static_cast<double>(b), 1);
        if (std::isfinite(d))
        {
            events.emplace_back(static_cast<double>(d), -1);
        }
    }

    if (events.empty())
        return {};

    std::sort(events.begin(), events.end());
    std::vector<std::pair<double, int>> curve;
    int count = 0;
    double prev_x = events[0].first;

    for (const auto &[x, delta] : events)
    {
        if (x > prev_x + 1e-10)
        {
            curve.emplace_back(prev_x, count);
        }
        count += delta;
        prev_x = x;
    }
    curve.emplace_back(prev_x, count);

    return curve;
}

template PersistenceLandscape compute_landscape<float>(std::span<const float>, size_t, int, double);
template PersistenceLandscape compute_landscape<double>(std::span<const double>, size_t, int,
                                                        double);
template PersistenceImage compute_persistence_image<float>(std::span<const float>, size_t, int,
                                                           double);
template PersistenceImage compute_persistence_image<double>(std::span<const double>, size_t, int,
                                                            double);
template std::vector<std::pair<double, int>> compute_betti_curve<float>(std::span<const float>,
                                                                        size_t, int);
template std::vector<std::pair<double, int>> compute_betti_curve<double>(std::span<const double>,
                                                                         size_t, int);

} // namespace nerve::algorithms
