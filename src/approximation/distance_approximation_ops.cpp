#include "nerve/approximation/distance_approximation.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace nerve::approximation
{

template <typename T>
void jlProjectRandom(const T *points, Size n, Size dim, Size target_dim, T *projected)
{
    static constexpr T kScale = 3.0;
    std::vector<T> proj(target_dim * dim);
    for (Size i = 0; i < target_dim * dim; ++i)
        proj[i] = (static_cast<T>(rand()) / RAND_MAX - T{0.5}) * kScale;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < target_dim; ++j)
        {
            T sum = T{0};
            for (Size k = 0; k < dim; ++k)
                sum += points[i * dim + k] * proj[j * dim + k];
            projected[i * target_dim + j] = sum;
        }
    }
}

template <typename T>
T computeEpsilonNetRadius(const T *density, Size n)
{
    T total = T{0};
    for (Size i = 0; i < n; ++i)
        total += density[i];
    T mean = total / static_cast<T>(n);
    T sq_total = T{0};
    for (Size i = 0; i < n; ++i)
    {
        T d = density[i] - mean;
        sq_total += d * d;
    }
    T stddev = std::sqrt(sq_total / static_cast<T>(n));
    return mean + T{2.0} * stddev;
}

template <typename T>
Size selectLandmarksUniform(const T *points, Size n, Size k, Size *landmarks)
{
    (void)points;
    if (k >= n)
    {
        for (Size i = 0; i < n; ++i)
            landmarks[i] = i;
        return n;
    }
    std::vector<Size> indices(n);
    for (Size i = 0; i < n; ++i)
        indices[i] = i;
    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);
    for (Size i = 0; i < k; ++i)
        landmarks[i] = indices[i];
    return k;
}

template <typename T>
Size selectLandmarksMaxmin(const T *points, Size n, Size dim, Size k, Size *landmarks, T *max_dist)
{
    if (n == 0)
        return 0;
    if (k >= n)
    {
        for (Size i = 0; i < n; ++i)
            landmarks[i] = i;
        return n;
    }

    std::vector<T> min_dist(n, std::numeric_limits<T>::max());
    landmarks[0] = 0;
    Size count = 1;
    T max_d = T{0};

    for (Size i = 0; i < n; ++i)
    {
        T d = T{0};
        for (Size d_ = 0; d_ < dim; ++d_)
            d += (points[i * dim + d_] - points[landmarks[0] * dim + d_]) *
                 (points[i * dim + d_] - points[landmarks[0] * dim + d_]);
        min_dist[i] = d;
        if (d > max_d)
        {
            max_d = d;
            landmarks[1] = i;
        }
    }

    while (count < k)
    {
        Size next = 0;
        max_d = T{0};
        for (Size i = 0; i < n; ++i)
        {
            T d = T{0};
            for (Size d_ = 0; d_ < dim; ++d_)
                d += (points[i * dim + d_] - points[landmarks[count] * dim + d_]) *
                     (points[i * dim + d_] - points[landmarks[count] * dim + d_]);
            min_dist[i] = std::min(min_dist[i], d);
            if (min_dist[i] > max_d)
            {
                max_d = min_dist[i];
                next = i;
            }
        }
        landmarks[++count] = next;
    }
    *max_dist = std::sqrt(max_d);
    return count;
}

template void jlProjectRandom<float>(const float *, Size, Size, Size, float *);
template void jlProjectRandom<double>(const double *, Size, Size, Size, double *);
template float computeEpsilonNetRadius<float>(const float *, Size);
template double computeEpsilonNetRadius<double>(const double *, Size);
template Size selectLandmarksUniform<float>(const float *, Size, Size, Size *);
template Size selectLandmarksUniform<double>(const double *, Size, Size, Size *);
template Size selectLandmarksMaxmin<float>(const float *, Size, Size, Size, Size *, float *);
template Size selectLandmarksMaxmin<double>(const double *, Size, Size, Size, Size *, double *);

} // namespace nerve::approximation
