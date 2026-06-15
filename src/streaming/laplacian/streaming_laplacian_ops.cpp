#include "nerve/streaming/streaming_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nerve::streaming
{

std::vector<double> lanczosIteration(const LaplacianState &L, Size k, double tolerance)
{
    Size n = L.diagonal.size();
    if (k > n)
        k = n;
    std::vector<double> alpha(k), beta(k + 1);
    std::vector<std::vector<double>> Q(k, std::vector<double>(n));
    std::vector<double> q(n, 0.0);
    q[0] = 1.0;
    for (Size i = 0; i < n; ++i)
        Q[0][i] = q[i];
    for (Size j = 0; j < k; ++j)
    {
        std::vector<double> w(n, 0.0);
        for (Size i = 0; i < n; ++i)
        {
            w[i] += L.diagonal[i] * Q[j][i];
            Size begin = L.column_start[i];
            Size end = L.column_start[i + 1];
            for (Size p = begin; p < end; ++p)
            {
                w[i] += L.values[p] * Q[j][L.row_index[p]];
            }
        }
        alpha[j] = 0.0;
        for (Size i = 0; i < n; ++i)
            alpha[j] += Q[j][i] * w[i];
        if (j > 0)
        {
            for (Size i = 0; i < n; ++i)
                w[i] -= beta[j] * Q[j - 1][i];
        }
        for (Size i = 0; i < n; ++i)
            w[i] -= alpha[j] * Q[j][i];
        beta[j + 1] = 0.0;
        for (double x : w)
            beta[j + 1] += x * x;
        beta[j + 1] = std::sqrt(beta[j + 1]);
        if (j + 1 < k)
        {
            if (beta[j + 1] < tolerance)
            {
                k = j + 1;
                break;
            }
            for (Size i = 0; i < n; ++i)
                Q[j + 1][i] = w[i] / beta[j + 1];
        }
    }
    std::vector<double> T(k * k, 0.0);
    for (Size i = 0; i < k; ++i)
    {
        T[i * k + i] = alpha[i];
        if (i + 1 < k)
        {
            T[i * k + i + 1] = beta[i + 1];
            T[(i + 1) * k + i] = beta[i + 1];
        }
    }
    std::vector<double> eig(k);
    if (k == 1)
    {
        eig[0] = T[0];
        return eig;
    }
    for (Size i = 0; i < k; ++i)
        eig[i] = T[i * k + i];
    for (Size iter = 0; iter < 30; ++iter)
    {
        double off = 0.0;
        for (Size i = 0; i < k - 1; ++i)
        {
            off += std::fabs(T[i * k + i + 1]);
        }
        if (off < tolerance)
            break;
        for (Size p = 0; p < k - 1; ++p)
        {
            if (std::fabs(T[p * k + p + 1]) < tolerance)
                continue;
            double f = T[p * k + p];
            double g = T[p * k + p + 1];
            double h = T[(p + 1) * k + p + 1];
            double d = (f - h) / 2.0;
            double t = (d >= 0 ? 1.0 / (d + std::sqrt(d * d + g * g))
                               : 1.0 / (d - std::sqrt(d * d + g * g)));
            double c = 1.0 / std::sqrt(1.0 + t * t);
            double s = t * c;
            for (Size i = 0; i < k; ++i)
            {
                double a = T[i * k + p];
                double b = T[i * k + p + 1];
                T[i * k + p] = c * a - s * b;
                T[i * k + p + 1] = s * a + c * b;
            }
            for (Size i = 0; i < k; ++i)
            {
                double a = T[p * k + i];
                double b = T[(p + 1) * k + i];
                T[p * k + i] = c * a - s * b;
                T[(p + 1) * k + i] = s * a + c * b;
            }
            T[p * k + p + 1] = 0.0;
            if (p > 0)
                T[(p - 1) * k + p] = 0.0;
            T[(p + 1) * k + p] = 0.0;
        }
        for (Size i = 0; i < k; ++i)
            eig[i] = T[i * k + i];
    }
    return eig;
}

} // namespace nerve::streaming
