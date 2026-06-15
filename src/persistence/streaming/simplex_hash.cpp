#include "nerve/persistence/streaming/streaming_reducer.hpp"

namespace nerve::persistence::streaming
{

SimplexHash::SimplexHash(size_t n_points)
    : n_points_(n_points)
{
    binom_.resize(n_points + 1);
    for (size_t n = 0; n <= n_points; ++n)
    {
        binom_[n].resize(n + 1);
        binom_[n][0] = binom_[n][n] = 1;
        for (size_t k = 1; k < n; ++k)
        {
            binom_[n][k] = binom_[n - 1][k - 1] + binom_[n - 1][k];
        }
    }
}

size_t SimplexHash::encode(const std::vector<int> &vertices) const
{
    size_t result = 0;

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        int vi = vertices[i];
        int ki = static_cast<int>(i);
        result += binomial(vi, ki);
    }

    return result;
}

std::vector<int> SimplexHash::decode(size_t index, int dim) const
{
    std::vector<int> vertices;
    if (dim < 0)
    {
        return vertices;
    }
    int k = dim;
    size_t remaining = index;

    for (int i = k; i >= 0; --i)
    {
        int v = i;
        while (v < static_cast<int>(n_points_) && binomial(v + 1, i) <= remaining)
        {
            v++;
        }
        vertices.push_back(v);
        remaining -= binomial(v, i);
    }

    std::reverse(vertices.begin(), vertices.end());
    return vertices;
}

size_t SimplexHash::binomial(int n, int k) const
{
    if (k < 0 || k > n || n < 0)
        return 0;
    if (static_cast<size_t>(n) >= binom_.size() ||
        static_cast<size_t>(k) >= binom_[static_cast<size_t>(n)].size())
    {
        return 0;
    }
    return binom_[n][k];
}

} // namespace nerve::persistence::streaming
