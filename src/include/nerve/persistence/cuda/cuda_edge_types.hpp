#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace nerve::persistence::accelerated
{

struct Edge
{
    int u = -1;
    int v = -1;
    double w = 0.0;

    __host__ __device__ Edge()
        : u(-1)
        , v(-1)
        , w(0.0)
    {}
    __host__ __device__ Edge(int u_, int v_, double w_)
        : u(u_)
        , v(v_)
        , w(w_)
    {}

    bool operator==(const Edge &other) const
    {
        return u == other.u && v == other.v && std::abs(w - other.w) < 1e-10;
    }
    bool operator!=(const Edge &other) const { return !(*this == other); }
    bool operator<(const Edge &other) const
    {
        if (u != other.u)
            return u < other.u;
        if (v != other.v)
            return v < other.v;
        return w < other.w;
    }
    double length() const { return w; }
    bool isValid() const { return u >= 0 && v >= 0 && u < v && std::isfinite(w) && w > 0.0; }
};

} // namespace nerve::persistence::accelerated
