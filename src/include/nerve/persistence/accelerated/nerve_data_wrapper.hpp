
#pragma once

#include "nerve/core.hpp"

#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

class NerveDataWrapper
{
public:
    static core::BufferView<const double> createBufferView(const std::vector<double> &data)
    {
        return core::BufferView<const double>(data.data(), data.size());
    }

    static core::BufferView<const double> createBufferView(const double *data, size_t size)
    {
        return core::BufferView<const double>(data, size);
    }

    static core::BufferView<const int> createBufferView(const std::vector<int> &data)
    {
        return core::BufferView<const int>(data.data(), data.size());
    }

    static core::BufferView<const int> createBufferView(const int *data, size_t size)
    {
        return core::BufferView<const int>(data, size);
    }

    template <typename T>
    static core::BufferView<const T> createBufferView(const std::vector<T> &data)
    {
        return core::BufferView<const T>(data.data(), data.size());
    }

    template <typename T>
    static core::BufferView<const T> createBufferView(const T *data, size_t size)
    {
        return core::BufferView<const T>(data, size);
    }
};

} // namespace nerve::persistence::accelerated
