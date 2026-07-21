#include <cstddef>
#include <cstdint>
#include <vector>

#include "nerve/core.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 16) return 0;

    const size_t n = (size % 32) + 2;
    const size_t dim = (data[0] % 5) + 1;
    const size_t total = n * dim;

    std::vector<double> points(total);
    for (size_t i = 0; i < total; ++i)
    {
        points[i] = static_cast<double>(data[i % size]) / 255.0;
    }

    try
    {
        nerve::core::BufferView<const double> view(points.data(), points.size());
    }
    catch (...)
    {
    }

    return 0;
}
