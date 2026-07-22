#include "nerve/core.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 8)
        return 0;

    const size_t num_points = (size % 64) + 1;
    const size_t dim = 3;

    std::vector<double> points;
    points.reserve(num_points * dim);
    for (size_t i = 0; i < num_points * dim && i < size; ++i)
    {
        points.push_back(static_cast<double>(data[i % size]) / 255.0);
    }

    try
    {
        nerve::core::BufferView<const double> view(points.data(), points.size());
    }
    catch (...)
    {}

    return 0;
}
