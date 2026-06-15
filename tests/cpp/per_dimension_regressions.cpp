#include "nerve/persistence/core/per_dimension_exact.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

nerve::persistence::perdim::PerDimensionConfig h0OnlyConfig()
{
    nerve::persistence::perdim::PerDimensionConfig config;
    config.max_dim = 0;
    config.max_radius = 2.0;
    config.compute_h0 = true;
    config.compute_h1 = false;
    config.compute_h2 = false;
    config.compute_h3 = false;
    config.compute_h4 = false;
    config.compute_h5 = false;
    config.compute_h6 = false;
    return config;
}

template <typename Exception, typename Func>
void assertThrows(Func &&func)
{
    bool rejected = false;
    try
    {
        func();
    }
    catch (const Exception &)
    {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main()
{
    using nerve::persistence::perdim::compute0To6DPerDimension;

    const auto config = h0OnlyConfig();
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const auto result = compute0To6DPerDimension(points, 2, 3, config);
    assert(result.config.max_dim == 0);

    const std::vector<double> nonfinite{0.0, std::numeric_limits<double>::quiet_NaN()};
    assertThrows<std::invalid_argument>(
        [&] { (void)compute0To6DPerDimension(nonfinite, 1, 2, config); });

    const std::vector<double> mismatched{0.0, 1.0, 2.0};
    assertThrows<std::invalid_argument>(
        [&] { (void)compute0To6DPerDimension(mismatched, 2, 2, config); });

    const std::vector<double> overflow{0.0, std::numeric_limits<double>::max()};
    assertThrows<std::overflow_error>(
        [&] { (void)compute0To6DPerDimension(overflow, 1, 2, config); });

    return 0;
}
