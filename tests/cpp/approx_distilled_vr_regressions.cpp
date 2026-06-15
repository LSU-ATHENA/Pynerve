#include "nerve/persistence/approximate/distilled_vr_filtration.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

nerve::persistence::distilled::DistilledVRConfig config()
{
    nerve::persistence::distilled::DistilledVRConfig cfg;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    cfg.use_bit_parallel = false;
    return cfg;
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
    using nerve::persistence::distilled::buildDistilledFiltration;

    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto filtration = buildDistilledFiltration(points, 2, 4, config());
    assert(!filtration.simplices.empty());
    assert(filtration.distilled_size == static_cast<int>(filtration.simplices.size()));

    const std::vector<double> trailing_value{0.0, 0.0, 1.0, 0.0, 9.0};
    assertThrows<std::invalid_argument>(
        [&] { (void)buildDistilledFiltration(trailing_value, 2, 2, config()); });

    const std::vector<double> nonfinite{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    assertThrows<std::invalid_argument>(
        [&] { (void)buildDistilledFiltration(nonfinite, 2, 2, config()); });

    const std::vector<double> overflow_prone{0.0, 0.0, std::numeric_limits<double>::max(), 0.0};
    assertThrows<std::invalid_argument>(
        [&] { (void)buildDistilledFiltration(overflow_prone, 2, 2, config()); });

    return 0;
}
