#include "nerve/streaming/windowed_ph.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

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
    using nerve::streaming::PartialRecomputeHeuristic;

    PartialRecomputeHeuristic heuristic;
    nerve::persistence::Diagram valid;
    valid.addPair({0.0, 1.0, 0});
    valid.addPair({0.0, std::numeric_limits<double>::infinity(), 0});
    const std::vector<nerve::algebra::Simplex> added{nerve::algebra::Simplex({0})};
    const auto strategy = heuristic.determineStrategy(added, {}, valid);
    (void)strategy;

    nerve::persistence::Diagram nan_death;
    nan_death.addPair({0.0, std::numeric_limits<double>::quiet_NaN(), 0});
    assertThrows<std::invalid_argument>(
        [&] { (void)heuristic.determineStrategy(added, {}, nan_death); });

    nerve::persistence::Diagram inverted;
    inverted.addPair({2.0, 1.0, 0});
    assertThrows<std::invalid_argument>(
        [&] { (void)heuristic.determineStrategy(added, {}, inverted); });

    nerve::persistence::Diagram negative_dim;
    negative_dim.addPair({0.0, 1.0, -1});
    assertThrows<std::invalid_argument>(
        [&] { (void)heuristic.determineStrategy(added, {}, negative_dim); });

    return 0;
}
