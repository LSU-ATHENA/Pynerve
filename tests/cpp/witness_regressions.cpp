#include "nerve/persistence/vr/vr_landmark_ops.hpp"
#include "nerve/persistence/vr/vr_lazy_witness_ops.hpp"

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
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
    const auto landmarks = nerve::persistence::LandmarkSelector::selectLandmarks(
        points, 2, 2, 1, nerve::persistence::LandmarkSelector::Strategy::MAXMIN);
    assert(landmarks.size() == 1);

    const std::vector<double> trailing_value{0.0, 0.0, 1.0, 0.0, 9.0};
    assertThrows<std::invalid_argument>([&] {
        (void)nerve::persistence::LandmarkSelector::selectLandmarks(
            trailing_value, 2, 2, 1, nerve::persistence::LandmarkSelector::Strategy::MAXMIN);
    });

    const std::vector<double> nonfinite{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    assertThrows<std::invalid_argument>([&] {
        (void)nerve::persistence::LandmarkSelector::selectLandmarks(
            nonfinite, 2, 2, 1, nerve::persistence::LandmarkSelector::Strategy::MAXMIN);
    });

    const std::vector<double> safe_max{0.0, 0.0, 1.0, 0.0};
    const auto safe_landmarks = nerve::persistence::LandmarkSelector::selectLandmarks(
        safe_max, 2, 2, 1, nerve::persistence::LandmarkSelector::Strategy::MAXMIN);
    assert(safe_landmarks.size() == 1);

    nerve::algebra::SimplicialComplex complex;
    const std::vector<size_t> lazy_landmarks{0, 1};
    nerve::persistence::LazyWitnessComplex witness(points, 2, lazy_landmarks, 1, 2.0);
    witness.buildComplex(complex);
    assert(complex.simplicesOfDimension(0).size() == 2);
    assert(complex.simplicesOfDimension(1).size() == 1);

    nerve::algebra::SimplicialComplex valid_complex;
    const std::vector<double> test_points{0.0, 0.0, 1.0, 0.0};
    nerve::persistence::LazyWitnessComplex valid_lazy(test_points, 2, lazy_landmarks, 1, 2.0);
    valid_lazy.buildComplex(valid_complex);
    assert(!valid_complex.simplicesOfDimension(0).empty());

    return 0;
}
