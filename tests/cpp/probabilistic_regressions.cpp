#include "nerve/probabilistic/probabilistic.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename Exception, typename Func>
void assertThrows(Func&& func) {
    bool rejected = false;
    try {
        func();
    } catch (const Exception&) {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main() {
    nerve::probabilistic::MonteCarloPersistence monte_carlo;
    const std::vector<std::vector<double>> points{{0.0}, {1.0}};
    const auto samples = monte_carlo.monteCarloSample(points, 1, "uniform");
    assert(samples.size() == 1);

    const std::vector<std::vector<double>> nonfinite{
        {0.0}, {std::numeric_limits<double>::quiet_NaN()}};
    assertThrows<std::invalid_argument>([&] {
        (void)monte_carlo.monteCarloSample(nonfinite, 1, "uniform");
    });

    const std::vector<std::vector<double>> overflow_prone{
        {-std::numeric_limits<double>::max()}, {std::numeric_limits<double>::max()}};
    assertThrows<std::overflow_error>([&] {
        (void)monte_carlo.monteCarloSample(overflow_prone, 1, "uniform");
    });

    nerve::persistence::Diagram diagram;
    diagram.addPair({0.0, 1.0, 0});
    diagram.addPair({0.0, std::numeric_limits<double>::infinity(), 0});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(diagram);
    model.fitGaussianModel();
    assert(model.computeProbability({0.0, 1.0, 0}) > 0.0);

    nerve::persistence::Diagram invalid_diagram;
    invalid_diagram.addPair({0.0, std::numeric_limits<double>::quiet_NaN(), 0});
    nerve::probabilistic::ProbabilisticPersistenceDiagram invalid_model(invalid_diagram);
    assertThrows<std::invalid_argument>([&] { invalid_model.fitGaussianModel(); });
    assertThrows<std::invalid_argument>([&] { invalid_model.fitMixtureModel(1); });
    assertThrows<std::invalid_argument>([&] {
        (void)model.computeProbability({0.0, std::numeric_limits<double>::quiet_NaN(), 0});
    });

    return 0;
}
