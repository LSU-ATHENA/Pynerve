#include "nerve/persistence/core/ph_gradient_basic.hpp"

#include <cassert>
#include <limits>
#include <span>
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
    using namespace nerve::persistence::gradient;

    const double points[] = {0.0, 1.0};
    const auto diagram = computeDifferentiable(std::span<const double>(points, 2), 2, 1, 2.0, 1);
    assert(!diagram.empty());

    assertThrows<std::invalid_argument>([&] {
        (void)computeDifferentiable(std::span<const double>(points, 2), 2, 1, -1.0, 1);
    });

    const double invalid_points[] = {0.0, std::numeric_limits<double>::quiet_NaN()};
    assertThrows<std::invalid_argument>([&] {
        (void)computeDifferentiable(std::span<const double>(invalid_points, 2), 2, 1, 2.0, 1);
    });

    const double overflow_points[] = {0.0, std::numeric_limits<double>::max()};
    assertThrows<std::overflow_error>([&] {
        (void)computeDifferentiable(std::span<const double>(overflow_points, 2), 2, 1, 2.0, 1);
    });

    const double scalar = 0.0;
    const size_t oversized = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    assertThrows<std::length_error>([&] {
        (void)computeDifferentiable(std::span<const double>(&scalar, oversized), oversized, 1,
                                    2.0, 1);
    });

    DifferentiableDiagram backward_diagram;
    backward_diagram.persistence_pairs.push_back({0.0, 1.0});
    backward_diagram.dimensions.push_back(0);
    backward_diagram.birth_simplices.emplace_back(1);
    backward_diagram.birth_simplices.back()[0] = 0;
    backward_diagram.death_simplices.emplace_back(1);
    backward_diagram.death_simplices.back()[0] = 1;

    const std::vector<double> grad_birth{1.0};
    const std::vector<double> grad_death{1.0};
    const auto backward =
        persistenceBackward(std::span<const double>(points, 2), 2, 1, backward_diagram, grad_birth,
                            grad_death);
    assert(backward.size() == 2);

    const std::vector<double> invalid_grad{std::numeric_limits<double>::infinity()};
    assertThrows<std::invalid_argument>([&] {
        (void)persistenceBackward(std::span<const double>(points, 2), 2, 1, backward_diagram,
                                  invalid_grad, grad_death);
    });

    DifferentiableDiagram invalid_backward_diagram = backward_diagram;
    invalid_backward_diagram.persistence_pairs[0] = {1.0, 0.0};
    assertThrows<std::invalid_argument>([&] {
        (void)persistenceBackward(std::span<const double>(points, 2), 2, 1,
                                  invalid_backward_diagram, grad_birth, grad_death);
    });

    invalid_backward_diagram.persistence_pairs[0] = {
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    assertThrows<std::invalid_argument>([&] {
        (void)persistenceBackward(std::span<const double>(points, 2), 2, 1,
                                  invalid_backward_diagram, grad_birth, grad_death);
    });

    assertThrows<std::length_error>([&] {
        (void)persistenceBackward(std::span<const double>(&scalar, oversized), oversized, 1,
                                  backward_diagram, grad_birth, grad_death);
    });

    return 0;
}
