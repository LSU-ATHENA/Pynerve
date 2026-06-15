#include <cassert>
#include <cmath>
#include <limits>

#include "nerve/persistence/utils/adaptive_selector.hpp"

int main() {
    nerve::persistence::AdaptiveSelector selector;
    nerve::persistence::DataCharacteristics invalid_data{};
    invalid_data.num_simplices = std::numeric_limits<std::size_t>::max();
    invalid_data.num_vertices = std::numeric_limits<std::size_t>::max();
    invalid_data.sparsity = std::numeric_limits<double>::quiet_NaN();
    invalid_data.max_dimension = std::numeric_limits<std::size_t>::max();
    invalid_data.avg_simplex_size = std::numeric_limits<double>::infinity();
    invalid_data.memory_footprint = std::numeric_limits<std::size_t>::max();

    const auto algorithm = selector.selectOptimalAlgorithm(invalid_data);
    assert(!algorithm.empty());

    const auto diagnostics = selector.getLastSelectionDiagnostics();
    assert(std::isfinite(diagnostics.confidence));
    assert(std::isfinite(diagnostics.error_bound));
}
