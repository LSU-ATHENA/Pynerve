// PersistenceGradient basic implementation split into detail includes.

#include "nerve/persistence/core/ph_gradient_basic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nerve::persistence::gradient
{

#include "detail/persistence_gradient_algorithm_helpers.inl"
#include "detail/persistence_gradient_api_ops.inl"
#include "detail/persistence_gradient_core_ops.inl"
#include "detail/persistence_gradient_matrix_helpers.inl"

} // namespace nerve::persistence::gradient
