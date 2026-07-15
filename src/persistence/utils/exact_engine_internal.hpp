#pragma once

#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"

#include <utility>
#include <vector>

namespace nerve::persistence
{
namespace detail
{

using Column = std::vector<Size>;

Column symmetricDifferenceSorted(const Column &a, const Column &b);

bool simplexFiltrationOrder(const std::pair<algebra::Simplex, double> &a,
                            const std::pair<algebra::Simplex, double> &b);

} // namespace detail

} // namespace nerve::persistence
