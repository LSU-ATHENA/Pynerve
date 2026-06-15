#pragma once

#include "nerve/persistence/reduction/bit_tree_pivot_column.hpp"

#include <vector>

namespace nerve::persistence
{

template <typename Index>
using DefaultPivotColumn = std::vector<Index>;

using FastPivotColumn = BitTreePivotColumn;

} // namespace nerve::persistence
