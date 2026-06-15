
#pragma once

#include <span>

namespace nerve
{

// Buffer view type for streaming operations
template <typename T>
using BufferView = std::span<T>;

} // namespace nerve
