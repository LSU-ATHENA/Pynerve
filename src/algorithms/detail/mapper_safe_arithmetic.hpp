#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nerve::algorithms::detail
{

inline bool checked_product(size_t lhs, size_t rhs, size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline size_t checked_product_or_throw(size_t lhs, size_t rhs, std::string_view name)
{
    size_t out = 0;
    if (!checked_product(lhs, rhs, out))
    {
        throw std::invalid_argument(std::string(name) + " size overflows size_t");
    }
    return out;
}

inline bool has_flat_span(size_t values_size, size_t rows, size_t dim) noexcept
{
    size_t required = 0;
    return dim != 0 && checked_product(rows, dim, required) && values_size >= required;
}

template <typename T>
inline bool has_flat_span(std::span<const T> values, size_t rows, size_t dim) noexcept
{
    return has_flat_span(values.size(), rows, dim);
}

inline bool fits_int(size_t value) noexcept
{
    return value <= static_cast<size_t>(std::numeric_limits<int>::max());
}

template <typename T>
inline bool is_finite_value(T value) noexcept
{
    return std::isfinite(static_cast<long double>(value));
}

template <typename T>
inline void require_finite_prefix(std::span<const T> values, size_t count, std::string_view name)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (!is_finite_value(values[i]))
        {
            throw std::invalid_argument(std::string(name) + " contains non-finite value");
        }
    }
}

} // namespace nerve::algorithms::detail
