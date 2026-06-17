#include "nerve/math/constants.hpp"
#include "nerve/nn/diagram_conv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
[[noreturn]] void throw_invalid_argument(const std::string &function_name,
                                         const std::string &detail)
{
    throw std::invalid_argument(function_name + ": " + detail);
}

template <typename T>
size_t checked_vector_size(std::initializer_list<size_t> factors, const std::string &function_name,
                           const std::string &detail)
{
    size_t value = 1;
    for (const size_t factor : factors)
    {
        if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor)
        {
            throw std::length_error(function_name + ": " + detail);
        }
        value *= factor;
    }
    if (value > std::vector<T>().max_size())
    {
        throw std::length_error(function_name + ": " + detail);
    }
    return value;
}
} // namespace

namespace nerve::nn
{

template <typename T>
std::vector<T> PersistenceImageLayer<T>::gaussian_kernel_2d(T sigma, int size) const
{
    if (sigma <= nerve::math::Constants<T>::kZero)
    {
        throw_invalid_argument("PersistenceImageLayer::gaussian_kernel_2d",
                               "sigma must be positive");
    }
    if (size <= 0)
    {
        throw_invalid_argument("PersistenceImageLayer::gaussian_kernel_2d",
                               "size must be positive");
    }
    if ((size % 2) == 0)
    {
        ++size;
    }

    const size_t kernel_values = checked_vector_size<T>(
        {static_cast<size_t>(size), static_cast<size_t>(size)},
        "PersistenceImageLayer::gaussian_kernel_2d", "kernel size overflows");
    std::vector<T> kernel(kernel_values, nerve::math::Constants<T>::kZero);
    const int center = size / 2;
    T sum = nerve::math::Constants<T>::kZero;
    const T two_sigma_sq = nerve::math::Constants<T>::kTwo * sigma * sigma;

    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            const T dy = static_cast<T>(y - center);
            const T dx = static_cast<T>(x - center);
            const T value = std::exp(-(dx * dx + dy * dy) / two_sigma_sq);
            if (!std::isfinite(value))
            {
                throw_invalid_argument("PersistenceImageLayer::gaussian_kernel_2d",
                                       "kernel contains non-finite values");
            }
            kernel[static_cast<size_t>(y) * static_cast<size_t>(size) + static_cast<size_t>(x)] =
                value;
            sum += value;
        }
    }

    if (sum > nerve::math::Constants<T>::kZero)
    {
        for (auto &value : kernel)
        {
            value /= sum;
        }
    }
    return kernel;
}

template <typename T>
T PersistenceImageLayer<T>::compute_weight(T persistence) const
{
    switch (config_.weight)
    {
        case Config::Weight::LINEAR:
            return persistence;
        case Config::Weight::QUADRATIC:
        {
            const T weight = persistence * persistence;
            if (!std::isfinite(weight))
            {
                throw_invalid_argument("PersistenceImageLayer::compute_weight",
                                       "weight is non-finite");
            }
            return weight;
        }
        case Config::Weight::CONSTANT:
        default:
            return nerve::math::Constants<T>::kOne;
    }
}

template std::vector<float> PersistenceImageLayer<float>::gaussian_kernel_2d(float, int) const;
template std::vector<double> PersistenceImageLayer<double>::gaussian_kernel_2d(double, int) const;
template float PersistenceImageLayer<float>::compute_weight(float) const;
template double PersistenceImageLayer<double>::compute_weight(double) const;

namespace
{
void __force_link_diagram_conv_image_ops()
{
    volatile auto w = &PersistenceImageLayer<double>::compute_weight;
    (void)w;
}
} // namespace

} // namespace nerve::nn
