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

[[noreturn]] void throw_length_error(const std::string &function_name, const std::string &detail)
{
    throw std::length_error(function_name + ": " + detail);
}

size_t checked_product(std::initializer_list<size_t> factors, const std::string &function_name,
                       const std::string &detail)
{
    size_t value = 1;
    for (const size_t factor : factors)
    {
        if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor)
        {
            throw_length_error(function_name, detail);
        }
        value *= factor;
    }
    return value;
}

template <typename T>
size_t checked_vector_size(std::initializer_list<size_t> factors, const std::string &function_name,
                           const std::string &detail)
{
    const size_t value = checked_product(factors, function_name, detail);
    if (value > std::vector<T>().max_size())
    {
        throw_length_error(function_name, detail);
    }
    return value;
}

template <typename T>
bool is_safe_finite(T value)
{
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<T>::max())) / 4.0L;
    const long double wide = static_cast<long double>(value);
    return std::isfinite(wide) && std::abs(wide) <= safe_abs;
}

template <typename T>
void validate_numeric_span(std::span<const T> values, const std::string &function_name,
                           const std::string &label)
{
    if (!std::all_of(values.begin(), values.end(), is_safe_finite<T>))
    {
        throw_invalid_argument(function_name, label + " contains non-finite or unsafe values");
    }
}

template <typename T>
void validate_diagram_layout(std::span<const T> diagram, size_t batch_size, size_t n_pairs,
                             const std::string &function_name)
{
    const size_t expected =
        checked_product({batch_size, n_pairs, 3}, function_name, "diagram shape size overflows");
    if (diagram.size() != expected)
    {
        throw_invalid_argument(function_name, "diagram size mismatch; expected " +
                                                  std::to_string(expected) + ", got " +
                                                  std::to_string(diagram.size()));
    }
    validate_numeric_span(diagram, function_name, "diagram");
}
} // namespace

namespace nerve::nn
{

template <typename T>
std::vector<T> DiagramConv1D<T>::conv1d(std::span<const T> input, size_t batch_size,
                                        size_t n_pairs) const
{
    const size_t output_size =
        checked_vector_size<T>({batch_size, static_cast<size_t>(config_.out_channels), n_pairs},
                               "DiagramConv1D::conv1d", "output size overflows");
    std::vector<T> output(output_size, nerve::math::Constants<T>::kZero);

    int pad = config_.kernel_size / 2;

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (int oc = 0; oc < config_.out_channels; ++oc)
        {
            for (size_t i = 0; i < n_pairs; ++i)
            {
                T sum = nerve::math::Constants<T>::kZero;

                for (int ic = 0; ic < config_.in_channels; ++ic)
                {
                    for (int k = 0; k < config_.kernel_size; ++k)
                    {
                        int in_idx = static_cast<int>(i) + k - pad;

                        if (in_idx >= 0 && in_idx < static_cast<int>(n_pairs))
                        {
                            size_t input_idx = (b * n_pairs + static_cast<size_t>(in_idx)) *
                                                   static_cast<size_t>(config_.in_channels) +
                                               static_cast<size_t>(ic);
                            T val = input[input_idx];

                            size_t kernel_idx = ((static_cast<size_t>(oc) *
                                                      static_cast<size_t>(config_.in_channels) * 2 +
                                                  static_cast<size_t>(ic) * 2) *
                                                     static_cast<size_t>(config_.kernel_size) +
                                                 static_cast<size_t>(k));
                            const T contribution = val * weights_.kernel[kernel_idx];
                            const T next = sum + contribution;
                            if (!std::isfinite(contribution) || !std::isfinite(next))
                            {
                                throw_invalid_argument("DiagramConv1D::conv1d",
                                                       "convolution produced non-finite values");
                            }
                            sum = next;
                        }
                    }
                }

                size_t output_idx = ((b * config_.out_channels + oc) * n_pairs + i);
                output[output_idx] = sum;
            }
        }
    }

    return output;
}

template <typename T>
std::vector<T> DiagramConv1D<T>::apply_persistence_gate(std::span<const T> output,
                                                        std::span<const T> diagram,
                                                        size_t batch_size, size_t n_pairs) const
{
    std::vector<T> gated(output.begin(), output.end());

    for (size_t b = 0; b < batch_size; ++b)
    {
        T mean = nerve::math::Constants<T>::kZero;
        T second_moment = nerve::math::Constants<T>::kZero;
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            const T persistence = death - birth;
            const T moment = persistence * persistence;
            const T next_mean = mean + persistence;
            const T next_second_moment = second_moment + moment;
            if (!std::isfinite(moment) || !std::isfinite(next_mean) ||
                !std::isfinite(next_second_moment))
            {
                throw_invalid_argument("DiagramConv1D::apply_persistence_gate",
                                       "persistence statistics are non-finite");
            }
            mean = next_mean;
            second_moment = next_second_moment;
        }

        if (n_pairs > 0)
        {
            mean /= static_cast<T>(n_pairs);
            second_moment /= static_cast<T>(n_pairs);
        }
        const T variance = std::max(nerve::math::Constants<T>::kZero, second_moment - mean * mean);
        const T inv_std =
            nerve::math::Constants<T>::kOne / std::sqrt(variance + static_cast<T>(1e-6));
        if (!std::isfinite(variance) || !std::isfinite(inv_std))
        {
            throw_invalid_argument("DiagramConv1D::apply_persistence_gate",
                                   "persistence gate is non-finite");
        }

        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            const T persistence = death - birth;
            const T standardized = (persistence - mean) * inv_std;
            const T gate = nerve::math::Constants<T>::kOne /
                           (nerve::math::Constants<T>::kOne + std::exp(-standardized));
            if (!std::isfinite(standardized) || !std::isfinite(gate))
            {
                throw_invalid_argument("DiagramConv1D::apply_persistence_gate",
                                       "persistence gate is non-finite");
            }

            for (int oc = 0; oc < config_.out_channels; ++oc)
            {
                size_t idx = ((b * config_.out_channels + oc) * n_pairs + i);
                const T value = gated[idx] * gate;
                if (!std::isfinite(value))
                {
                    throw_invalid_argument("DiagramConv1D::apply_persistence_gate",
                                           "gated output is non-finite");
                }
                gated[idx] = value;
            }
        }
    }

    return gated;
}

template std::vector<float> DiagramConv1D<float>::conv1d(std::span<const float>, size_t,
                                                         size_t) const;
template std::vector<double> DiagramConv1D<double>::conv1d(std::span<const double>, size_t,
                                                           size_t) const;
template std::vector<float> DiagramConv1D<float>::apply_persistence_gate(std::span<const float>,
                                                                         std::span<const float>,
                                                                         size_t, size_t) const;
template std::vector<double> DiagramConv1D<double>::apply_persistence_gate(std::span<const double>,
                                                                           std::span<const double>,
                                                                           size_t, size_t) const;

} // namespace nerve::nn
