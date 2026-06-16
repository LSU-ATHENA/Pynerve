#ifndef DIAGRAM_CONV_KERNELS_INCLUDED
#define DIAGRAM_CONV_KERNELS_INCLUDED

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
    const size_t element_count = checked_product(factors, function_name, detail);
    if (element_count > std::vector<T>().max_size())
    {
        throw_length_error(function_name, detail);
    }
    return element_count;
}

template <typename T>
void validate_numeric_span(std::span<const T> values, const std::string &function_name,
                           const std::string &name)
{
    for (const T val : values)
    {
        if (!std::isfinite(val))
        {
            throw_invalid_argument(function_name,
                                   "non-finite " + name + " value detected (" + std::to_string(val) +
                                       ")");
        }
    }
}

inline void validate_diagram_layout(std::span<const float> diagram, size_t batch_size, size_t n_pairs,
                             const std::string &function_name)
{
    const size_t expected_diagram =
        checked_product({batch_size, n_pairs, 3}, function_name, "diagram shape size overflows");
    if (diagram.size() != expected_diagram)
    {
        throw_invalid_argument(function_name, "diagram size mismatch; expected " +
                                                  std::to_string(expected_diagram) + ", got " +
                                                  std::to_string(diagram.size()));
    }
}

void validate_diagram_layout(std::span<const double> diagram, size_t batch_size, size_t n_pairs,
                             const std::string &function_name)
{
    const size_t expected_diagram =
        checked_product({batch_size, n_pairs, 3}, function_name, "diagram shape size overflows");
    if (diagram.size() != expected_diagram)
    {
        throw_invalid_argument(function_name, "diagram size mismatch; expected " +
                                                  std::to_string(expected_diagram) + ", got " +
                                                  std::to_string(diagram.size()));
    }
}

} // namespace

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
                        const int input_idx = static_cast<int>(i) + k - pad;
                        if (input_idx >= 0 && input_idx < static_cast<int>(n_pairs))
                        {
                            const size_t input_offset =
                                (b * config_.in_channels + ic) * n_pairs +
                                static_cast<size_t>(input_idx);
                            const size_t kernel_offset =
                                (oc * config_.in_channels + ic) * config_.kernel_size +
                                static_cast<size_t>(k);
                            sum += input[input_offset] * weights_.kernel[kernel_offset];
                        }
                    }
                }
                const size_t out_idx = (b * config_.out_channels + oc) * n_pairs + i;
                output[out_idx] = sum;
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> DiagramConv1D<T>::apply_persistence_gate(std::span<const T> features,
                                                        std::span<const T> diagram,
                                                        size_t batch_size, size_t n_pairs) const
{
    std::vector<T> result(features.begin(), features.end());
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            const T persistence = death - birth;
            if (persistence <= nerve::math::Constants<T>::kZero)
            {
                continue;
            }
            const T gate = std::log1p(persistence);
            for (int oc = 0; oc < config_.out_channels; ++oc)
            {
                const size_t idx = (b * config_.out_channels + oc) * n_pairs + i;
                result[idx] *= gate;
            }
        }
    }
    return result;
}

#endif // DIAGRAM_CONV_KERNELS_INCLUDED
