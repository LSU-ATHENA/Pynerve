#include "nerve/math/constants.hpp"
#include "nerve/nn/diagram_conv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace
{
constexpr float kXavierScale = 2.0f;

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
DiagramConv1D<T>::DiagramConv1D(Config config)
    : config_(config)
{
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0)
    {
        throw_invalid_argument("DiagramConv1D::DiagramConv1D",
                               "channels and kernel_size must be positive");
    }
    const size_t kernel_size = checked_vector_size<T>(
        {static_cast<size_t>(config_.out_channels), static_cast<size_t>(config_.in_channels), 2,
         static_cast<size_t>(config_.kernel_size)},
        "DiagramConv1D::DiagramConv1D", "kernel size overflows");
    weights_.kernel.resize(kernel_size);
    weights_.bias.resize(config_.out_channels);

    const double fan_in =
        static_cast<double>(config_.in_channels) * 2.0 * static_cast<double>(config_.kernel_size);
    T scale = std::sqrt(static_cast<T>(static_cast<double>(kXavierScale) / fan_in));
    const uint32_t seed = static_cast<uint32_t>(config_.in_channels) * 73856093U ^
                          static_cast<uint32_t>(config_.out_channels) * 19349663U ^
                          static_cast<uint32_t>(config_.kernel_size) * 83492791U;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> init(-1.0, 1.0);
    for (auto &w : weights_.kernel)
    {
        w = static_cast<T>(init(rng)) * scale;
    }
    for (auto &b : weights_.bias)
    {
        b = nerve::math::Constants<T>::kZero;
    }
}

template <typename T>
std::vector<T> DiagramConv1D<T>::forward(std::span<const T> diagram, std::span<const T> features,
                                         size_t batch_size, size_t n_pairs) const
{
    validate_diagram_layout(diagram, batch_size, n_pairs, "DiagramConv1D::forward");
    const size_t expected_features =
        checked_product({batch_size, n_pairs, static_cast<size_t>(config_.in_channels)},
                        "DiagramConv1D::forward", "feature shape size overflows");
    if (features.size() != expected_features)
    {
        throw_invalid_argument("DiagramConv1D::forward", "feature size mismatch; expected " +
                                                             std::to_string(expected_features) +
                                                             ", got " +
                                                             std::to_string(features.size()));
    }
    if (n_pairs > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw_length_error("DiagramConv1D::forward", "n_pairs exceeds index range");
    }
    validate_numeric_span(features, "DiagramConv1D::forward", "features");

    auto output = conv1d(features, batch_size, n_pairs);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (int oc = 0; oc < config_.out_channels; ++oc)
        {
            for (size_t i = 0; i < n_pairs; ++i)
            {
                size_t idx = ((b * config_.out_channels + oc) * n_pairs + i);
                output[idx] += weights_.bias[oc];
            }
        }
    }

    if (config_.use_persistence_weighting)
    {
        output = apply_persistence_gate(output, diagram, batch_size, n_pairs);
    }

    return output;
}

template <typename T>
void DiagramConv1D<T>::set_weights(std::span<const T> kernel, std::span<const T> bias)
{
    if (kernel.size() != weights_.kernel.size())
    {
        throw_invalid_argument("DiagramConv1D::set_weights",
                               "kernel size mismatch; expected " +
                                   std::to_string(weights_.kernel.size()) + ", got " +
                                   std::to_string(kernel.size()));
    }
    if (bias.size() != weights_.bias.size())
    {
        throw_invalid_argument("DiagramConv1D::set_weights",
                               "bias size mismatch; expected " +
                                   std::to_string(weights_.bias.size()) + ", got " +
                                   std::to_string(bias.size()));
    }
    validate_numeric_span(kernel, "DiagramConv1D::set_weights", "kernel");
    validate_numeric_span(bias, "DiagramConv1D::set_weights", "bias");
    std::copy(kernel.begin(), kernel.end(), weights_.kernel.begin());
    std::copy(bias.begin(), bias.end(), weights_.bias.begin());
}

template <typename T>
size_t DiagramConv1D<T>::output_size(size_t n_pairs) const
{
    return n_pairs;
}

template <typename T>
LandscapeLayer<T>::LandscapeLayer(Config config)
    : config_(config)
{
    if (config_.n_layers <= 0 || config_.resolution <= 0)
    {
        throw_invalid_argument("LandscapeLayer::LandscapeLayer",
                               "n_layers and resolution must be positive");
    }
}

template <typename T>
std::vector<T> LandscapeLayer<T>::forward(std::span<const T> diagram, size_t batch_size,
                                          size_t n_pairs) const
{
    validate_diagram_layout(diagram, batch_size, n_pairs, "LandscapeLayer::forward");

    const size_t landscape_size =
        checked_vector_size<T>({batch_size, static_cast<size_t>(config_.n_layers),
                                static_cast<size_t>(config_.resolution)},
                               "LandscapeLayer::forward", "landscape size overflows");
    std::vector<T> landscape(landscape_size, nerve::math::Constants<T>::kZero);

    for (size_t b = 0; b < batch_size; ++b)
    {
        std::vector<T> persistences;
        for (size_t i = 0; i < n_pairs; ++i)
        {
            T birth = diagram[(b * n_pairs + i) * 3 + 0];
            T death = diagram[(b * n_pairs + i) * 3 + 1];
            T pers = death - birth;
            if (pers > config_.min_persistence)
            {
                persistences.push_back(pers);
            }
        }

        std::sort(persistences.begin(), persistences.end(), std::greater<T>());

        const size_t layer_count =
            std::min(static_cast<size_t>(config_.n_layers), persistences.size());
        for (size_t layer = 0; layer < layer_count; ++layer)
        {
            T max_pers = persistences[layer];

            for (int r = 0; r < config_.resolution; ++r)
            {
                T x = max_pers * r / config_.resolution;
                T mid = max_pers * nerve::math::Constants<T>::kHalf;
                T val = max_pers - std::abs(x - mid);
                val = std::max(val, nerve::math::Constants<T>::kZero);

                size_t idx = (b * static_cast<size_t>(config_.n_layers) + layer) *
                                 static_cast<size_t>(config_.resolution) +
                             static_cast<size_t>(r);
                landscape[idx] = val;
            }
        }
    }

    return landscape;
}

template <typename T>
DiagramVectorizer<T>::DiagramVectorizer(Config config)
    : config_(config)
{
    if (config_.output_dim <= 0)
    {
        throw_invalid_argument("DiagramVectorizer::DiagramVectorizer",
                               "output_dim must be positive");
    }
}

template <typename T>
std::vector<T> DiagramVectorizer<T>::forward(std::span<const T> diagram, size_t batch_size,
                                             size_t n_pairs) const
{
    validate_diagram_layout(diagram, batch_size, n_pairs, "DiagramVectorizer::forward");
    if (config_.output_dim <= 0)
    {
        throw_invalid_argument("DiagramVectorizer::forward", "output_dim must be positive");
    }

    switch (config_.method)
    {
        case Config::Method::PERSISTENCE_STATS:
            return persistence_stats(diagram, batch_size, n_pairs);
        case Config::Method::BETTI_CURVE:
            return betti_curve(diagram, batch_size, n_pairs);
        case Config::Method::ENTROPY:
        case Config::Method::LANDSCAPE:
        default:
            return persistence_stats(diagram, batch_size, n_pairs);
    }
}

template <typename T>
std::vector<T> DiagramVectorizer<T>::persistence_stats(std::span<const T> diagram,
                                                       size_t batch_size, size_t n_pairs) const
{
    const size_t out_dim = static_cast<size_t>(config_.output_dim);
    const size_t feature_size = checked_vector_size<T>(
        {batch_size, out_dim}, "DiagramVectorizer::persistence_stats", "feature size overflows");
    std::vector<T> features(feature_size, nerve::math::Constants<T>::kZero);

    for (size_t b = 0; b < batch_size; ++b)
    {
        std::vector<T> persistences;
        for (size_t i = 0; i < n_pairs; ++i)
        {
            T birth = diagram[(b * n_pairs + i) * 3 + 0];
            T death = diagram[(b * n_pairs + i) * 3 + 1];
            T pers = death - birth;
            if (pers > nerve::math::Constants<T>::kZero)
            {
                persistences.push_back(pers);
            }
        }

        if (!persistences.empty())
        {
            T sum = nerve::math::Constants<T>::kZero;
            for (const T persistence : persistences)
            {
                const T next = sum + persistence;
                if (!std::isfinite(next))
                {
                    throw_invalid_argument("DiagramVectorizer::persistence_stats",
                                           "persistence sum is non-finite");
                }
                sum = next;
            }
            T mean = sum / persistences.size();

            T var_sum = nerve::math::Constants<T>::kZero;
            for (auto p : persistences)
            {
                const T delta = p - mean;
                const T contribution = delta * delta;
                const T next = var_sum + contribution;
                if (!std::isfinite(contribution) || !std::isfinite(next))
                {
                    throw_invalid_argument("DiagramVectorizer::persistence_stats",
                                           "persistence variance is non-finite");
                }
                var_sum = next;
            }
            T std_dev = std::sqrt(var_sum / persistences.size());
            if (!std::isfinite(mean) || !std::isfinite(std_dev))
            {
                throw_invalid_argument("DiagramVectorizer::persistence_stats",
                                       "persistence statistics are non-finite");
            }

            T max_pers = *std::max_element(persistences.begin(), persistences.end());

            std::vector<T> stats = {static_cast<T>(persistences.size()), // count
                                    mean, std_dev, max_pers};

            for (size_t i = 0; i < out_dim; ++i)
            {
                features[b * out_dim + i] = stats[i % stats.size()];
            }
        }
    }

    return features;
}

template <typename T>
std::vector<T> DiagramVectorizer<T>::betti_curve(std::span<const T> diagram, size_t batch_size,
                                                 size_t n_pairs) const
{
    const size_t out_dim = static_cast<size_t>(config_.output_dim);
    const int n_bins = std::max(2, config_.n_bins);
    const size_t feature_size = checked_vector_size<T>(
        {batch_size, out_dim}, "DiagramVectorizer::betti_curve", "feature size overflows");
    std::vector<T> features(feature_size, nerve::math::Constants<T>::kZero);
    if (n_pairs == 0 || config_.output_dim <= 0)
    {
        return features;
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
        T min_birth = std::numeric_limits<T>::max();
        T max_death = std::numeric_limits<T>::lowest();

        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            min_birth = std::min(min_birth, birth);
            max_death = std::max(max_death, death);
        }
        if (!(max_death > min_birth))
        {
            max_death = min_birth + nerve::math::Constants<T>::kOne;
        }

        const size_t curve_size =
            checked_vector_size<T>({static_cast<size_t>(n_bins)}, "DiagramVectorizer::betti_curve",
                                   "curve size overflows");
        std::vector<T> curve(curve_size, nerve::math::Constants<T>::kZero);
        for (int bin = 0; bin < n_bins; ++bin)
        {
            const T t = min_birth +
                        (max_death - min_birth) * static_cast<T>(bin) / static_cast<T>(n_bins - 1);
            T alive = nerve::math::Constants<T>::kZero;
            for (size_t i = 0; i < n_pairs; ++i)
            {
                const T birth = diagram[(b * n_pairs + i) * 3 + 0];
                const T death = diagram[(b * n_pairs + i) * 3 + 1];
                if (birth <= t && death > t)
                {
                    alive += nerve::math::Constants<T>::kOne;
                }
            }
            curve[static_cast<size_t>(bin)] = alive / static_cast<T>(n_pairs);
        }

        for (int out = 0; out < config_.output_dim; ++out)
        {
            const T x = static_cast<T>(out) * static_cast<T>(n_bins - 1) /
                        static_cast<T>(std::max(1, config_.output_dim - 1));
            const int left = std::clamp(static_cast<int>(std::floor(x)), 0, n_bins - 1);
            const int right = std::min(left + 1, n_bins - 1);
            const T alpha = x - static_cast<T>(left);
            const T value =
                curve[static_cast<size_t>(left)] * (nerve::math::Constants<T>::kOne - alpha) +
                curve[static_cast<size_t>(right)] * alpha;
            features[b * out_dim + static_cast<size_t>(out)] = value;
        }
    }
    return features;
}

#include "diagram_conv_kernels.inl"

template class DiagramConv1D<float>;
template class DiagramConv1D<double>;
template class LandscapeLayer<float>;
template class LandscapeLayer<double>;
template class DiagramVectorizer<float>;
template class DiagramVectorizer<double>;

} // namespace nerve::nn
