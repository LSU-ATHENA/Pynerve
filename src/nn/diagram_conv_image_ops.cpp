#include "nerve/math/constants.hpp"
#include "nerve/nn/diagram_conv.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#ifdef NERVE_HAS_CUDA
#include "nerve/gpu/persistence_image.cuh"

#include <future>
#endif

namespace
{

constexpr int kDefaultResolution = nerve::math::diagram::kDefaultResolution;
constexpr float kDefaultSigma = nerve::math::diagram::kDefaultSigma;
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
void validate_config_scalar(T value, const std::string &function_name, const std::string &label)
{
    if (!is_safe_finite(value))
    {
        throw_invalid_argument(function_name, label + " must be finite and safe");
    }
}

template <typename T>
int bounded_spread(T sigma, int resolution_h, int resolution_w, const std::string &function_name)
{
    const long double raw =
        static_cast<long double>(sigma) * static_cast<long double>(resolution_w);
    if (!std::isfinite(raw) || raw > static_cast<long double>(std::numeric_limits<int>::max()))
    {
        throw_length_error(function_name, "kernel spread exceeds index range");
    }
    return std::min(static_cast<int>(raw), std::max(resolution_h, resolution_w));
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
DiagramConv2D<T>::DiagramConv2D(Config config)
    : config_(config)
{
    if (config_.in_channels <= 0)
    {
        config_.in_channels = 1;
    }
    if (config_.out_channels <= 0)
    {
        config_.out_channels = 1;
    }
    if (config_.kernel_h <= 0)
    {
        config_.kernel_h = 3;
    }
    if (config_.kernel_w <= 0)
    {
        config_.kernel_w = 3;
    }
    if (config_.stride_h <= 0)
    {
        config_.stride_h = 1;
    }
    if (config_.stride_w <= 0)
    {
        config_.stride_w = 1;
    }

    const size_t kernel_size = checked_vector_size<T>(
        {static_cast<size_t>(config_.out_channels), static_cast<size_t>(config_.in_channels),
         static_cast<size_t>(config_.kernel_h), static_cast<size_t>(config_.kernel_w)},
        "DiagramConv2D::DiagramConv2D", "kernel size overflows");
    kernel_.resize(kernel_size);
    bias_.resize(static_cast<size_t>(config_.out_channels), nerve::math::Constants<T>::kZero);

    const double fan_in = static_cast<double>(config_.in_channels) *
                          static_cast<double>(config_.kernel_h) *
                          static_cast<double>(config_.kernel_w);
    const T scale = std::sqrt(static_cast<T>(static_cast<double>(kXavierScale) / fan_in));
    const uint32_t seed = static_cast<uint32_t>(config_.in_channels) * 73856093U ^
                          static_cast<uint32_t>(config_.out_channels) * 19349663U ^
                          static_cast<uint32_t>(config_.kernel_h) * 83492791U ^
                          static_cast<uint32_t>(config_.kernel_w) * 2654435761U;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> init(-1.0, 1.0);
    for (auto &w : kernel_)
    {
        w = static_cast<T>(init(rng)) * scale;
    }
}

template <typename T>
std::vector<T> DiagramConv2D<T>::forward(std::span<const T> input, size_t batch_size, size_t height,
                                         size_t width) const
{
    const size_t expected =
        checked_product({batch_size, static_cast<size_t>(config_.in_channels), height, width},
                        "DiagramConv2D::forward", "input shape size overflows");
    if (input.size() != expected)
    {
        throw_invalid_argument("DiagramConv2D::forward", "input size mismatch; expected " +
                                                             std::to_string(expected) + ", got " +
                                                             std::to_string(input.size()));
    }
    validate_numeric_span(input, "DiagramConv2D::forward", "input");
    if (height < static_cast<size_t>(config_.kernel_h) ||
        width < static_cast<size_t>(config_.kernel_w))
    {
        throw_invalid_argument("DiagramConv2D::forward",
                               "kernel dimensions exceed input spatial dimensions");
    }

    const size_t out_h =
        (height - static_cast<size_t>(config_.kernel_h)) / static_cast<size_t>(config_.stride_h) +
        1;
    const size_t out_w =
        (width - static_cast<size_t>(config_.kernel_w)) / static_cast<size_t>(config_.stride_w) + 1;
    const size_t output_size = checked_vector_size<T>(
        {batch_size, static_cast<size_t>(config_.out_channels), out_h, out_w},
        "DiagramConv2D::forward", "output size overflows");
    std::vector<T> output(output_size, nerve::math::Constants<T>::kZero);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (int oc = 0; oc < config_.out_channels; ++oc)
        {
            for (size_t oy = 0; oy < out_h; ++oy)
            {
                for (size_t ox = 0; ox < out_w; ++ox)
                {
                    T sum = bias_[static_cast<size_t>(oc)];
                    for (int ic = 0; ic < config_.in_channels; ++ic)
                    {
                        const size_t ic_s = static_cast<size_t>(ic);
                        for (int ky = 0; ky < config_.kernel_h; ++ky)
                        {
                            const size_t ky_s = static_cast<size_t>(ky);
                            const size_t in_y = oy * static_cast<size_t>(config_.stride_h) + ky_s;
                            // input and kernel are contiguous for consecutive kx
                            const size_t input_base =
                                (((b * static_cast<size_t>(config_.in_channels) + ic_s) *
                                      height +
                                  in_y) *
                                     width +
                                 ox * static_cast<size_t>(config_.stride_w));
                            const size_t kernel_base =
                                (((static_cast<size_t>(oc) *
                                        static_cast<size_t>(config_.in_channels) +
                                    ic_s) *
                                       static_cast<size_t>(config_.kernel_h) +
                                   ky_s) *
                                      static_cast<size_t>(config_.kernel_w));
                            const size_t kw = static_cast<size_t>(config_.kernel_w);

                            // Batched dot product via dispatch table
                            T partial;
                            if constexpr (std::is_same_v<T, double>)
                                partial = nerve::simd::simd_dot(&input[input_base],
                                                                 &kernel_[kernel_base], kw);
                            else if constexpr (std::is_same_v<T, float>)
                                partial = nerve::simd::simd_dot_f32(&input[input_base],
                                                                     &kernel_[kernel_base], kw);
                            if (!std::isfinite(partial))
                            {
                                throw_invalid_argument(
                                    "DiagramConv2D::forward",
                                    "convolution produced non-finite values");
                            }
                            sum += partial;
                        }
                    }
                    const size_t out_idx = (((b * static_cast<size_t>(config_.out_channels) +
                                              static_cast<size_t>(oc)) *
                                                 out_h +
                                             oy) *
                                                out_w +
                                            ox);
                    output[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

template <typename T>
PersistenceImageLayer<T>::PersistenceImageLayer(Config config)
    : config_(config)
{
    if (config_.resolution_h <= 0)
    {
        config_.resolution_h = kDefaultResolution;
    }
    if (config_.resolution_w <= 0)
    {
        config_.resolution_w = kDefaultResolution;
    }
    if (config_.sigma <= 0)
    {
        config_.sigma = static_cast<T>(kDefaultSigma);
    }
    validate_config_scalar(config_.sigma, "PersistenceImageLayer::PersistenceImageLayer", "sigma");
}

template <typename T>
std::vector<T> PersistenceImageLayer<T>::forward(std::span<const T> diagram, size_t batch_size,
                                                 size_t n_pairs) const
{
    validate_diagram_layout(diagram, batch_size, n_pairs, "PersistenceImageLayer::forward");

#ifdef NERVE_HAS_CUDA
    if constexpr (std::is_same_v<T, float>)
    {
        if (batch_size == 1 && n_pairs >= 2)
        {
            std::vector<float> births(n_pairs), deaths(n_pairs);
            for (size_t i = 0; i < n_pairs; ++i)
            {
                births[i] = diagram[i * 3 + 0];
                deaths[i] = diagram[i * 3 + 1];
            }
            std::promise<std::vector<std::vector<double>>> promise;
            auto future = promise.get_future();
            nerve::gpu::persistence_image::compute_persistence_image_gpu(
                births, deaths, config_.resolution_w, config_.sigma,
                [&promise](const std::vector<std::vector<double>> &result) {
                    promise.set_value(result);
                });
            auto gpu_result = future.get();
            if (!gpu_result.empty())
            {
                const size_t h = gpu_result.size();
                const size_t w = gpu_result[0].size();
                std::vector<T> image(h * w);
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        image[y * w + x] = static_cast<T>(gpu_result[y][x]);
                return image;
            }
        }
    }
#endif

    const size_t image_size =
        checked_vector_size<T>({batch_size, static_cast<size_t>(config_.resolution_h),
                                static_cast<size_t>(config_.resolution_w)},
                               "PersistenceImageLayer::forward", "image size overflows");
    std::vector<T> image(image_size, nerve::math::Constants<T>::kZero);

    T min_birth = std::numeric_limits<T>::max();
    T max_death = std::numeric_limits<T>::lowest();

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            min_birth = std::min(min_birth, birth);
            max_death = std::max(max_death, death);
        }
    }

    if (!(max_death > min_birth))
    {
        max_death = min_birth + nerve::math::Constants<T>::kOne;
    }

    const T birth_range = max_death - min_birth;
    // Pre-compute Gaussian kernel once (constant per forward() call)
    const int spread =
        bounded_spread(config_.sigma, config_.resolution_h, config_.resolution_w,
                       "PersistenceImageLayer::forward");
    const int win = 2 * spread + 1;
    const size_t kernel_count = static_cast<size_t>(win) * static_cast<size_t>(win);
    const T inv_two_sigma_sq = T(1) / (T(2) * config_.sigma * config_.sigma);
    std::vector<T> kernel_vals(kernel_count);
    for (int dy = -spread; dy <= spread; ++dy)
    {
        const T dy_t = static_cast<T>(dy);
        for (int dx = -spread; dx <= spread; ++dx)
        {
            const T dx_t = static_cast<T>(dx);
            const size_t k_idx = static_cast<size_t>((dy + spread) * win + (dx + spread));
            kernel_vals[k_idx] = -(dx_t * dx_t + dy_t * dy_t) * inv_two_sigma_sq;
        }
    }
    // Batched SIMD.exp: replaces each element with exp(element)
    if constexpr (std::is_same_v<T, double>)
        nerve::simd::simd_exp(kernel_vals.data(), kernel_count);
    else if constexpr (std::is_same_v<T, float>)
        nerve::simd::simd_exp_f32(kernel_vals.data(), kernel_count);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            const T persistence = death - birth;

            int x =
                static_cast<int>((birth - min_birth) / birth_range * (config_.resolution_w - 1));
            int y =
                static_cast<int>((death - min_birth) / birth_range * (config_.resolution_h - 1));
            x = std::clamp(x, 0, config_.resolution_w - 1);
            y = std::clamp(y, 0, config_.resolution_h - 1);

            const T weight = compute_weight(persistence);
            for (int dy = -spread; dy <= spread; ++dy)
            {
                for (int dx = -spread; dx <= spread; ++dx)
                {
                    const int ny = y + dy;
                    const int nx = x + dx;
                    if (ny < 0 || ny >= config_.resolution_h || nx < 0 ||
                        nx >= config_.resolution_w)
                    {
                        continue;
                    }
                    const size_t k_idx = static_cast<size_t>((dy + spread) * win + (dx + spread));
                    const T gauss = kernel_vals[k_idx];
                    const size_t idx = (b * config_.resolution_h + static_cast<size_t>(ny)) *
                                           config_.resolution_w +
                                       static_cast<size_t>(nx);
                    const T contribution = weight * gauss;
                    const T next = image[idx] + contribution;
                    if (!std::isfinite(contribution) || !std::isfinite(next))
                    {
                        throw_invalid_argument("PersistenceImageLayer::forward",
                                               "image contains non-finite values");
                    }
                    image[idx] = next;
                }
            }
        }
    }

    return image;
}

template <typename T>
std::vector<T> PersistenceImageLayer<T>::forward_multi_dim(std::span<const T> diagram,
                                                           size_t batch_size, size_t n_pairs,
                                                           int max_dim) const
{
    validate_diagram_layout(diagram, batch_size, n_pairs,
                            "PersistenceImageLayer::forward_multi_dim");
    if (max_dim < 0)
    {
        throw_invalid_argument("PersistenceImageLayer::forward_multi_dim",
                               "max_dim must be non-negative");
    }

    const size_t n_dims = static_cast<size_t>(max_dim) + 1;
    const size_t image_size =
        checked_vector_size<T>({batch_size, n_dims, static_cast<size_t>(config_.resolution_h),
                                static_cast<size_t>(config_.resolution_w)},
                               "PersistenceImageLayer::forward_multi_dim", "image size overflows");
    std::vector<T> image(image_size, nerve::math::Constants<T>::kZero);

    T min_birth = std::numeric_limits<T>::max();
    T max_death = std::numeric_limits<T>::lowest();
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            min_birth = std::min(min_birth, birth);
            max_death = std::max(max_death, death);
        }
    }
    if (!(max_death > min_birth))
    {
        max_death = min_birth + nerve::math::Constants<T>::kOne;
    }
    const T birth_range = max_death - min_birth;

    // Pre-compute Gaussian kernel once (constant per forward_multi_dim call)
    const int spread =
        bounded_spread(config_.sigma, config_.resolution_h, config_.resolution_w,
                       "PersistenceImageLayer::forward_multi_dim");
    const int win = 2 * spread + 1;
    const size_t kernel_count = static_cast<size_t>(win) * static_cast<size_t>(win);
    const T inv_two_sigma_sq = T(1) / (T(2) * config_.sigma * config_.sigma);
    std::vector<T> kernel_vals(kernel_count);
    for (int dy = -spread; dy <= spread; ++dy)
    {
        const T dy_t = static_cast<T>(dy);
        for (int dx = -spread; dx <= spread; ++dx)
        {
            const T dx_t = static_cast<T>(dx);
            const size_t k_idx = static_cast<size_t>((dy + spread) * win + (dx + spread));
            kernel_vals[k_idx] = -(dx_t * dx_t + dy_t * dy_t) * inv_two_sigma_sq;
        }
    }
    // Batched SIMD.exp: replaces each element with exp(element)
    if constexpr (std::is_same_v<T, double>)
        nerve::simd::simd_exp(kernel_vals.data(), kernel_count);
    else if constexpr (std::is_same_v<T, float>)
        nerve::simd::simd_exp_f32(kernel_vals.data(), kernel_count);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < n_pairs; ++i)
        {
            const T birth = diagram[(b * n_pairs + i) * 3 + 0];
            const T death = diagram[(b * n_pairs + i) * 3 + 1];
            const T persistence = death - birth;
            const int dim = static_cast<int>(std::round(diagram[(b * n_pairs + i) * 3 + 2]));
            if (dim < 0 || dim > max_dim)
            {
                continue;
            }

            int x =
                static_cast<int>((birth - min_birth) / birth_range * (config_.resolution_w - 1));
            int y =
                static_cast<int>((death - min_birth) / birth_range * (config_.resolution_h - 1));
            x = std::clamp(x, 0, config_.resolution_w - 1);
            y = std::clamp(y, 0, config_.resolution_h - 1);
            const T weight = compute_weight(persistence);

            for (int dy = -spread; dy <= spread; ++dy)
            {
                for (int dx = -spread; dx <= spread; ++dx)
                {
                    const int ny = y + dy;
                    const int nx = x + dx;
                    if (ny < 0 || ny >= config_.resolution_h || nx < 0 ||
                        nx >= config_.resolution_w)
                    {
                        continue;
                    }
                    const size_t k_idx = static_cast<size_t>((dy + spread) * win + (dx + spread));
                    const T gauss = kernel_vals[k_idx];
                    const size_t out_idx = ((((b * n_dims + static_cast<size_t>(dim)) *
                                                  static_cast<size_t>(config_.resolution_h) +
                                              static_cast<size_t>(ny)) *
                                             static_cast<size_t>(config_.resolution_w)) +
                                            static_cast<size_t>(nx));
                    const T contribution = weight * gauss;
                    const T next = image[out_idx] + contribution;
                    if (!std::isfinite(contribution) || !std::isfinite(next))
                    {
                        throw_invalid_argument("PersistenceImageLayer::forward_multi_dim",
                                               "image contains non-finite values");
                    }
                    image[out_idx] = next;
                }
            }
        }
    }

    return image;
}

template class DiagramConv2D<float>;
template class DiagramConv2D<double>;
template class PersistenceImageLayer<float>;
template class PersistenceImageLayer<double>;

void __nerve_nn_diagram_conv_image_ops_pin() {}

} // namespace nerve::nn
