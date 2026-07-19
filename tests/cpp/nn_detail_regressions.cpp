#include "nerve/core_types.hpp"
#include "nerve/nn/diagram_conv.hpp"
#include "nerve/nn/simd_nn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace
{

constexpr double kTol = 1e-10;

template <typename T>
std::span<const T> as_span(const std::vector<T> &v)
{
    return std::span<const T>(v.data(), v.size());
}

template <typename T>
bool all_finite(const std::vector<T> &v)
{
    return std::all_of(v.begin(), v.end(), [](T x) { return std::isfinite(x); });
}

bool check_conv1d_forward()
{
    nerve::nn::DiagramConv1D<double>::Config cfg;
    cfg.in_channels = 1;
    cfg.out_channels = 1;
    cfg.kernel_size = 3;
    cfg.use_persistence_weighting = false;
    nerve::nn::DiagramConv1D<double> conv(cfg);
    std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 1.0};
    std::vector<double> features{0.5, 0.3};
    auto out = conv.forward(as_span(diagram), as_span(features), 1, 2);
    if (!all_finite(out))
    {
        std::cerr << "conv1d output contains non-finite values\n";
        return false;
    }
    if (out.empty())
    {
        std::cerr << "conv1d output empty\n";
        return false;
    }
    return true;
}

bool check_conv1d_output_size()
{
    nerve::nn::DiagramConv1D<double>::Config cfg;
    cfg.in_channels = 2;
    cfg.out_channels = 3;
    cfg.kernel_size = 3;
    cfg.stride = 1;
    nerve::nn::DiagramConv1D<double> conv(cfg);
    size_t osize = conv.output_size(10);
    if (osize == 0)
    {
        std::cerr << "conv1d output size should be non-zero\n";
        return false;
    }
    return true;
}

bool check_conv1d_set_weights()
{
    nerve::nn::DiagramConv1D<double>::Config cfg;
    cfg.in_channels = 1;
    cfg.out_channels = 1;
    cfg.kernel_size = 3;
    nerve::nn::DiagramConv1D<double> conv(cfg);
    std::vector<double> kernel(6, 0.5);
    std::vector<double> bias(1, 0.1);
    conv.set_weights(as_span(kernel), as_span(bias));
    std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 1.0};
    std::vector<double> features{0.5, 0.3};
    auto out = conv.forward(as_span(diagram), as_span(features), 1, 2);
    if (!all_finite(out))
    {
        std::cerr << "conv1d with set_weights output non-finite\n";
        return false;
    }
    return true;
}

bool check_conv2d_forward()
{
    nerve::nn::DiagramConv2D<double>::Config cfg;
    cfg.in_channels = 1;
    cfg.out_channels = 1;
    cfg.kernel_h = 2;
    cfg.kernel_w = 2;
    cfg.stride_h = 1;
    cfg.stride_w = 1;
    nerve::nn::DiagramConv2D<double> conv(cfg);
    std::vector<double> input{1.0, 0.0, 0.5, 0.2, 0.4, 0.6, 0.3, 0.7, 0.9};
    auto out = conv.forward(as_span(input), 1, 3, 3);
    if (!all_finite(out))
    {
        std::cerr << "conv2d output non-finite\n";
        return false;
    }
    if (out.empty())
    {
        std::cerr << "conv2d output empty\n";
        return false;
    }
    return true;
}

bool check_conv2d_output_dimensions()
{
    nerve::nn::DiagramConv2D<double>::Config cfg;
    cfg.in_channels = 1;
    cfg.out_channels = 1;
    cfg.kernel_h = 2;
    cfg.kernel_w = 2;
    cfg.stride_h = 1;
    cfg.stride_w = 1;
    nerve::nn::DiagramConv2D<double> conv(cfg);
    std::vector<double> input(16, 1.0);
    auto out = conv.forward(as_span(input), 1, 4, 4);
    size_t expected = ((4 - 2) / 1 + 1) * ((4 - 2) / 1 + 1);
    if (out.size() != expected)
    {
        std::cerr << "conv2d expected " << expected << " output, got " << out.size() << "\n";
        return false;
    }
    return true;
}

bool check_persistence_image_forward()
{
    nerve::nn::PersistenceImageLayer<double>::Config cfg;
    cfg.resolution_h = 5;
    cfg.resolution_w = 5;
    cfg.sigma = 0.1;
    cfg.weight = nerve::nn::PersistenceImageLayer<double>::Config::Weight::LINEAR;
    nerve::nn::PersistenceImageLayer<double> layer(cfg);
    std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 1.0};
    auto image = layer.forward(as_span(diagram), 1, 2);
    if (image.size() != 25)
    {
        std::cerr << "persistence image wrong size: " << image.size() << "\n";
        return false;
    }
    if (!all_finite(image))
    {
        std::cerr << "persistence image non-finite\n";
        return false;
    }
    return true;
}

bool check_persistence_image_multi_dim()
{
    nerve::nn::PersistenceImageLayer<double>::Config cfg;
    cfg.resolution_h = 4;
    cfg.resolution_w = 4;
    cfg.sigma = 0.1;
    nerve::nn::PersistenceImageLayer<double> layer(cfg);
    std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 0.0, 0.5, 2.0, 1.0};
    auto images = layer.forward_multi_dim(as_span(diagram), 1, 3, 1);
    if (images.size() != 32)
    {
        std::cerr << "multi dim image wrong size\n";
        return false;
    }
    return true;
}

bool check_simd_relu()
{
    std::vector<double> data{-1.0, 0.0, 0.5, -2.0, 3.0};
    std::vector<double> expected{0.0, 0.0, 0.5, 0.0, 3.0};
    nerve::nn::simdReLU(data.data(), data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (std::abs(data[i] - expected[i]) > kTol)
        {
            std::cerr << "simdReLU mismatch at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_simd_sigmoid()
{
    std::vector<double> data{0.0, 1.0, -1.0, 2.0};
    nerve::nn::simdSigmoid(data.data(), data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i] <= 0.0 || data[i] >= 1.0)
        {
            std::cerr << "simdSigmoid out of range at " << i << ": " << data[i] << "\n";
            return false;
        }
    }
    return true;
}

bool check_simd_tanh()
{
    std::vector<double> data{0.0, 1.0, -1.0, 2.0};
    nerve::nn::simdTanh(data.data(), data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i] < -1.0 || data[i] > 1.0)
        {
            std::cerr << "simdTanh out of range at " << i << ": " << data[i] << "\n";
            return false;
        }
    }
    return true;
}

bool check_simd_batch_norm()
{
    std::vector<double> data{1.0, 2.0, 3.0, 4.0};
    nerve::nn::simdBatchNorm(data.data(), data.size(), 2.5, 1.0);
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (!std::isfinite(data[i]))
        {
            std::cerr << "simdBatchNorm non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_simd_softmax()
{
    std::vector<double> data{1.0, 2.0, 3.0, 4.0};
    nerve::nn::simdSoftmax(data.data(), data.size());
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i] <= 0.0 || data[i] > 1.0)
        {
            std::cerr << "simdSoftmax out of range at " << i << ": " << data[i] << "\n";
            return false;
        }
        sum += data[i];
    }
    if (std::abs(sum - 1.0) > kTol)
    {
        std::cerr << "simdSoftmax sum=" << sum << " != 1\n";
        return false;
    }
    return true;
}

bool check_landscape_layer()
{
    nerve::nn::LandscapeLayer<double>::Config cfg;
    cfg.n_layers = 3;
    cfg.resolution = 10;
    cfg.min_persistence = 0.0;
    nerve::nn::LandscapeLayer<double> layer(cfg);
    std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 0.0};
    auto landscape = layer.forward(as_span(diagram), 1, 2);
    if (landscape.size() != 30)
    {
        std::cerr << "landscape wrong size: " << landscape.size() << "\n";
        return false;
    }
    if (!all_finite(landscape))
    {
        std::cerr << "landscape non-finite\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_conv1d_forward())
    {
        std::cerr << "FAIL: conv1d forward\n";
        return 1;
    }
    if (!check_conv1d_output_size())
    {
        std::cerr << "FAIL: conv1d output size\n";
        return 1;
    }
    if (!check_conv1d_set_weights())
    {
        std::cerr << "FAIL: conv1d set weights\n";
        return 1;
    }
    if (!check_conv2d_forward())
    {
        std::cerr << "FAIL: conv2d forward\n";
        return 1;
    }
    if (!check_conv2d_output_dimensions())
    {
        std::cerr << "FAIL: conv2d dims\n";
        return 1;
    }
    if (!check_persistence_image_forward())
    {
        std::cerr << "FAIL: persistence image\n";
        return 1;
    }
    if (!check_persistence_image_multi_dim())
    {
        std::cerr << "FAIL: persistence image multi\n";
        return 1;
    }
    if (!check_simd_relu())
    {
        std::cerr << "FAIL: simd relu\n";
        return 1;
    }
    if (!check_simd_sigmoid())
    {
        std::cerr << "FAIL: simd sigmoid\n";
        return 1;
    }
    if (!check_simd_tanh())
    {
        std::cerr << "FAIL: simd tanh\n";
        return 1;
    }
    if (!check_simd_batch_norm())
    {
        std::cerr << "FAIL: simd batch norm\n";
        return 1;
    }
    if (!check_simd_softmax())
    {
        std::cerr << "FAIL: simd softmax\n";
        return 1;
    }
    if (!check_landscape_layer())
    {
        std::cerr << "FAIL: landscape layer\n";
        return 1;
    }
    return 0;
}
