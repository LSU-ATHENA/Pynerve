
#include "nerve/encoders/encoders.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

double deterministicCnnWeight(Size index, Size fan_in)
{
    const std::uint64_t hash = (static_cast<std::uint64_t>(index) + 1ULL) * 7046029254386353131ULL;
    const double unit =
        static_cast<double>((hash >> 40U) & 0xFFFFFFU) / static_cast<double>(0xFFFFFFU);
    return (unit - 0.5) / std::sqrt(static_cast<double>(std::max<Size>(fan_in, 1)));
}

double deterministicUnit(Size index)
{
    const std::uint64_t hash = (static_cast<std::uint64_t>(index) + 1ULL) * 1609587929392839161ULL;
    return static_cast<double>((hash >> 40U) & 0xFFFFFFU) / static_cast<double>(0xFFFFFFU);
}

Size checkedMul(Size lhs, Size rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

Size checkedProduct(std::initializer_list<Size> factors, const char *context)
{
    Size total = 1;
    for (const Size factor : factors)
    {
        total = checkedMul(total, factor, context);
    }
    return total;
}

Size checkedVectorCount(std::initializer_list<Size> factors, const char *context)
{
    const Size count = checkedProduct(factors, context);
    if (count > std::vector<double>().max_size())
    {
        throw std::length_error(context);
    }
    return count;
}

bool isFiniteSafe(double value)
{
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max())) / 4.0L;
    const long double wide = static_cast<long double>(value);
    return std::isfinite(wide) && std::abs(wide) <= safe_abs;
}

void validateFiniteSafe(const std::vector<double> &values, const char *context)
{
    if (!std::all_of(values.begin(), values.end(), isFiniteSafe))
    {
        throw std::invalid_argument(context);
    }
}

} // namespace

CNNEncoder::CNNEncoder(Size input_channels, Size output_channels,
                       const std::vector<Size> &kernel_sizes)
{
    if (input_channels == 0 || output_channels == 0 || kernel_sizes.empty())
    {
        throw std::invalid_argument("CNN encoder dimensions and kernel list must be non-empty");
    }
    input_size_ = input_channels;
    output_size_ = output_channels;
    addConvLayer(input_channels, output_channels, kernel_sizes[0]);
    for (Size i = 1; i < kernel_sizes.size(); ++i)
    {
        addConvLayer(output_channels, output_channels, kernel_sizes[i]);
        addActivationLayer("relu");
    }
}
Tensor CNNEncoder::encode(const std::vector<std::vector<double>> &data) const
{
    Tensor input = preprocessData(data);
    return forward(input);
}
Tensor CNNEncoder::encode(const SimplicialComplex &complex) const
{
    Tensor input = preprocessComplex(complex);
    return forward(input);
}
Tensor CNNEncoder::encode(const Diagram &diagram) const
{
    Tensor input = preprocessDiagram(diagram);
    return forward(input);
}
std::vector<Tensor>
CNNEncoder::encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_data.size());
    for (const auto &data : batch_data)
    {
        batch_features.push_back(encode(data));
    }
    return batch_features;
}
std::vector<Tensor>
CNNEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        batch_features.push_back(encode(complex));
    }
    return batch_features;
}
std::vector<Tensor> CNNEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        batch_features.push_back(encode(diagram));
    }
    return batch_features;
}
void CNNEncoder::setInputSize(Size input_size)
{
    if (input_size == 0)
    {
        throw std::invalid_argument("CNN input size must be non-zero");
    }
    input_size_ = input_size;
}
void CNNEncoder::setOutputSize(Size output_size)
{
    if (output_size == 0)
    {
        throw std::invalid_argument("CNN output size must be non-zero");
    }
    output_size_ = output_size;
}
void CNNEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
}
void CNNEncoder::addConvLayer(Size in_channels, Size out_channels, Size kernel_size)
{
    if (in_channels == 0 || out_channels == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("convolution dimensions must be non-zero");
    }
    ConvLayer layer;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel_size;
    const Size kernel_area = checkedMul(kernel_size, kernel_size, "CNN kernel area overflow");
    const Size fan_in = checkedMul(in_channels, kernel_area, "CNN fan-in overflow");
    const Size channel_count = checkedMul(in_channels, out_channels, "CNN channel count overflow");
    const Size weight_count = checkedMul(channel_count, kernel_area, "CNN weight count overflow");
    std::vector<double> weightData(weight_count);
    for (Size i = 0; i < weightData.size(); ++i)
    {
        weightData[i] = deterministicCnnWeight(i, fan_in);
    }
    layer.weights = Tensor(weightData, {in_channels, out_channels, kernel_size, kernel_size});
    std::vector<double> biasData(out_channels, 0.0);
    layer.bias = Tensor(biasData, {out_channels});
    conv_layers_.push_back(layer);
}
void CNNEncoder::addPoolingLayer(Size pool_size, const std::string &pool_type)
{
    if (pool_size == 0)
    {
        throw std::invalid_argument("pool_size must be non-zero");
    }
    if (pool_type != "max" && pool_type != "avg")
    {
        throw std::invalid_argument("pool_type must be max or avg");
    }
    PoolingLayer layer;
    layer.pool_size = pool_size;
    layer.pool_type = pool_type;
    pooling_layers_.push_back(layer);
}
void CNNEncoder::addActivationLayer(const std::string &activation)
{
    if (activation != "relu" && activation != "sigmoid" && activation != "tanh" &&
        activation != "linear")
    {
        throw std::invalid_argument("unsupported CNN activation");
    }
    ActivationLayer layer;
    layer.activation_type = activation;
    activation_layers_.push_back(layer);
}
void CNNEncoder::addDropoutLayer(double dropout_rate)
{
    if (dropout_rate < 0.0 || dropout_rate >= 1.0 || !std::isfinite(dropout_rate))
    {
        throw std::invalid_argument("dropout_rate must be finite and in [0, 1)");
    }
    DropoutLayer layer;
    layer.dropout_rate = dropout_rate;
    dropout_layers_.push_back(layer);
}
Tensor CNNEncoder::forward(const Tensor &input) const
{
    Tensor current = input;
    for (Size i = 0; i < conv_layers_.size(); ++i)
    {
        current = applyConvolution(current, conv_layers_[i]);
        if (i < activation_layers_.size())
        {
            current = applyActivation(current, activation_layers_[i]);
        }
        if (i < pooling_layers_.size())
        {
            current = applyPooling(current, pooling_layers_[i]);
        }
        if (i < dropout_layers_.size())
        {
            current = applyDropout(current, dropout_layers_[i]);
        }
    }
    return current;
}
Size CNNEncoder::getInputSize() const
{
    return input_size_;
}
Size CNNEncoder::getOutputSize() const
{
    return output_size_;
}
std::string CNNEncoder::getEncoderType() const
{
    return "CNN";
}
Tensor CNNEncoder::applyConvolution(const Tensor &input, const ConvLayer &layer) const
{
    auto input_shape = input.shape();
    auto weights_shape = layer.weights.shape();
    if (input_shape.size() != 4 || weights_shape.size() != 4)
    {
        throw std::invalid_argument("CNN convolution expects rank-4 input and weights");
    }
    Size batch_size = input_shape[0];
    Size in_channels = input_shape[1];
    Size height = input_shape[2];
    Size width = input_shape[3];
    Size out_channels = weights_shape[1];
    Size kernel_h = weights_shape[2];
    Size kernel_w = weights_shape[3];
    if (weights_shape[0] != in_channels || in_channels != layer.in_channels ||
        out_channels != layer.out_channels || kernel_h != layer.kernel_size ||
        kernel_w != layer.kernel_size || layer.bias.data().size() != out_channels)
    {
        throw std::invalid_argument("CNN convolution tensor dimensions are inconsistent");
    }
    if (height < kernel_h || width < kernel_w)
    {
        throw std::invalid_argument("input image is smaller than convolution kernel");
    }
    validateFiniteSafe(input.data(), "CNN convolution input contains a non-finite or unsafe value");
    validateFiniteSafe(layer.weights.data(),
                       "CNN convolution weights contain a non-finite or unsafe value");
    validateFiniteSafe(layer.bias.data(), "CNN convolution bias contains a non-finite value");
    Size out_height = height - kernel_h + 1;
    Size out_width = width - kernel_w + 1;
    const Size output_count = checkedVectorCount({batch_size, out_channels, out_height, out_width},
                                                 "CNN convolution output size overflow");
    std::vector<double> outputData(output_count, 0.0);
    for (Size b = 0; b < batch_size; ++b)
    {
        for (Size oc = 0; oc < out_channels; ++oc)
        {
            for (Size oh = 0; oh < out_height; ++oh)
            {
                for (Size ow = 0; ow < out_width; ++ow)
                {
                    double sum = 0.0;
                    for (Size ic = 0; ic < in_channels; ++ic)
                    {
                        for (Size kh = 0; kh < kernel_h; ++kh)
                        {
                            {
                                Size input_base =
                                    ((b * in_channels + ic) * height + oh + kh) * width + ow;
                                Size weight_base =
                                    ((ic * out_channels + oc) * kernel_h + kh) * kernel_w;
                                const double partial = nerve::simd::simd_dot(
                                    &input.data()[input_base], &layer.weights.data()[weight_base],
                                    kernel_w);
                                if (!std::isfinite(partial))
                                {
                                    throw std::overflow_error(
                                        "CNN convolution produced a non-finite value");
                                }
                                sum += partial;
                            }
                        }
                    }
                    Size output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    const double value = sum + layer.bias.data()[oc];
                    if (!std::isfinite(value))
                    {
                        throw std::overflow_error("CNN convolution produced a non-finite value");
                    }
                    outputData[output_idx] = value;
                }
            }
        }
    }
    return Tensor(outputData, {batch_size, out_channels, out_height, out_width});
}
Tensor CNNEncoder::applyPooling(const Tensor &input, const PoolingLayer &layer) const
{
    auto input_shape = input.shape();
    Size pool_size = layer.pool_size;
    if (input_shape.size() != 4 || pool_size == 0)
    {
        throw std::invalid_argument("CNN pooling expects rank-4 input and non-zero pool size");
    }
    Size batch_size = input_shape[0];
    Size channels = input_shape[1];
    Size height = input_shape[2];
    Size width = input_shape[3];
    if (pool_size > height || pool_size > width)
    {
        throw std::invalid_argument("pool_size must not exceed input dimensions");
    }
    Size out_height = height / pool_size;
    Size out_width = width / pool_size;
    validateFiniteSafe(input.data(), "CNN pooling input contains a non-finite or unsafe value");
    const Size output_count = checkedVectorCount({batch_size, channels, out_height, out_width},
                                                 "CNN pooling output size overflow");
    std::vector<double> outputData(output_count, 0.0);
    for (Size b = 0; b < batch_size; ++b)
    {
        for (Size c = 0; c < channels; ++c)
        {
            for (Size oh = 0; oh < out_height; ++oh)
            {
                for (Size ow = 0; ow < out_width; ++ow)
                {
                    // Pooling window: pool_size x pool_size contiguous rows
                    // Each row of pool_size elements is contiguous in memory (width dimension)
                    if (layer.pool_type == "max")
                    {
                        double pooled = -std::numeric_limits<double>::max();
                        for (Size ph = 0; ph < pool_size; ++ph)
                        {
                            Size h = oh * pool_size + ph;
                            Size base = ((b * channels + c) * height + h) * width + ow * pool_size;
                            double row_max =
                                nerve::simd::simd_reduce_max(&input.data()[base], pool_size);
                            if (row_max > pooled)
                                pooled = row_max;
                        }
                        Size output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        outputData[output_idx] = pooled;
                    }
                    else
                    {
                        double pooled = 0.0;
                        for (Size ph = 0; ph < pool_size; ++ph)
                        {
                            Size h = oh * pool_size + ph;
                            Size base = ((b * channels + c) * height + h) * width + ow * pool_size;
                            double row_sum =
                                nerve::simd::simd_reduce_sum(&input.data()[base], pool_size);
                            if (!std::isfinite(row_sum))
                            {
                                throw std::overflow_error(
                                    "CNN pooling produced a non-finite value");
                            }
                            pooled += row_sum;
                        }
                        Size output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        outputData[output_idx] =
                            pooled / static_cast<double>(pool_size * pool_size);
                    }
                }
            }
        }
    }
    return Tensor(outputData, {batch_size, channels, out_height, out_width});
}
Tensor CNNEncoder::applyActivation(const Tensor &input, const ActivationLayer &layer) const
{
    if (layer.activation_type == "relu")
    {
        return input.relu();
    }
    else if (layer.activation_type == "sigmoid")
    {
        return input.sigmoid();
    }
    else if (layer.activation_type == "tanh")
    {
        return input.tanh();
    }
    else if (layer.activation_type == "linear")
    {
        return input;
    }
    return input;
}
Tensor CNNEncoder::applyDropout(const Tensor &input, const DropoutLayer &layer) const
{
    if (layer.dropout_rate == 0.0)
    {
        return input;
    }
    const double keep_probability = 1.0 - layer.dropout_rate;
    std::vector<double> output = input.data();
    for (Size i = 0; i < output.size(); ++i)
    {
        output[i] = deterministicUnit(i) < keep_probability ? output[i] / keep_probability : 0.0;
    }
    return Tensor(output, input.shape());
}
Tensor CNNEncoder::preprocessData(const std::vector<std::vector<double>> &data) const
{
    Size batch_size = data.size();
    Size features = 0;
    for (const auto &row : data)
    {
        features = std::max(features, row.size());
    }
    Size size =
        static_cast<Size>(std::ceil(std::sqrt(static_cast<double>(std::max<Size>(features, 1)))));
    const Size padded_features = checkedMul(size, size, "CNN padded feature size overflow");
    const Size tensor_count =
        checkedVectorCount({batch_size, padded_features}, "CNN tensor size overflow");
    std::vector<double> tensorData(tensor_count, 0.0);
    for (Size b = 0; b < batch_size; ++b)
    {
        for (Size f = 0; f < padded_features && f < data[b].size(); ++f)
        {
            if (!std::isfinite(data[b][f]))
            {
                throw std::invalid_argument("CNN encoder input contains a non-finite value");
            }
            tensorData[b * padded_features + f] = data[b][f];
        }
    }
    return Tensor(tensorData, {batch_size, 1, size, size});
}
Tensor CNNEncoder::preprocessComplex(const SimplicialComplex &complex) const
{
    auto simplices = complex.getSimplices();
    std::vector<std::vector<double>> data;
    for (const auto &simplex : simplices)
    {
        std::vector<double> features;
        features.push_back(static_cast<double>(simplex.dimension()));
        features.push_back(static_cast<double>(simplex.numVertices()));
        data.push_back(features);
    }
    return preprocessData(data);
}
Tensor CNNEncoder::preprocessDiagram(const Diagram &diagram) const
{
    const auto &pairs = diagram.getPairs();
    std::vector<std::vector<double>> data;
    for (const auto &pair : pairs)
    {
        std::vector<double> features;
        features.push_back(pair.birth);
        features.push_back(pair.death);
        features.push_back(pair.lifetime());
        data.push_back(features);
    }
    return preprocessData(data);
}

} // namespace nerve::encoders
