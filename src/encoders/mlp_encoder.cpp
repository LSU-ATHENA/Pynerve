
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

double deterministicMlpWeight(Size index, Size fan_in)
{
    const std::uint64_t hash = (static_cast<std::uint64_t>(index) + 1ULL) * 11400714819323198485ULL;
    const double unit =
        static_cast<double>((hash >> 40U) & 0xFFFFFFU) / static_cast<double>(0xFFFFFFU);
    return (unit - 0.5) / std::sqrt(static_cast<double>(std::max<Size>(fan_in, 1)));
}

double deterministicUnit(Size index)
{
    const std::uint64_t hash = (static_cast<std::uint64_t>(index) + 1ULL) * 14029467366897019727ULL;
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

MLPEncoder::MLPEncoder(Size input_size, Size hidden_size, Size output_size, Size num_layers)
{
    if (input_size == 0 || hidden_size == 0 || output_size == 0 || num_layers == 0)
    {
        throw std::invalid_argument("MLP encoder dimensions and layer count must be non-zero");
    }
    input_size_ = input_size;
    output_size_ = output_size;
    Size current_input = input_size;
    for (Size i = 0; i < num_layers - 1; ++i)
    {
        addLayer(current_input, hidden_size, "relu");
        current_input = hidden_size;
    }
    addLayer(current_input, output_size, "linear");
}
Tensor MLPEncoder::encode(const std::vector<std::vector<double>> &data) const
{
    Tensor input = preprocessData(data);
    return forward(input);
}
Tensor MLPEncoder::encode(const SimplicialComplex &complex) const
{
    Tensor input = preprocessComplex(complex);
    return forward(input);
}
Tensor MLPEncoder::encode(const Diagram &diagram) const
{
    Tensor input = preprocessDiagram(diagram);
    return forward(input);
}
std::vector<Tensor>
MLPEncoder::encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const
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
MLPEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        batch_features.push_back(encode(complex));
    }
    return batch_features;
}
std::vector<Tensor> MLPEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        batch_features.push_back(encode(diagram));
    }
    return batch_features;
}
void MLPEncoder::setInputSize(Size input_size)
{
    if (input_size == 0)
    {
        throw std::invalid_argument("MLP input size must be non-zero");
    }
    input_size_ = input_size;
}
void MLPEncoder::setOutputSize(Size output_size)
{
    if (output_size == 0)
    {
        throw std::invalid_argument("MLP output size must be non-zero");
    }
    output_size_ = output_size;
}
void MLPEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
}
void MLPEncoder::addLayer(Size input_dim, Size output_dim, const std::string &activation)
{
    if (input_dim == 0 || output_dim == 0)
    {
        throw std::invalid_argument("MLP layer dimensions must be non-zero");
    }
    if (activation != "relu" && activation != "sigmoid" && activation != "tanh" &&
        activation != "linear")
    {
        throw std::invalid_argument("unsupported MLP activation");
    }
    LinearLayer layer;
    layer.input_dim = input_dim;
    layer.output_dim = output_dim;
    layer.activation = activation;
    std::vector<double> weightData(checkedMul(input_dim, output_dim, "MLP weight count overflow"));
    for (Size i = 0; i < weightData.size(); ++i)
    {
        weightData[i] = deterministicMlpWeight(i, input_dim);
    }
    layer.weights = Tensor(weightData, {input_dim, output_dim});
    std::vector<double> biasData(output_dim, 0.0);
    layer.bias = Tensor(biasData, {output_dim});
    linear_layers_.push_back(layer);
}
void MLPEncoder::addBatchNormLayer()
{
    if (linear_layers_.empty())
    {
        throw std::invalid_argument("batch normalization requires an existing MLP layer");
    }
    BatchNormLayer layer;
    layer.momentum = 0.1;
    layer.epsilon = 1e-5;
    Size num_features = linear_layers_.back().output_dim;
    std::vector<double> gammaData(num_features, 1.0);
    layer.gamma = Tensor(gammaData, {num_features});
    std::vector<double> betaData(num_features, 0.0);
    layer.beta = Tensor(betaData, {num_features});
    std::vector<double> meanData(num_features, 0.0);
    layer.running_mean = Tensor(meanData, {num_features});
    std::vector<double> varData(num_features, 1.0);
    layer.running_var = Tensor(varData, {num_features});
    batch_norm_layers_.push_back(layer);
}
void MLPEncoder::addDropoutLayer(double dropout_rate)
{
    if (dropout_rate < 0.0 || dropout_rate >= 1.0 || !std::isfinite(dropout_rate))
    {
        throw std::invalid_argument("dropout_rate must be finite and in [0, 1)");
    }
    DropoutLayer layer;
    layer.dropout_rate = dropout_rate;
    dropout_layers_.push_back(layer);
}
Tensor MLPEncoder::forward(const Tensor &input) const
{
    Tensor current = input;
    for (Size i = 0; i < linear_layers_.size(); ++i)
    {
        current = applyLinear(current, linear_layers_[i]);
        if (i < batch_norm_layers_.size())
        {
            current = applyBatchNorm(current, batch_norm_layers_[i]);
        }
        current = applyActivation(current, linear_layers_[i].activation);
        if (i < dropout_layers_.size())
        {
            current = applyDropout(current, dropout_layers_[i]);
        }
    }
    return current;
}
Size MLPEncoder::getInputSize() const
{
    return input_size_;
}
Size MLPEncoder::getOutputSize() const
{
    return output_size_;
}
std::string MLPEncoder::getEncoderType() const
{
    return "MLP";
}
Tensor MLPEncoder::applyLinear(const Tensor &input, const LinearLayer &layer) const
{
    auto input_shape = input.shape();
    auto weights_shape = layer.weights.shape();
    if (input_shape.size() != 2 || weights_shape.size() != 2)
    {
        throw std::invalid_argument("MLP linear layer expects rank-2 input and weights");
    }
    Size batch_size = input_shape[0];
    Size input_dim = input_shape[1];
    Size output_dim = weights_shape[1];
    if (weights_shape[0] != input_dim || input_dim != layer.input_dim ||
        output_dim != layer.output_dim || layer.bias.data().size() != output_dim)
    {
        throw std::invalid_argument("MLP linear tensor dimensions are inconsistent");
    }
    validateFiniteSafe(input.data(), "MLP linear input contains a non-finite or unsafe value");
    validateFiniteSafe(layer.weights.data(),
                       "MLP linear weights contain a non-finite or unsafe value");
    validateFiniteSafe(layer.bias.data(), "MLP linear bias contains a non-finite value");
    const Size output_count =
        checkedVectorCount({batch_size, output_dim}, "MLP linear output size overflow");
    std::vector<double> outputData(output_count);
    for (Size b = 0; b < batch_size; ++b)
    {
        double *out_row = &outputData[b * output_dim];
        const double *in_row = &input.data()[b * input_dim];

        // y = A * x  (A is output_dim x input_dim, row-major)
        nerve::simd::simd_gemv(1.0, layer.weights.data().data(), in_row, 0.0, out_row, output_dim,
                               input_dim);
        // y += bias
        nerve::simd::simd_add(out_row, layer.bias.data().data(), output_dim);

        // Check finiteness of output
        for (Size o = 0; o < output_dim; ++o)
        {
            if (!std::isfinite(out_row[o]))
            {
                throw std::overflow_error("MLP linear layer produced a non-finite value");
            }
        }
    }
    return Tensor(outputData, {batch_size, output_dim});
}
Tensor MLPEncoder::applyBatchNorm(const Tensor &input, const BatchNormLayer &layer) const
{
    auto input_shape = input.shape();
    if (input_shape.size() != 2)
    {
        throw std::invalid_argument("MLP batch normalization expects rank-2 input");
    }
    Size batch_size = input_shape[0];
    Size num_features = input_shape[1];
    if (layer.gamma.data().size() != num_features || layer.beta.data().size() != num_features ||
        layer.running_mean.data().size() != num_features ||
        layer.running_var.data().size() != num_features || !std::isfinite(layer.epsilon) ||
        layer.epsilon <= 0.0)
    {
        throw std::invalid_argument("MLP batch normalization parameters are inconsistent");
    }
    validateFiniteSafe(input.data(),
                       "MLP batch normalization input contains a non-finite or unsafe value");
    validateFiniteSafe(layer.gamma.data(), "MLP batch normalization gamma is non-finite");
    validateFiniteSafe(layer.beta.data(), "MLP batch normalization beta is non-finite");
    validateFiniteSafe(layer.running_mean.data(),
                       "MLP batch normalization mean contains a non-finite value");
    validateFiniteSafe(layer.running_var.data(),
                       "MLP batch normalization variance contains a non-finite value");
    const Size output_count =
        checkedVectorCount({batch_size, num_features}, "MLP batch normalization size overflow");
    std::vector<double> outputData(output_count);
    for (Size b = 0; b < batch_size; ++b)
    {
        for (Size f = 0; f < num_features; ++f)
        {
            Size idx = b * num_features + f;
            const double variance = layer.running_var.data()[f] + layer.epsilon;
            if (!std::isfinite(variance) || variance <= 0.0)
            {
                throw std::invalid_argument("MLP batch normalization variance must be positive");
            }
            double normalized =
                (input.data()[idx] - layer.running_mean.data()[f]) / std::sqrt(variance);
            const double value = layer.gamma.data()[f] * normalized + layer.beta.data()[f];
            if (!std::isfinite(normalized) || !std::isfinite(value))
            {
                throw std::overflow_error("MLP batch normalization produced a non-finite value");
            }
            outputData[idx] = value;
        }
    }
    return Tensor(outputData, {batch_size, num_features});
}
Tensor MLPEncoder::applyDropout(const Tensor &input, const DropoutLayer &layer) const
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
Tensor MLPEncoder::applyActivation(const Tensor &input, const std::string &activation) const
{
    if (activation == "relu")
    {
        return input.relu();
    }
    else if (activation == "sigmoid")
    {
        return input.sigmoid();
    }
    else if (activation == "tanh")
    {
        return input.tanh();
    }
    else if (activation == "linear")
    {
        return input;
    }
    return input;
}
Tensor MLPEncoder::preprocessData(const std::vector<std::vector<double>> &data) const
{
    Size batch_size = data.size();
    Size features = input_size_;
    if (features == 0)
    {
        for (const auto &row : data)
        {
            features = std::max(features, row.size());
        }
    }
    const Size tensor_count =
        checkedVectorCount({batch_size, features}, "MLP tensor size overflow");
    std::vector<double> tensorData(tensor_count);
    for (Size b = 0; b < batch_size; ++b)
    {
        for (Size f = 0; f < features; ++f)
        {
            if (f < data[b].size())
            {
                if (!std::isfinite(data[b][f]))
                {
                    throw std::invalid_argument("MLP encoder input contains a non-finite value");
                }
                tensorData[b * features + f] = data[b][f];
            }
        }
    }
    return Tensor(tensorData, {batch_size, features});
}
Tensor MLPEncoder::preprocessComplex(const SimplicialComplex &complex) const
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
Tensor MLPEncoder::preprocessDiagram(const Diagram &diagram) const
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
