#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

std::unique_ptr<FeatureEncoder> createHybridMember(const std::string &type)
{
    if (type == "CNN" || type == "cnn")
    {
        return EncoderFactory::createCnnEncoder(1, 16, {3});
    }
    if (type == "MLP" || type == "mlp")
    {
        return EncoderFactory::createMlpEncoder(8, 32, 16, 2);
    }
    if (type == "Topological" || type == "topological")
    {
        return EncoderFactory::createTopologicalEncoder(16);
    }
    if (type == "Persistence" || type == "persistence")
    {
        return EncoderFactory::createPersistenceEncoder(16);
    }
    if (type == "Graph" || type == "graph")
    {
        return EncoderFactory::createGraphEncoder(8, 4, 16);
    }
    throw std::invalid_argument("unsupported hybrid encoder member type");
}

std::vector<double> fitHybridFeatures(std::vector<double> features, Size output_size)
{
    if (output_size == 0)
    {
        return features;
    }
    if (features.size() > output_size)
    {
        features.resize(output_size);
    }
    features.resize(output_size, 0.0);
    return features;
}

} // namespace

HybridEncoder::HybridEncoder(const std::vector<std::string> &encoder_types)
    : fusion_method_("concat")
{
    input_size_ = 0;
    output_size_ = 0;
    for (const auto &type : encoder_types)
    {
        addEncoder(createHybridMember(type));
    }
    encoder_weights_.assign(encoders_.size(),
                            encoders_.empty() ? 0.0 : 1.0 / static_cast<double>(encoders_.size()));
}

Tensor HybridEncoder::encode(const std::vector<std::vector<double>> &data) const
{
    const auto features = encodeWithAllEncoders(data);
    if (fusion_method_ == "weighted_average")
    {
        return weightedAverageFeatures(features);
    }
    if (fusion_method_ == "attention")
    {
        return attentionFusion(features);
    }
    return concatenateFeatures(features);
}

Tensor HybridEncoder::encode(const SimplicialComplex &complex) const
{
    const auto features = encodeComplexWithAllEncoders(complex);
    if (fusion_method_ == "weighted_average")
    {
        return weightedAverageFeatures(features);
    }
    if (fusion_method_ == "attention")
    {
        return attentionFusion(features);
    }
    return concatenateFeatures(features);
}

Tensor HybridEncoder::encode(const Diagram &diagram) const
{
    const auto features = encodeDiagramWithAllEncoders(diagram);
    if (fusion_method_ == "weighted_average")
    {
        return weightedAverageFeatures(features);
    }
    if (fusion_method_ == "attention")
    {
        return attentionFusion(features);
    }
    return concatenateFeatures(features);
}

std::vector<Tensor>
HybridEncoder::encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const
{
    std::vector<Tensor> output;
    output.reserve(batch_data.size());
    for (const auto &data : batch_data)
    {
        output.push_back(encode(data));
    }
    return output;
}

std::vector<Tensor>
HybridEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> output;
    output.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        output.push_back(encode(complex));
    }
    return output;
}

std::vector<Tensor> HybridEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> output;
    output.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        output.push_back(encode(diagram));
    }
    return output;
}

void HybridEncoder::setInputSize(Size input_size)
{
    input_size_ = input_size;
    for (auto &encoder : encoders_)
    {
        encoder->setInputSize(input_size);
    }
}

void HybridEncoder::setOutputSize(Size output_size)
{
    output_size_ = output_size;
}

void HybridEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
    for (auto &encoder : encoders_)
    {
        encoder->setParameters(params);
    }
}

void HybridEncoder::addEncoder(std::unique_ptr<FeatureEncoder> encoder)
{
    if (!encoder)
    {
        throw std::invalid_argument("hybrid encoder member must not be null");
    }
    output_size_ += encoder->getOutputSize();
    encoders_.push_back(std::move(encoder));
    encoder_weights_.assign(encoders_.size(),
                            encoders_.empty() ? 0.0 : 1.0 / static_cast<double>(encoders_.size()));
}

void HybridEncoder::setFusionMethod(const std::string &fusion_method)
{
    if (fusion_method != "concat" && fusion_method != "weighted_average" &&
        fusion_method != "attention")
    {
        throw std::invalid_argument("unsupported hybrid fusion method");
    }
    fusion_method_ = fusion_method;
}

void HybridEncoder::setEncoderWeights(const std::vector<double> &weights)
{
    if (weights.size() != encoders_.size())
    {
        throw std::invalid_argument("hybrid encoder weight count must match encoder count");
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0.0 || !std::isfinite(total))
    {
        throw std::invalid_argument("hybrid encoder weights must have positive finite sum");
    }
    encoder_weights_.resize(weights.size());
    for (Size i = 0; i < weights.size(); ++i)
    {
        if (weights[i] < 0.0 || !std::isfinite(weights[i]))
        {
            throw std::invalid_argument("hybrid encoder weights must be finite and non-negative");
        }
        encoder_weights_[i] = weights[i] / total;
    }
}

Tensor HybridEncoder::concatenateFeatures(const std::vector<Tensor> &features) const
{
    std::vector<double> fused;
    for (const auto &tensor : features)
    {
        fused.insert(fused.end(), tensor.data().begin(), tensor.data().end());
    }
    const Size shape_size = output_size_ == 0 ? fused.size() : output_size_;
    return Tensor(fitHybridFeatures(std::move(fused), output_size_), {shape_size});
}

Tensor HybridEncoder::weightedAverageFeatures(const std::vector<Tensor> &features) const
{
    Size width = 0;
    for (const auto &tensor : features)
    {
        width = std::max(width, tensor.size());
    }
    std::vector<double> fused(width, 0.0);
    for (Size i = 0; i < features.size(); ++i)
    {
        const double weight = i < encoder_weights_.size() ? encoder_weights_[i] : 0.0;
        for (Size j = 0; j < features[i].size(); ++j)
        {
            fused[j] += weight * features[i].data()[j];
        }
    }
    const Size shape_size = output_size_ == 0 ? fused.size() : output_size_;
    return Tensor(fitHybridFeatures(std::move(fused), output_size_), {shape_size});
}

Tensor HybridEncoder::attentionFusion(const std::vector<Tensor> &features) const
{
    std::vector<double> scores(features.size(), 0.0);
    for (Size i = 0; i < features.size(); ++i)
    {
        for (double value : features[i].data())
        {
            scores[i] += std::abs(value);
        }
    }
    const double total = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (total <= 0.0)
    {
        return weightedAverageFeatures(features);
    }

    std::vector<double> weights(scores.size(), 0.0);
    for (Size i = 0; i < scores.size(); ++i)
    {
        weights[i] = scores[i] / total;
    }
    HybridEncoder weighted({});
    for (const auto &tensor : features)
    {
        weighted.output_size_ = std::max(weighted.output_size_, tensor.size());
    }
    weighted.encoder_weights_ = std::move(weights);
    return weighted.weightedAverageFeatures(features);
}

Size HybridEncoder::getInputSize() const
{
    return input_size_;
}

Size HybridEncoder::getOutputSize() const
{
    return output_size_;
}

std::string HybridEncoder::getEncoderType() const
{
    return "Hybrid";
}

std::vector<Tensor>
HybridEncoder::encodeWithAllEncoders(const std::vector<std::vector<double>> &data) const
{
    std::vector<Tensor> output;
    output.reserve(encoders_.size());
    for (const auto &encoder : encoders_)
    {
        output.push_back(encoder->encode(data));
    }
    return output;
}

std::vector<Tensor>
HybridEncoder::encodeComplexWithAllEncoders(const SimplicialComplex &complex) const
{
    std::vector<Tensor> output;
    output.reserve(encoders_.size());
    for (const auto &encoder : encoders_)
    {
        output.push_back(encoder->encode(complex));
    }
    return output;
}

std::vector<Tensor> HybridEncoder::encodeDiagramWithAllEncoders(const Diagram &diagram) const
{
    std::vector<Tensor> output;
    output.reserve(encoders_.size());
    for (const auto &encoder : encoders_)
    {
        output.push_back(encoder->encode(diagram));
    }
    return output;
}

} // namespace nerve::encoders
