#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nerve::encoders
{
namespace
{

bool isFiniteSafe(double value)
{
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max())) / 4.0L;
    const long double wide = static_cast<long double>(value);
    return std::isfinite(wide) && std::abs(wide) <= safe_abs;
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
    if (total > std::vector<double>().max_size())
    {
        throw std::length_error(context);
    }
    return total;
}

void validateRows(const std::vector<std::vector<double>> &data)
{
    if (data.empty())
        return;
    const Size dim = data.front().size();
    if (dim == 0)
    {
        throw std::invalid_argument("graph encoder rows must contain coordinates");
    }
    for (const auto &row : data)
    {
        if (row.size() != dim)
        {
            throw std::invalid_argument("graph encoder rows must have consistent dimensions");
        }
        if (!std::ranges::all_of(row, isFiniteSafe))
        {
            throw std::invalid_argument("graph encoder rows must contain finite safe values");
        }
    }
}

void validateDiagramPair(const persistence::Pair &pair)
{
    const bool death_ok =
        isFiniteSafe(pair.death) || pair.death == std::numeric_limits<double>::infinity();
    if (!isFiniteSafe(pair.birth) || !death_ok ||
        (std::isfinite(pair.death) && pair.death < pair.birth))
    {
        throw std::invalid_argument("graph encoder diagram contains an invalid pair");
    }
}

std::vector<double> fitGraphFeatures(std::vector<double> features, Size output_size)
{
    if (output_size > std::vector<double>().max_size())
    {
        throw std::length_error("graph feature output size exceeds vector capacity");
    }
    if (features.size() > output_size)
    {
        features.resize(output_size);
    }
    features.resize(output_size, 0.0);
    return features;
}

std::vector<double> summarizeAdjacency(const Tensor &adjacency)
{
    const auto &shape = adjacency.shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[0] != shape[1])
    {
        return {0.0, 0.0, 0.0, 0.0};
    }

    const Size n = shape[0];
    Size directed_edges = 0;
    double max_degree = 0.0;
    for (Size row = 0; row < n; ++row)
    {
        double degree = 0.0;
        for (Size col = 0; col < n; ++col)
        {
            if (adjacency.data()[row * n + col] != 0.0)
            {
                degree += 1.0;
            }
        }
        directed_edges += static_cast<Size>(degree);
        max_degree = std::max(max_degree, degree);
    }

    const double undirected_edges = static_cast<double>(directed_edges) * 0.5;
    const double possible = n > 1 ? static_cast<double>(n * (n - 1)) * 0.5 : 1.0;
    return {static_cast<double>(n), undirected_edges, undirected_edges / possible, max_degree};
}

Tensor tensorFromFeatures(std::vector<double> features, Size output_size)
{
    return Tensor(fitGraphFeatures(std::move(features), output_size), {output_size});
}

} // namespace

GraphEncoder::GraphEncoder(Size node_dim, Size edge_dim, Size output_dim)
    : node_dim_(node_dim)
    , edge_dim_(edge_dim)
    , graph_construction_method_("threshold")
    , distance_threshold_(1.0)
    , k_neighbors_(8)
    , global_pooling_type_("mean")
{
    if (node_dim == 0 || edge_dim == 0 || output_dim == 0)
    {
        throw std::invalid_argument("graph encoder dimensions must be non-zero");
    }
    input_size_ = node_dim;
    output_size_ = output_dim;
}

Tensor GraphEncoder::encode(const std::vector<std::vector<double>> &data) const
{
    validateRows(data);
    std::vector<double> features = summarizeAdjacency(constructGraph(data));
    Size value_count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    Size max_dim = 0;
    for (const auto &row : data)
    {
        max_dim = std::max(max_dim, row.size());
        for (double value : row)
        {
            const double next_sum = sum + value;
            const double square = value * value;
            const double next_sum_sq = sum_sq + square;
            if (!std::isfinite(next_sum) || !std::isfinite(square) || !std::isfinite(next_sum_sq))
            {
                throw std::overflow_error("graph encoder feature statistics overflow");
            }
            sum = next_sum;
            sum_sq = next_sum_sq;
            ++value_count;
        }
    }
    const double mean = value_count == 0 ? 0.0 : sum / static_cast<double>(value_count);
    const double variance =
        value_count == 0 ? 0.0 : sum_sq / static_cast<double>(value_count) - mean * mean;
    features.push_back(static_cast<double>(max_dim));
    features.push_back(mean);
    features.push_back(std::max(0.0, variance));
    return tensorFromFeatures(std::move(features), output_size_);
}

Tensor GraphEncoder::encode(const SimplicialComplex &complex) const
{
    std::vector<double> features = summarizeAdjacency(constructGraphFromComplex(complex));
    const auto simplices = complex.getSimplices();
    Size vertices = 0;
    Size edges = 0;
    Size higher = 0;
    for (const auto &simplex : simplices)
    {
        if (simplex.dimension() == 0)
        {
            ++vertices;
        }
        else if (simplex.dimension() == 1)
        {
            ++edges;
        }
        else
        {
            ++higher;
        }
    }
    features.push_back(static_cast<double>(simplices.size()));
    features.push_back(static_cast<double>(vertices));
    features.push_back(static_cast<double>(edges));
    features.push_back(static_cast<double>(higher));
    features.push_back(static_cast<double>(std::max<Dimension>(0, complex.maxDimension())));
    return tensorFromFeatures(std::move(features), output_size_);
}

Tensor GraphEncoder::encode(const Diagram &diagram) const
{
    std::vector<double> features = summarizeAdjacency(constructGraphFromDiagram(diagram));
    double total_lifetime = 0.0;
    double max_lifetime = 0.0;
    Size finite_count = 0;
    for (const auto &pair : diagram.getPairs())
    {
        validateDiagramPair(pair);
        if (std::isfinite(pair.birth) && std::isfinite(pair.death) && pair.death > pair.birth)
        {
            const double lifetime = pair.death - pair.birth;
            const double next_total = total_lifetime + lifetime;
            if (!std::isfinite(lifetime) || !std::isfinite(next_total))
            {
                throw std::overflow_error("graph encoder lifetime statistics overflow");
            }
            total_lifetime = next_total;
            max_lifetime = std::max(max_lifetime, lifetime);
            ++finite_count;
        }
    }
    features.push_back(static_cast<double>(finite_count));
    features.push_back(total_lifetime);
    features.push_back(max_lifetime);
    return tensorFromFeatures(std::move(features), output_size_);
}

std::vector<Tensor>
GraphEncoder::encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const
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
GraphEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> output;
    output.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        output.push_back(encode(complex));
    }
    return output;
}

std::vector<Tensor> GraphEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> output;
    output.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        output.push_back(encode(diagram));
    }
    return output;
}

void GraphEncoder::setInputSize(Size input_size)
{
    if (input_size == 0)
    {
        throw std::invalid_argument("graph encoder input size must be non-zero");
    }
    input_size_ = input_size;
    node_dim_ = input_size;
}

void GraphEncoder::setOutputSize(Size output_size)
{
    if (output_size == 0)
    {
        throw std::invalid_argument("graph encoder output size must be non-zero");
    }
    output_size_ = output_size;
}

void GraphEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
    for (const auto &[key, value] : params)
    {
        if (value < 0.0 || !std::isfinite(value))
        {
            throw std::invalid_argument("graph encoder parameters must be finite and non-negative");
        }
        if (key == "distance_threshold")
        {
            distance_threshold_ = value;
        }
        else if (key == "k_neighbors")
        {
            if (std::floor(value) != value)
            {
                throw std::invalid_argument("k_neighbors must be an integer value");
            }
            k_neighbors_ = static_cast<Size>(value);
        }
    }
}

void GraphEncoder::addGcnLayer(Size hidden_dim)
{
    if (hidden_dim == 0)
    {
        throw std::invalid_argument("GCN hidden dimension must be non-zero");
    }
    GCNLayer layer;
    layer.input_dim = gcn_layers_.empty() ? node_dim_ : gcn_layers_.back().output_dim;
    layer.output_dim = hidden_dim;
    const Size weight_count =
        checkedProduct({layer.input_dim, layer.output_dim}, "GCN weight count overflow");
    std::vector<double> weights(weight_count, 0.0);
    for (Size i = 0; i < weights.size(); ++i)
    {
        weights[i] = 1.0 / static_cast<double>(1 + (i % std::max<Size>(1, layer.input_dim)));
    }
    layer.weights = Tensor(weights, {layer.input_dim, layer.output_dim});
    layer.bias = Tensor(std::vector<double>(layer.output_dim, 0.0), {layer.output_dim});
    gcn_layers_.push_back(std::move(layer));
}

void GraphEncoder::addGatLayer(Size hidden_dim, Size num_heads)
{
    if (hidden_dim == 0 || num_heads == 0)
    {
        throw std::invalid_argument("GAT hidden dimension and head count must be non-zero");
    }
    GATLayer layer;
    layer.input_dim = gat_layers_.empty() ? node_dim_ : gat_layers_.back().output_dim;
    layer.output_dim = hidden_dim;
    layer.num_heads = num_heads;
    const Size weight_count =
        checkedProduct({layer.input_dim, layer.output_dim}, "GAT weight count overflow");
    layer.weights =
        Tensor(std::vector<double>(weight_count, 1.0 / static_cast<double>(layer.num_heads)),
               {layer.input_dim, layer.output_dim});
    layer.attention_weights =
        Tensor(std::vector<double>(layer.output_dim, 1.0), {layer.output_dim});
    gat_layers_.push_back(std::move(layer));
}

void GraphEncoder::addGraphConvLayer(const std::string &conv_type, Size hidden_dim)
{
    if (conv_type == "gcn")
    {
        addGcnLayer(hidden_dim);
    }
    else if (conv_type == "gat")
    {
        addGatLayer(hidden_dim, 1);
    }
    else
    {
        throw std::invalid_argument("unsupported graph convolution type");
    }
}

void GraphEncoder::addGlobalPooling(const std::string &pooling_type)
{
    if (pooling_type != "mean" && pooling_type != "sum" && pooling_type != "max")
    {
        throw std::invalid_argument("unsupported graph pooling type");
    }
    global_pooling_type_ = pooling_type;
}

void GraphEncoder::setGraphConstructionMethod(const std::string &method)
{
    if (method != "threshold" && method != "knn")
    {
        throw std::invalid_argument("unsupported graph construction method");
    }
    graph_construction_method_ = method;
}

void GraphEncoder::setDistanceThreshold(double threshold)
{
    if (threshold < 0.0 || !std::isfinite(threshold))
    {
        throw std::invalid_argument("graph distance threshold must be finite and non-negative");
    }
    distance_threshold_ = threshold;
}

void GraphEncoder::setKNeighbors(Size k)
{
    k_neighbors_ = k;
}

Tensor GraphEncoder::forward(const Tensor &node_features, const Tensor &edge_features,
                             const Tensor &adjacency) const
{
    static_cast<void>(edge_features);
    Tensor current = node_features;
    for (const auto &layer : gcn_layers_)
    {
        current = applyGcn(current, adjacency, layer);
    }
    for (const auto &layer : gat_layers_)
    {
        current = applyGat(current, adjacency, layer);
    }
    return applyGlobalPooling(current);
}

Size GraphEncoder::getInputSize() const
{
    return input_size_;
}

Size GraphEncoder::getOutputSize() const
{
    return output_size_;
}

std::string GraphEncoder::getEncoderType() const
{
    return "Graph";
}

} // namespace nerve::encoders
