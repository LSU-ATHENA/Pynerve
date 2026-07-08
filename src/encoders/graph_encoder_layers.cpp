#include "nerve/encoders/encoders.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

std::vector<double> fitLayerFeatures(std::vector<double> features, Size output_size)
{
    if (output_size > std::vector<double>().max_size())
    {
        throw std::length_error("graph layer output size exceeds vector capacity");
    }
    if (features.size() > output_size)
    {
        features.resize(output_size);
    }
    features.resize(output_size, 0.0);
    return features;
}

Tensor tensorFromLayerFeatures(std::vector<double> features, Size output_size)
{
    return Tensor(fitLayerFeatures(std::move(features), output_size), {output_size});
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

Tensor GraphEncoder::applyGcn(const Tensor &features, const Tensor &adjacency,
                              const GCNLayer &layer) const
{
    const auto &feature_shape = features.shape();
    const auto &adjacency_shape = adjacency.shape();
    const auto &weight_shape = layer.weights.shape();
    if (feature_shape.size() != 2 || adjacency_shape.size() != 2 || weight_shape.size() != 2 ||
        adjacency_shape[0] != adjacency_shape[1] || feature_shape[0] != adjacency_shape[0])
    {
        throw std::invalid_argument("GCN layer expects rank-2 features and square adjacency");
    }
    if (weight_shape[0] != layer.input_dim || weight_shape[1] != layer.output_dim ||
        layer.bias.data().size() != layer.output_dim)
    {
        throw std::invalid_argument("GCN layer tensor dimensions are inconsistent");
    }
    validateFiniteSafe(features.data(), "GCN features contain a non-finite or unsafe value");
    validateFiniteSafe(adjacency.data(), "GCN adjacency contains a non-finite or unsafe value");
    validateFiniteSafe(layer.weights.data(), "GCN weights contain a non-finite or unsafe value");
    validateFiniteSafe(layer.bias.data(), "GCN bias contains a non-finite value");
    if (!std::all_of(adjacency.data().begin(), adjacency.data().end(),
                     [](double value) { return value >= 0.0; }))
    {
        throw std::invalid_argument("GCN adjacency must be non-negative");
    }
    const Size n = adjacency_shape[0];
    const Size in_dim = layer.input_dim;
    const Size usable_dim = std::min(in_dim, feature_shape[1]);
    const Size output_count = checkedProduct({n, layer.output_dim}, "GCN output size overflow");
    std::vector<double> output(output_count, 0.0);
    for (Size row = 0; row < n; ++row)
    {
        double degree = 1.0 + nerve::simd::simd_reduce_sum(&adjacency.data()[row * n], n);
        if (!std::isfinite(degree))
        {
            throw std::overflow_error("GCN degree accumulation overflow");
        }
        if (degree <= 0.0)
        {
            throw std::invalid_argument("GCN degree must be positive");
        }
        for (Size nbr = 0; nbr < n; ++nbr)
        {
            const double weight = (row == nbr ? 1.0 : adjacency.data()[row * n + nbr]) / degree;
            if (weight == 0.0)
            {
                continue;
            }
            // Loop-swap: in outer, out inner - weights are contiguous for consecutive out
            double* out_row = &output[row * layer.output_dim];
            const Size out_dim = layer.output_dim;
            for (Size in = 0; in < usable_dim; ++in)
            {
                const double scale = weight * features.data()[nbr * feature_shape[1] + in];
                nerve::simd::simd_axpy(scale, &layer.weights.data()[in * out_dim], out_row, out_dim);
            }
            // Check finiteness of the partial result for this neighbor
            for (Size o = 0; o < out_dim; ++o)
            {
                if (!std::isfinite(out_row[o]))
                {
                    throw std::overflow_error("GCN layer produced a non-finite value");
                }
            }
            // Add bias/degree (preserving original per-nbr bias semantics)
            for (Size out = 0; out < out_dim; ++out)
            {
                const double bias = layer.bias.data()[out] / degree;
                const double next = out_row[out] + bias;
                if (!std::isfinite(bias) || !std::isfinite(next))
                {
                    throw std::overflow_error("GCN layer produced a non-finite value");
                }
                out_row[out] = next;
            }
        }
    }
    return Tensor(output, {n, layer.output_dim});
}

Tensor GraphEncoder::applyGat(const Tensor &features, const Tensor &adjacency,
                              const GATLayer &layer) const
{
    GCNLayer projected;
    projected.input_dim = layer.input_dim;
    projected.output_dim = layer.output_dim;
    projected.weights = layer.weights;
    projected.bias = Tensor(std::vector<double>(layer.output_dim, 0.0), {layer.output_dim});
    Tensor output = applyGcn(features, adjacency, projected);
    if (layer.output_dim == 0 || layer.attention_weights.data().size() != layer.output_dim)
    {
        throw std::invalid_argument("GAT attention dimensions are inconsistent");
    }
    validateFiniteSafe(layer.attention_weights.data(),
                       "GAT attention weights contain a non-finite value");
    for (Size i = 0; i < output.data().size(); ++i)
    {
        const double value =
            output.data()[i] * layer.attention_weights.data()[i % layer.output_dim];
        if (!std::isfinite(value))
        {
            throw std::overflow_error("GAT layer produced a non-finite value");
        }
        output.data()[i] = value;
    }
    return output;
}

Tensor GraphEncoder::applyGlobalPooling(const Tensor &features) const
{
    const auto &shape = features.shape();
    if (shape.size() != 2 || shape[0] == 0)
    {
        return Tensor(std::vector<double>(output_size_, 0.0), {output_size_});
    }
    const Size n = shape[0];
    const Size dim = shape[1];
    validateFiniteSafe(features.data(),
                       "graph pooling input contains a non-finite or unsafe value");
    std::vector<double> pooled(
        dim, global_pooling_type_ == "max" ? -std::numeric_limits<double>::infinity() : 0.0);
    if (global_pooling_type_ == "max")
    {
        for (Size row = 0; row < n; ++row)
        {
            nerve::simd::simd_max(pooled.data(), &features.data()[row * dim], dim);
        }
    }
    else
    {
        for (Size row = 0; row < n; ++row)
        {
            nerve::simd::simd_add(pooled.data(), &features.data()[row * dim], dim);
        }
        // Check finiteness after each row's accumulation
        for (Size col = 0; col < dim; ++col)
        {
            if (!std::isfinite(pooled[col]))
            {
                throw std::overflow_error("graph pooling produced a non-finite value");
            }
        }
        if (global_pooling_type_ == "mean")
        {
            for (double &value : pooled)
            {
                value /= static_cast<double>(n);
            }
        }
    }
    return tensorFromLayerFeatures(std::move(pooled), output_size_);
}

} // namespace nerve::encoders
