
#include "nerve/encoders/encoders.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

std::vector<double> fitFeatureSize(std::vector<double> features, Size output_size)
{
    if (features.size() > output_size)
    {
        features.resize(output_size);
    }
    features.resize(output_size, 0.0);
    return features;
}

void validateDiagramPair(const persistence::Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    const bool infinite_death = pair.isInfinite();
    if (!std::isfinite(pair.birth) || (!finite_death && !infinite_death) || pair.dimension < 0 ||
        (finite_death && pair.death < pair.birth))
    {
        throw std::invalid_argument("topological encoder diagram contains an invalid pair");
    }
}

bool hasFiniteLifetime(const persistence::Pair &pair)
{
    validateDiagramPair(pair);
    return std::isfinite(pair.birth) && std::isfinite(pair.death) && pair.death > pair.birth;
}

double finiteLifetime(const persistence::Pair &pair)
{
    return pair.death - pair.birth;
}

} // namespace

TopologicalEncoder::TopologicalEncoder(Size feature_dim)
    : enable_betti_numbers_(true)
    , enable_persistence_landscape_(true)
    , enable_persistence_images_(false)
    , enable_persistence_entropy_(false)
    , landscape_resolution_(100)
    , image_resolution_(50)
    , image_sigma_(1.0)
{
    input_size_ = 0;
    output_size_ = feature_dim;
}
Tensor TopologicalEncoder::encode(const std::vector<std::vector<double>> &data) const
{
    std::vector<double> features;
    features.reserve(output_size_);
    for (const auto &row : data)
    {
        for (double value : row)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    "topological encoder input contains a non-finite value");
            }
            if (features.size() < output_size_)
            {
                features.push_back(value);
            }
        }
    }
    return Tensor(fitFeatureSize(std::move(features), output_size_), {output_size_});
}
Tensor TopologicalEncoder::encode(const SimplicialComplex &complex) const
{
    std::vector<double> features;
    if (enable_betti_numbers_)
    {
        auto betti = extractBettiNumbers(complex);
        features.insert(features.end(), betti.begin(), betti.end());
    }
    return Tensor(fitFeatureSize(std::move(features), output_size_), {output_size_});
}
Tensor TopologicalEncoder::encode(const Diagram &diagram) const
{
    std::vector<double> features;
    if (enable_persistence_landscape_)
    {
        auto landscape = extractPersistenceLandscape(diagram);
        features.insert(features.end(), landscape.begin(), landscape.end());
    }
    if (enable_persistence_images_)
    {
        auto images = extractPersistenceImages(diagram);
        features.insert(features.end(), images.begin(), images.end());
    }
    if (enable_persistence_entropy_)
    {
        auto entropy = extractPersistenceEntropy(diagram);
        features.insert(features.end(), entropy.begin(), entropy.end());
    }
    return Tensor(fitFeatureSize(std::move(features), output_size_), {output_size_});
}
std::vector<Tensor> TopologicalEncoder::encodeBatch(
    const std::vector<std::vector<std::vector<double>>> &batch_data) const
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
TopologicalEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        batch_features.push_back(encode(complex));
    }
    return batch_features;
}
std::vector<Tensor>
TopologicalEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> batch_features;
    batch_features.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        batch_features.push_back(encode(diagram));
    }
    return batch_features;
}
void TopologicalEncoder::setInputSize(Size input_size)
{
    input_size_ = input_size;
}
void TopologicalEncoder::setOutputSize(Size output_size)
{
    output_size_ = output_size;
}
void TopologicalEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
    for (const auto &[key, value] : params)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("topological encoder parameters must be finite");
        }
        if (key == "landscape_resolution")
        {
            if (value < 0.0)
            {
                throw std::invalid_argument("landscape_resolution must be non-negative");
            }
            landscape_resolution_ = static_cast<Size>(value);
        }
        else if (key == "image_resolution")
        {
            if (value < 0.0)
            {
                throw std::invalid_argument("image_resolution must be non-negative");
            }
            image_resolution_ = static_cast<Size>(value);
        }
        else if (key == "image_sigma")
        {
            if (value <= 0.0 || !std::isfinite(value))
            {
                throw std::invalid_argument("image_sigma must be finite and positive");
            }
            image_sigma_ = value;
        }
    }
}
void TopologicalEncoder::enableBettiNumbers(bool enable)
{
    enable_betti_numbers_ = enable;
}
void TopologicalEncoder::enablePersistenceLandscape(bool enable)
{
    enable_persistence_landscape_ = enable;
}
void TopologicalEncoder::enablePersistenceImages(bool enable)
{
    enable_persistence_images_ = enable;
}
void TopologicalEncoder::enablePersistenceEntropy(bool enable)
{
    enable_persistence_entropy_ = enable;
}
std::vector<int> TopologicalEncoder::extractBettiNumbers(const SimplicialComplex &complex) const
{
    const Dimension complex_dim = complex.maxDimension();
    const Size max_dim =
        complex_dim < 0 ? Size{2} : std::max<Size>(Size{2}, static_cast<Size>(complex_dim));
    const auto result = persistence::computeExactPersistenceZ2(complex, max_dim);

    std::vector<int> betti(std::max<Size>(Size{3}, result.betti_numbers.size()), 0);
    for (Size dim = 0; dim < result.betti_numbers.size(); ++dim)
    {
        if (result.betti_numbers[dim] > static_cast<Size>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("Betti number exceeds int range");
        }
        betti[dim] = static_cast<int>(result.betti_numbers[dim]);
    }
    return betti;
}
std::vector<double> TopologicalEncoder::extractPersistenceLandscape(const Diagram &diagram) const
{
    return computePersistenceLandscape(diagram, landscape_resolution_);
}
std::vector<double> TopologicalEncoder::extractPersistenceImages(const Diagram &diagram) const
{
    return computePersistenceImage(diagram, image_resolution_, image_sigma_);
}
std::vector<double> TopologicalEncoder::extractPersistenceEntropy(const Diagram &diagram) const
{
    std::vector<double> entropy;
    entropy.push_back(computePersistenceEntropy(diagram));
    return entropy;
}
Size TopologicalEncoder::getInputSize() const
{
    return input_size_;
}
Size TopologicalEncoder::getOutputSize() const
{
    return output_size_;
}
std::string TopologicalEncoder::getEncoderType() const
{
    return "Topological";
}
std::vector<double> TopologicalEncoder::computePersistenceLandscape(const Diagram &diagram,
                                                                    Size resolution) const
{
    std::vector<double> landscape(resolution, 0.0);
    if (resolution == 0)
    {
        return landscape;
    }

    const auto &pairs = diagram.getPairs();
    double min_birth = std::numeric_limits<double>::infinity();
    double max_death = -std::numeric_limits<double>::infinity();
    for (const auto &pair : pairs)
    {
        if (!hasFiniteLifetime(pair))
        {
            continue;
        }
        min_birth = std::min(min_birth, pair.birth);
        max_death = std::max(max_death, pair.death);
    }
    if (!std::isfinite(min_birth) || max_death <= min_birth)
    {
        return landscape;
    }

    const double span = max_death - min_birth;
    for (Size i = 0; i < resolution; ++i)
    {
        const double t = resolution == 1 ? min_birth + 0.5 * span
                                         : min_birth + span * static_cast<double>(i) /
                                                           static_cast<double>(resolution - 1);
        double value = 0.0;
        for (const auto &pair : pairs)
        {
            if (!hasFiniteLifetime(pair))
            {
                continue;
            }
            const double tent = std::min(t - pair.birth, pair.death - t);
            value = std::max(value, std::max(0.0, tent));
        }
        landscape[i] = value;
    }
    return landscape;
}
std::vector<double> TopologicalEncoder::computePersistenceImage(const Diagram &diagram,
                                                                Size resolution, double sigma) const
{
    std::vector<double> image(resolution * resolution, 0.0);
    if (resolution == 0)
    {
        return image;
    }
    if (sigma <= 0.0 || !std::isfinite(sigma))
    {
        throw std::invalid_argument("image_sigma must be finite and positive");
    }

    const auto &pairs = diagram.getPairs();
    double min_birth = std::numeric_limits<double>::infinity();
    double max_birth = -std::numeric_limits<double>::infinity();
    double max_persistence = 0.0;
    for (const auto &pair : pairs)
    {
        if (!hasFiniteLifetime(pair))
        {
            continue;
        }
        min_birth = std::min(min_birth, pair.birth);
        max_birth = std::max(max_birth, pair.birth);
        max_persistence = std::max(max_persistence, finiteLifetime(pair));
    }
    if (!std::isfinite(min_birth) || max_persistence <= 0.0)
    {
        return image;
    }

    const double birth_span = std::max(max_birth - min_birth, 1.0);
    const double persistence_span = std::max(max_persistence, 1.0);
    const double two_sigma_sq = 2.0 * sigma * sigma;
    // Pre-allocate batch buffer for SIMD exp
    std::vector<double> exp_args(resolution);
    for (const auto &pair : pairs)
    {
        if (!hasFiniteLifetime(pair))
        {
            continue;
        }
        const double persistence = finiteLifetime(pair);
        for (Size y = 0; y < resolution; ++y)
        {
            const double py =
                persistence_span * (static_cast<double>(y) + 0.5) / static_cast<double>(resolution);
            const double dp = (py - persistence) / persistence_span;
            // exp_dp is constant across all x for this (pair, y) row
            const double exp_dp = std::exp(-(dp * dp) / two_sigma_sq);

            // Compute -db*db / two_sigma_sq for all x and batch the exp
            for (Size x = 0; x < resolution; ++x)
            {
                const double bx = min_birth + birth_span * (static_cast<double>(x) + 0.5) /
                                                  static_cast<double>(resolution);
                const double db = (bx - pair.birth) / birth_span;
                exp_args[x] = -(db * db) / two_sigma_sq;
            }
            nerve::simd::simd_exp(exp_args.data(), resolution);

            // Accumulate: image += persistence * exp_dp * exp(-db^2 / two_sigma_sq)
            for (Size x = 0; x < resolution; ++x)
            {
                image[y * resolution + x] += persistence * exp_dp * exp_args[x];
            }
        }
    }
    return image;
}
double TopologicalEncoder::computePersistenceEntropy(const Diagram &diagram) const
{
    const auto &pairs = diagram.getPairs();
    double total_persistence = 0.0;
    for (const auto &pair : pairs)
    {
        if (hasFiniteLifetime(pair))
        {
            total_persistence += finiteLifetime(pair);
        }
    }
    if (total_persistence <= 0.0 || !std::isfinite(total_persistence))
    {
        return 0.0;
    }

    double entropy = 0.0;
    for (const auto &pair : pairs)
    {
        if (hasFiniteLifetime(pair))
        {
            const double p = finiteLifetime(pair) / total_persistence;
            if (p > 0.0)
            {
                entropy -= p * std::log(p);
            }
        }
    }
    return entropy;
}

} // namespace nerve::encoders
