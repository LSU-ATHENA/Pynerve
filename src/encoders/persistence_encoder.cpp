#include "nerve/encoders/encoders.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

std::vector<double> fitPersistenceFeatures(std::vector<double> features, Size output_size)
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
        throw std::invalid_argument("persistence encoder diagram contains an invalid pair");
    }
}

bool finitePair(const persistence::Pair &pair)
{
    validateDiagramPair(pair);
    return std::isfinite(pair.birth) && std::isfinite(pair.death) && pair.death > pair.birth;
}

double pairLifetime(const persistence::Pair &pair)
{
    return pair.death - pair.birth;
}

Diagram diagramFromComplex(const SimplicialComplex &complex)
{
    const Dimension complex_dim = complex.maxDimension();
    const Size max_dim = complex_dim < 0 ? Size{0} : static_cast<Size>(complex_dim);
    const auto result = persistence::computeExactPersistenceZ2(complex, max_dim);
    Diagram diagram;
    for (const auto &pair : result.pairs)
    {
        diagram.addPair(pair);
    }
    return diagram;
}

} // namespace

PersistenceEncoder::PersistenceEncoder(Size output_dim)
    : encoding_strategy_("statistics")
    , num_landscapes_(1)
    , landscape_resolution_(64)
    , image_resolution_(16)
    , image_sigma_(1.0)
    , use_moments_(true)
    , use_quantiles_(true)
{
    input_size_ = 0;
    output_size_ = output_dim;
}

Tensor PersistenceEncoder::encode(const std::vector<std::vector<double>> &data) const
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
                    "persistence encoder input contains a non-finite value");
            }
            if (features.size() < output_size_)
            {
                features.push_back(value);
            }
        }
    }
    return Tensor(fitPersistenceFeatures(std::move(features), output_size_), {output_size_});
}

Tensor PersistenceEncoder::encode(const SimplicialComplex &complex) const
{
    return encode(diagramFromComplex(complex));
}

Tensor PersistenceEncoder::encode(const Diagram &diagram) const
{
    if (encoding_strategy_ == "landscapes")
    {
        return encodeLandscapes(diagram);
    }
    if (encoding_strategy_ == "images")
    {
        return encodeImages(diagram);
    }
    if (encoding_strategy_ == "vectors")
    {
        return encodePersistenceVectors(diagram);
    }
    return encodeStatistics(diagram);
}

std::vector<Tensor> PersistenceEncoder::encodeBatch(
    const std::vector<std::vector<std::vector<double>>> &batch_data) const
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
PersistenceEncoder::encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const
{
    std::vector<Tensor> output;
    output.reserve(batch_complexes.size());
    for (const auto &complex : batch_complexes)
    {
        output.push_back(encode(complex));
    }
    return output;
}

std::vector<Tensor>
PersistenceEncoder::encodeBatch(const std::vector<Diagram> &batch_diagrams) const
{
    std::vector<Tensor> output;
    output.reserve(batch_diagrams.size());
    for (const auto &diagram : batch_diagrams)
    {
        output.push_back(encode(diagram));
    }
    return output;
}

void PersistenceEncoder::setInputSize(Size input_size)
{
    input_size_ = input_size;
}

void PersistenceEncoder::setOutputSize(Size output_size)
{
    output_size_ = output_size;
}

void PersistenceEncoder::setParameters(const std::map<std::string, double> &params)
{
    parameters_ = params;
    for (const auto &[key, value] : params)
    {
        if (value < 0.0 || !std::isfinite(value))
        {
            throw std::invalid_argument(
                "persistence encoder parameters must be finite and non-negative");
        }
        if (key == "num_landscapes")
        {
            num_landscapes_ = static_cast<Size>(value);
        }
        else if (key == "landscape_resolution")
        {
            landscape_resolution_ = static_cast<Size>(value);
        }
        else if (key == "image_resolution")
        {
            image_resolution_ = static_cast<Size>(value);
        }
        else if (key == "image_sigma")
        {
            if (value <= 0.0)
            {
                throw std::invalid_argument("image_sigma must be positive");
            }
            image_sigma_ = value;
        }
        else if (key == "use_moments")
        {
            use_moments_ = value != 0.0;
        }
        else if (key == "use_quantiles")
        {
            use_quantiles_ = value != 0.0;
        }
    }
}

void PersistenceEncoder::setEncodingStrategy(const std::string &strategy)
{
    if (strategy != "statistics" && strategy != "landscapes" && strategy != "images" &&
        strategy != "vectors")
    {
        throw std::invalid_argument("unsupported persistence encoding strategy");
    }
    encoding_strategy_ = strategy;
}

void PersistenceEncoder::setLandscapesParams(Size num_landscapes, Size resolution)
{
    num_landscapes_ = num_landscapes;
    landscape_resolution_ = resolution;
}

void PersistenceEncoder::setImagesParams(Size resolution, double sigma)
{
    if (sigma <= 0.0 || !std::isfinite(sigma))
    {
        throw std::invalid_argument("image sigma must be finite and positive");
    }
    image_resolution_ = resolution;
    image_sigma_ = sigma;
}

void PersistenceEncoder::setStatisticsParams(bool use_moments, bool use_quantiles)
{
    use_moments_ = use_moments;
    use_quantiles_ = use_quantiles;
}

Tensor PersistenceEncoder::encodeLandscapes(const Diagram &diagram) const
{
    return Tensor(fitPersistenceFeatures(
                      computePersistenceLandscapes(diagram, num_landscapes_, landscape_resolution_),
                      output_size_),
                  {output_size_});
}

Tensor PersistenceEncoder::encodeImages(const Diagram &diagram) const
{
    return Tensor(
        fitPersistenceFeatures(computePersistenceImages(diagram, image_resolution_, image_sigma_),
                               output_size_),
        {output_size_});
}

Tensor PersistenceEncoder::encodeStatistics(const Diagram &diagram) const
{
    return Tensor(fitPersistenceFeatures(computePersistenceStatistics(diagram), output_size_),
                  {output_size_});
}

Tensor PersistenceEncoder::encodePersistenceVectors(const Diagram &diagram) const
{
    return Tensor(fitPersistenceFeatures(computePersistenceVectors(diagram), output_size_),
                  {output_size_});
}

Size PersistenceEncoder::getInputSize() const
{
    return input_size_;
}

Size PersistenceEncoder::getOutputSize() const
{
    return output_size_;
}

std::string PersistenceEncoder::getEncoderType() const
{
    return "Persistence";
}

std::vector<double> PersistenceEncoder::computePersistenceLandscapes(const Diagram &diagram,
                                                                     Size num_landscapes,
                                                                     Size resolution) const
{
    std::vector<double> landscape(num_landscapes * resolution, 0.0);
    if (num_landscapes == 0 || resolution == 0)
    {
        return landscape;
    }

    double min_birth = std::numeric_limits<double>::infinity();
    double max_death = -std::numeric_limits<double>::infinity();
    for (const auto &pair : diagram.getPairs())
    {
        if (finitePair(pair))
        {
            min_birth = std::min(min_birth, pair.birth);
            max_death = std::max(max_death, pair.death);
        }
    }
    if (!std::isfinite(min_birth) || max_death <= min_birth)
    {
        return landscape;
    }

    const double span = max_death - min_birth;
    std::vector<double> tents;
    tents.reserve(diagram.count());
    for (Size x = 0; x < resolution; ++x)
    {
        const double t = resolution == 1 ? min_birth + 0.5 * span
                                         : min_birth + span * static_cast<double>(x) /
                                                           static_cast<double>(resolution - 1);
        tents.clear();
        for (const auto &pair : diagram.getPairs())
        {
            if (finitePair(pair))
            {
                tents.push_back(std::max(0.0, std::min(t - pair.birth, pair.death - t)));
            }
        }
        std::ranges::sort(tents, std::greater<>());
        for (Size k = 0; k < num_landscapes && k < tents.size(); ++k)
        {
            landscape[k * resolution + x] = tents[k];
        }
    }
    return landscape;
}

std::vector<double> PersistenceEncoder::computePersistenceImages(const Diagram &diagram,
                                                                 Size resolution,
                                                                 double sigma) const
{
    std::vector<double> image(resolution * resolution, 0.0);
    if (resolution == 0)
    {
        return image;
    }
    if (sigma <= 0.0 || !std::isfinite(sigma))
    {
        throw std::invalid_argument("image sigma must be finite and positive");
    }

    double min_birth = std::numeric_limits<double>::infinity();
    double max_birth = -std::numeric_limits<double>::infinity();
    double max_lifetime = 0.0;
    for (const auto &pair : diagram.getPairs())
    {
        if (finitePair(pair))
        {
            min_birth = std::min(min_birth, pair.birth);
            max_birth = std::max(max_birth, pair.birth);
            max_lifetime = std::max(max_lifetime, pairLifetime(pair));
        }
    }
    if (!std::isfinite(min_birth) || max_lifetime <= 0.0)
    {
        return image;
    }

    const double birth_span = std::max(max_birth - min_birth, 1.0);
    const double persistence_span = std::max(max_lifetime, 1.0);
    const double two_sigma_sq = 2.0 * sigma * sigma;
    const double inv_res = 1.0 / static_cast<double>(resolution);
    // Pre-allocate batch buffer for the x-loop; reuse across all (pair, y) rows
    std::vector<double> exp_args(resolution);
    for (const auto &pair : diagram.getPairs())
    {
        if (!finitePair(pair))
        {
            continue;
        }
        const double lifetime = pairLifetime(pair);
        // db(x) = (min_birth - pair.birth) / birth_span + (x + 0.5) / resolution
        const double db_base = (min_birth - pair.birth) / birth_span;
        for (Size y = 0; y < resolution; ++y)
        {
            const double py = persistence_span * (static_cast<double>(y) + 0.5) * inv_res;
            const double dp = (py - lifetime) / persistence_span;
            const double exp_dp = std::exp(-(dp * dp) / two_sigma_sq);
            // Fill buffer with -db^2 / two_sigma_sq for all x
            for (Size x = 0; x < resolution; ++x)
            {
                const double db = db_base + (static_cast<double>(x) + 0.5) * inv_res;
                exp_args[x] = -(db * db) / two_sigma_sq;
            }
            // Batched SIMD exp: replaces each element with exp(element)
            nerve::simd::simd_exp(exp_args.data(), resolution);
            // Accumulate: image += lifetime * exp(-dp^2 / two_sigma_sq) * exp(-db^2 / two_sigma_sq)
            for (Size x = 0; x < resolution; ++x)
            {
                image[y * resolution + x] += lifetime * exp_dp * exp_args[x];
            }
        }
    }
    return image;
}

std::vector<double> PersistenceEncoder::computePersistenceStatistics(const Diagram &diagram) const
{
    std::vector<double> lifetimes;
    std::vector<double> features;
    Size infinite_count = 0;
    for (const auto &pair : diagram.getPairs())
    {
        if (finitePair(pair))
        {
            lifetimes.push_back(pairLifetime(pair));
        }
        else if (pair.isInfinite())
        {
            ++infinite_count;
        }
    }

    features.push_back(static_cast<double>(lifetimes.size()));
    features.push_back(static_cast<double>(infinite_count));
    if (lifetimes.empty())
    {
        return features;
    }
    std::ranges::sort(lifetimes);
    const double total = std::accumulate(lifetimes.begin(), lifetimes.end(), 0.0);
    const double mean = total / static_cast<double>(lifetimes.size());
    features.push_back(total);
    features.push_back(lifetimes.back());
    if (use_moments_)
    {
        double variance = 0.0;
        for (double lifetime : lifetimes)
        {
            const double diff = lifetime - mean;
            variance += diff * diff;
        }
        features.push_back(mean);
        features.push_back(variance / static_cast<double>(lifetimes.size()));
    }
    if (use_quantiles_)
    {
        features.push_back(lifetimes[lifetimes.size() / 4]);
        features.push_back(lifetimes[lifetimes.size() / 2]);
        features.push_back(lifetimes[(lifetimes.size() * 3) / 4]);
    }
    return features;
}

std::vector<double> PersistenceEncoder::computePersistenceVectors(const Diagram &diagram) const
{
    std::vector<persistence::Pair> pairs = diagram.getPairs();
    for (const auto &pair : pairs)
    {
        validateDiagramPair(pair);
    }
    std::ranges::sort(pairs, [](const auto &lhs, const auto &rhs) {
        return pairLifetime(lhs) > pairLifetime(rhs);
    });

    std::vector<double> features;
    features.reserve(pairs.size() * 4);
    for (const auto &pair : pairs)
    {
        if (!finitePair(pair))
        {
            continue;
        }
        features.push_back(pair.birth);
        features.push_back(pair.death);
        features.push_back(pairLifetime(pair));
        features.push_back(static_cast<double>(pair.dimension));
    }
    return features;
}

} // namespace nerve::encoders
