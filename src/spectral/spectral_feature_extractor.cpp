#include "nerve/spectral/persistent_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace nerve::spectral
{

SpectralFeatureExtractor::SpectralFeatureExtractor(const FeatureConfig &config)
    : config_(config)
{}

std::vector<float> SpectralFeatureExtractor::extractFeatures(const SpectralDecomposition &spectrum)
{
    std::vector<float> features;

    if (config_.use_eigenvalues)
    {
        auto eigenvalue_features = extractSpectralStatistics(spectrum);
        features.insert(features.end(), eigenvalue_features.begin(), eigenvalue_features.end());
    }

    if (config_.use_eigenvectors)
    {
        auto harmonic = extractHarmonicFeatures(spectrum);
        auto nonharmonic = extractNonharmonicFeatures(spectrum);
        features.insert(features.end(), harmonic.begin(), harmonic.end());
        features.insert(features.end(), nonharmonic.begin(), nonharmonic.end());
    }

    if (config_.use_spectral_gaps)
    {
        auto gaps = computeSpectralGaps(spectrum.eigenpairs);
        features.insert(features.end(), gaps.begin(), gaps.end());
    }

    if (config_.use_participation_ratios)
    {
        auto ratios = computeParticipationRatios(spectrum.eigenpairs);
        features.insert(features.end(), ratios.begin(), ratios.end());
    }

    auto topological = extractTopologicalFeatures(spectrum);
    features.insert(features.end(), topological.begin(), topological.end());

    if (config_.normalize_features)
    {
        features = normalizeFeatureVector(features);
    }

    if (features.size() > config_.max_features)
    {
        features.resize(config_.max_features);
    }
    return features;
}

std::vector<float>
SpectralFeatureExtractor::extractHarmonicFeatures(const SpectralDecomposition &spectrum)
{
    std::vector<float> features;

    int harmonic_count = 0;
    constexpr double tolerance = 1e-10;
    for (const auto &pair : spectrum.eigenpairs)
    {
        if (std::abs(pair.eigenvalue) < tolerance)
        {
            ++harmonic_count;
            if (pair.eigenvector.size() > 0)
            {
                double max_val = 0.0;
                double sum = 0.0;
                for (double value : pair.eigenvector)
                {
                    max_val = std::max(max_val, std::abs(value));
                    sum += value * value;
                }
                features.push_back(static_cast<float>(max_val));
                features.push_back(static_cast<float>(std::sqrt(sum)));
            }
        }
    }

    features.push_back(static_cast<float>(harmonic_count));
    return features;
}

std::vector<float>
SpectralFeatureExtractor::extractNonharmonicFeatures(const SpectralDecomposition &spectrum)
{
    std::vector<float> features;

    constexpr double tolerance = 1e-10;
    double min_nonzero = std::numeric_limits<double>::max();
    double max_nonzero = 0.0;
    double sum_nonzero = 0.0;
    int nonzero_count = 0;

    for (const auto &pair : spectrum.eigenpairs)
    {
        if (std::abs(pair.eigenvalue) >= tolerance)
        {
            min_nonzero = std::min(min_nonzero, pair.eigenvalue);
            max_nonzero = std::max(max_nonzero, pair.eigenvalue);
            sum_nonzero += pair.eigenvalue;
            ++nonzero_count;
        }
    }

    if (nonzero_count > 0)
    {
        features.push_back(static_cast<float>(min_nonzero));
        features.push_back(static_cast<float>(max_nonzero));
        features.push_back(static_cast<float>(sum_nonzero / nonzero_count));
        features.push_back(static_cast<float>(nonzero_count));
    }
    return features;
}

std::vector<float>
SpectralFeatureExtractor::extractSpectralStatistics(const SpectralDecomposition &spectrum)
{
    std::vector<float> features;
    if (spectrum.eigenpairs.empty())
    {
        return features;
    }

    std::vector<double> eigenvalues;
    eigenvalues.reserve(spectrum.eigenpairs.size());
    for (const auto &pair : spectrum.eigenpairs)
    {
        eigenvalues.push_back(pair.eigenvalue);
    }
    std::ranges::sort(eigenvalues);

    const double min_val = eigenvalues.front();
    const double max_val = eigenvalues.back();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (double value : eigenvalues)
    {
        sum += value;
        sum_sq += value * value;
    }

    const double eigenvalue_count = static_cast<double>(eigenvalues.size());
    const double mean = sum / eigenvalue_count;
    const double variance = (sum_sq / eigenvalue_count) - mean * mean;
    const double std_dev = std::sqrt(std::max(0.0, variance));
    const std::size_t n = eigenvalues.size();

    features.push_back(static_cast<float>(min_val));
    features.push_back(static_cast<float>(max_val));
    features.push_back(static_cast<float>(mean));
    features.push_back(static_cast<float>(std_dev));
    features.push_back(static_cast<float>(n));
    features.push_back(static_cast<float>(eigenvalues[n / 4]));
    features.push_back(static_cast<float>(eigenvalues[n / 2]));
    features.push_back(static_cast<float>(eigenvalues[(3 * n) / 4]));
    return features;
}

std::vector<float>
SpectralFeatureExtractor::extractTopologicalFeatures(const SpectralDecomposition &spectrum)
{
    std::vector<float> features;

    constexpr double tolerance = 1e-10;
    int zero_eigenvalues = 0;
    for (const auto &pair : spectrum.eigenpairs)
    {
        if (std::abs(pair.eigenvalue) < tolerance)
        {
            ++zero_eigenvalues;
        }
    }
    features.push_back(static_cast<float>(zero_eigenvalues));

    double spectral_sum = 0.0;
    for (const auto &pair : spectrum.eigenpairs)
    {
        if (pair.eigenvalue > 0)
        {
            spectral_sum += std::log(pair.eigenvalue);
        }
    }
    features.push_back(static_cast<float>(spectral_sum));
    return features;
}

std::vector<std::vector<float>>
SpectralFeatureExtractor::extractFeaturesBatch(const std::vector<SpectralDecomposition> &spectra)
{
    std::vector<std::vector<float>> batch_features;
    batch_features.reserve(spectra.size());
    for (const auto &spectrum : spectra)
    {
        batch_features.push_back(extractFeatures(spectrum));
    }
    return batch_features;
}

std::vector<float>
SpectralFeatureExtractor::normalizeFeatureVector(const std::vector<float> &features)
{
    if (features.empty())
    {
        return features;
    }

    float sum = 0.0F;
    float sum_sq = 0.0F;
    for (float value : features)
    {
        sum += value;
        sum_sq += value * value;
    }

    const float mean = sum / static_cast<float>(features.size());
    const float variance = (sum_sq / static_cast<float>(features.size())) - (mean * mean);
    float std_dev = std::sqrt(std::max(0.0F, variance));
    if (std_dev < 1e-10F)
    {
        std_dev = 1.0F;
    }

    std::vector<float> normalized;
    normalized.reserve(features.size());
    for (float value : features)
    {
        normalized.push_back((value - mean) / std_dev);
    }
    return normalized;
}

std::vector<float>
SpectralFeatureExtractor::computeSpectralGaps(const std::vector<Eigenpair> &eigenpairs)
{
    std::vector<float> gaps;
    if (eigenpairs.size() < 2)
    {
        return gaps;
    }

    std::vector<double> eigenvalues;
    eigenvalues.reserve(eigenpairs.size());
    for (const auto &pair : eigenpairs)
    {
        eigenvalues.push_back(pair.eigenvalue);
    }
    std::ranges::sort(eigenvalues);

    for (std::size_t i = 1; i < eigenvalues.size() && i < 10; ++i)
    {
        gaps.push_back(static_cast<float>(eigenvalues[i] - eigenvalues[i - 1]));
    }
    return gaps;
}

std::vector<float>
SpectralFeatureExtractor::computeParticipationRatios(const std::vector<Eigenpair> &eigenpairs)
{
    std::vector<float> ratios;
    ratios.reserve(eigenpairs.size());

    for (const auto &pair : eigenpairs)
    {
        if (pair.eigenvector.size() == 0)
        {
            ratios.push_back(0.0F);
            continue;
        }
        double sum_fourth = 0.0;
        for (double value : pair.eigenvector)
        {
            const double value_sq = value * value;
            sum_fourth += value_sq * value_sq;
        }

        if (sum_fourth > 1e-10)
        {
            const double vector_size = static_cast<double>(pair.eigenvector.size());
            ratios.push_back(static_cast<float>(1.0 / (sum_fourth * vector_size)));
        }
        else
        {
            ratios.push_back(static_cast<float>(pair.eigenvector.size()));
        }
    }
    return ratios;
}

} // namespace nerve::spectral
