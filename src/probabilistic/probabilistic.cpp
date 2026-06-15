
#include "probabilistic_detail.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace nerve::probabilistic
{
namespace
{

double pairLifetime(const Pair &pair)
{
    return detail::finiteLifetime(pair);
}

} // namespace

BootstrapPersistence::BootstrapPersistence(const Diagram &diagram)
    : original_diagram_(diagram)
    , random_generator_(5489u)
    , bootstrap_method_("with_replacement")
    , num_bootstrap_samples_(1000)
{}

std::vector<Diagram> BootstrapPersistence::bootstrapResample(Size num_samples) const
{
    std::vector<Diagram> samples;
    samples.reserve(num_samples);
    for (Size i = 0; i < num_samples; ++i)
    {
        if (bootstrap_method_ == "without_replacement")
        {
            samples.push_back(resampleWithoutReplacement());
        }
        else if (bootstrap_method_ == "gaussian_noise")
        {
            samples.push_back(resampleGaussianNoise());
        }
        else
        {
            samples.push_back(resampleWithReplacement());
        }
    }
    return samples;
}

std::vector<std::vector<double>>
BootstrapPersistence::bootstrapPersistenceIntervals(double confidence_level) const
{
    if (!std::isfinite(confidence_level) || confidence_level <= 0.0 || confidence_level >= 1.0)
    {
        throw std::invalid_argument("confidence level must be in (0, 1)");
    }
    const auto samples = bootstrapResample(num_bootstrap_samples_);
    std::vector<std::vector<double>> values_by_index;
    for (const auto &diagram : samples)
    {
        const auto &pairs = diagram.getPairs();
        if (values_by_index.size() < pairs.size())
        {
            values_by_index.resize(pairs.size());
        }
        for (std::size_t index = 0; index < pairs.size(); ++index)
        {
            values_by_index[index].push_back(pairLifetime(pairs[index]));
        }
    }

    std::vector<std::vector<double>> intervals;
    for (auto &values : values_by_index)
    {
        if (values.empty())
        {
            continue;
        }
        std::ranges::sort(values);
        const std::size_t lower_idx = static_cast<std::size_t>(
            ((1.0 - confidence_level) * 0.5) * static_cast<double>(values.size() - 1));
        const std::size_t upper_idx = static_cast<std::size_t>(
            ((1.0 + confidence_level) * 0.5) * static_cast<double>(values.size() - 1));
        intervals.push_back({values[lower_idx], values[upper_idx]});
    }
    return intervals;
}

std::vector<double> BootstrapPersistence::computeMeanPersistence() const
{
    const auto intervals = bootstrapPersistenceIntervals(0.95);
    std::vector<double> means;
    means.reserve(intervals.size());
    for (const auto &interval : intervals)
    {
        means.push_back((interval[0] + interval[1]) * 0.5);
    }
    return means;
}

std::vector<double> BootstrapPersistence::computeStdPersistence() const
{
    const auto samples = bootstrapResample(num_bootstrap_samples_);
    std::vector<std::vector<double>> values_by_index;
    for (const auto &diagram : samples)
    {
        const auto &pairs = diagram.getPairs();
        if (values_by_index.size() < pairs.size())
        {
            values_by_index.resize(pairs.size());
        }
        for (std::size_t index = 0; index < pairs.size(); ++index)
        {
            values_by_index[index].push_back(pairLifetime(pairs[index]));
        }
    }

    std::vector<double> stddev;
    stddev.reserve(values_by_index.size());
    for (const auto &values : values_by_index)
    {
        if (values.size() < 2)
        {
            stddev.push_back(0.0);
            continue;
        }
        const double mean =
            std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        double accum = 0.0;
        for (double value : values)
        {
            const double diff = value - mean;
            accum += diff * diff;
        }
        stddev.push_back(std::sqrt(accum / static_cast<double>(values.size() - 1)));
    }
    return stddev;
}

std::vector<double> BootstrapPersistence::computeConfidenceIntervals(double confidence_level) const
{
    const auto intervals = bootstrapPersistenceIntervals(confidence_level);
    std::vector<double> half_widths;
    half_widths.reserve(intervals.size());
    for (const auto &interval : intervals)
    {
        half_widths.push_back((interval[1] - interval[0]) * 0.5);
    }
    return half_widths;
}

void BootstrapPersistence::setRandomSeed(unsigned int seed)
{
    random_generator_.seed(seed);
}

void BootstrapPersistence::setBootstrapMethod(const std::string &method)
{
    if (method != "with_replacement" && method != "without_replacement" &&
        method != "gaussian_noise")
    {
        throw std::invalid_argument("unsupported bootstrap method");
    }
    bootstrap_method_ = method;
}

void BootstrapPersistence::setNumBootstrapSamples(Size num_samples)
{
    if (num_samples == 0)
    {
        throw std::invalid_argument("bootstrap sample count must be positive");
    }
    num_bootstrap_samples_ = num_samples;
}

Diagram BootstrapPersistence::resampleWithReplacement() const
{
    Diagram output;
    const auto sampled = samplePairsWithReplacement();
    for (const auto &pair : sampled)
    {
        output.addPair(pair);
    }
    return output;
}

Diagram BootstrapPersistence::resampleWithoutReplacement() const
{
    Diagram output;
    std::vector<Pair> shuffled = original_diagram_.getPairs();
    std::shuffle(shuffled.begin(), shuffled.end(), random_generator_);
    const std::size_t keep =
        shuffled.empty() ? 0 : std::max<std::size_t>(1, shuffled.size() * 8 / 10);
    shuffled.resize(keep);
    for (const auto &pair : shuffled)
    {
        output.addPair(pair);
    }
    return output;
}

Diagram BootstrapPersistence::resampleGaussianNoise(double noise_level) const
{
    if (!std::isfinite(noise_level) || noise_level < 0.0)
    {
        throw std::invalid_argument("noise level must be finite and non-negative");
    }
    Diagram output;
    std::normal_distribution<double> noise(0.0, noise_level);
    for (auto pair : original_diagram_.getPairs())
    {
        pair.birth += noise(random_generator_);
        if (!pair.isInfinite())
        {
            pair.death += noise(random_generator_);
            if (pair.death < pair.birth)
            {
                std::swap(pair.birth, pair.death);
            }
        }
        output.addPair(pair);
    }
    return output;
}

std::vector<Pair> BootstrapPersistence::samplePairsWithReplacement() const
{
    const auto &pairs = original_diagram_.getPairs();
    if (pairs.empty())
    {
        return {};
    }
    std::uniform_int_distribution<std::size_t> pick(0, pairs.size() - 1);
    std::vector<Pair> sampled;
    sampled.reserve(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        sampled.push_back(pairs[pick(random_generator_)]);
    }
    return sampled;
}

void NoiseAwarePersistence::addGaussianNoise(const std::vector<std::vector<double>> &points,
                                             double std_dev)
{
    detail::pointDimension(points);
    if (!std::isfinite(std_dev) || std_dev < 0.0)
    {
        throw std::invalid_argument("gaussian noise standard deviation must be non-negative");
    }
    noise_model_ = "gaussian";
    noise_parameter_ = std_dev;
}

void NoiseAwarePersistence::addUniformNoise(const std::vector<std::vector<double>> &points,
                                            double range)
{
    detail::pointDimension(points);
    if (!std::isfinite(range) || range < 0.0)
    {
        throw std::invalid_argument("uniform noise range must be non-negative");
    }
    noise_model_ = "uniform";
    noise_parameter_ = range;
}

void NoiseAwarePersistence::addPoissonNoise(const std::vector<std::vector<double>> &points,
                                            double lambda)
{
    detail::pointDimension(points);
    if (!std::isfinite(lambda) || lambda < 0.0)
    {
        throw std::invalid_argument("poisson noise lambda must be non-negative");
    }
    noise_model_ = "poisson";
    noise_parameter_ = lambda;
}

Diagram
NoiseAwarePersistence::computeRobustPersistence(const std::vector<std::vector<double>> &points,
                                                Size num_bootstrap_samples) const
{
    const auto ensemble = computePersistenceEnsemble(points, num_bootstrap_samples);
    if (ensemble.empty())
    {
        return Diagram();
    }
    if (robustness_method_ == "mean_lifetime")
    {
        return *std::min_element(
            ensemble.begin(), ensemble.end(), [&](const Diagram &a, const Diagram &b) {
                const double target =
                    std::accumulate(ensemble.begin(), ensemble.end(), 0.0,
                                    [](double sum, const Diagram &diagram) {
                                        return sum + detail::totalLifetime(diagram);
                                    }) /
                    static_cast<double>(ensemble.size());
                return std::abs(detail::totalLifetime(a) - target) <
                       std::abs(detail::totalLifetime(b) - target);
            });
    }
    std::vector<double> totals;
    totals.reserve(ensemble.size());
    for (const auto &diagram : ensemble)
    {
        totals.push_back(detail::totalLifetime(diagram));
    }
    const double target = detail::median(std::move(totals));
    return *std::min_element(ensemble.begin(), ensemble.end(),
                             [&](const Diagram &a, const Diagram &b) {
                                 return std::abs(detail::totalLifetime(a) - target) <
                                        std::abs(detail::totalLifetime(b) - target);
                             });
}

std::vector<std::vector<double>>
NoiseAwarePersistence::computePersistenceUncertainty(const Diagram &diagram, Size num_samples) const
{
    if (num_samples == 0)
    {
        throw std::invalid_argument("uncertainty sample count must be positive");
    }
    BootstrapPersistence bootstrap(diagram);
    bootstrap.setRandomSeed(5489u);
    bootstrap.setNumBootstrapSamples(num_samples);
    const auto stddev = bootstrap.computeStdPersistence();
    std::vector<std::vector<double>> uncertainty;
    uncertainty.reserve(diagram.getPairs().size());
    for (Size i = 0; i < diagram.getPairs().size(); ++i)
    {
        const double sigma = i < stddev.size() ? stddev[i] : 0.0;
        uncertainty.push_back({sigma, sigma});
    }
    return uncertainty;
}

std::vector<std::pair<double, double>>
NoiseAwarePersistence::computeConfidenceBands(const Diagram &diagram, double confidence_level) const
{
    std::vector<std::pair<double, double>> bands;
    if (!std::isfinite(confidence_level) || confidence_level <= 0.0 || confidence_level >= 1.0)
    {
        throw std::invalid_argument("confidence level must be in (0, 1)");
    }
    const auto uncertainty = computePersistenceUncertainty(diagram, 100);
    for (Size i = 0; i < diagram.getPairs().size(); ++i)
    {
        const auto &pair = diagram.getPairs()[i];
        const double life = pairLifetime(pair);
        const double sigma = i < uncertainty.size() ? uncertainty[i][0] : 0.0;
        const double half_width = detail::confidenceZ(confidence_level) * sigma;
        bands.emplace_back(std::max(0.0, life - half_width), life + half_width);
    }
    return bands;
}

void NoiseAwarePersistence::setNoiseModel(const std::string &model)
{
    if (model != "gaussian" && model != "uniform" && model != "poisson")
    {
        throw std::invalid_argument("unsupported noise model");
    }
    noise_model_ = model;
}

void NoiseAwarePersistence::setRobustnessMethod(const std::string &method)
{
    if (method != "median_diagram" && method != "mean_lifetime")
    {
        throw std::invalid_argument("unsupported robustness method");
    }
    robustness_method_ = method;
}

std::vector<std::vector<double>>
NoiseAwarePersistence::applyNoiseModel(const std::vector<std::vector<double>> &points) const
{
    detail::pointDimension(points);
    std::vector<std::vector<double>> noisy = points;
    auto &generator = random_generator_;
    if (noise_parameter_ <= detail::kEpsilon)
    {
        return noisy;
    }
    if (noise_model_ == "uniform")
    {
        std::uniform_real_distribution<double> noise(-noise_parameter_, noise_parameter_);
        for (auto &point : noisy)
        {
            for (double &coord : point)
            {
                coord += noise(generator);
            }
        }
    }
    else if (noise_model_ == "poisson")
    {
        std::poisson_distribution<int> noise(noise_parameter_);
        for (auto &point : noisy)
        {
            for (double &coord : point)
            {
                coord += static_cast<double>(noise(generator)) - noise_parameter_;
            }
        }
    }
    else
    {
        std::normal_distribution<double> noise(0.0, noise_parameter_);
        for (auto &point : noisy)
        {
            for (double &coord : point)
            {
                coord += noise(generator);
            }
        }
    }
    return noisy;
}

std::vector<Diagram>
NoiseAwarePersistence::computePersistenceEnsemble(const std::vector<std::vector<double>> &points,
                                                  Size num_samples) const
{
    detail::pointDimension(points);
    if (num_samples == 0)
    {
        throw std::invalid_argument("ensemble sample count must be positive");
    }
    std::vector<Diagram> ensemble;
    ensemble.reserve(num_samples);
    random_generator_ = detail::seededGenerator(
        points, static_cast<std::uint32_t>(
                    noise_model_ == "uniform" ? 2u : (noise_model_ == "poisson" ? 3u : 1u)));
    for (Size i = 0; i < num_samples; ++i)
    {
        const auto noisy_points = applyNoiseModel(points);
        ensemble.push_back(detail::diagramFromPoints(noisy_points, 2));
    }
    return ensemble;
}

} // namespace nerve::probabilistic
