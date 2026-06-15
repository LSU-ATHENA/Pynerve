#include "probabilistic_detail.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nerve::probabilistic
{
namespace
{

constexpr double kTwoPi = 6.28318530717958647692;

double normalDensity(double value, double mean, double variance)
{
    const double var = std::max(variance, detail::kEpsilon);
    const double centered = value - mean;
    return std::exp(-(centered * centered) / (2.0 * var)) / std::sqrt(kTwoPi * var);
}

std::vector<double> pairLifetimes(const std::vector<Pair> &pairs)
{
    std::vector<double> values;
    values.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        values.push_back(detail::finiteLifetime(pair));
    }
    return values;
}

} // namespace

ProbabilisticPersistenceDiagram::ProbabilisticPersistenceDiagram(const Diagram &diagram)
    : original_diagram_(diagram)
    , mean_pairs_(diagram.getPairs())
    , model_type_("empirical")
    , random_generator_(5489u)
{}

void ProbabilisticPersistenceDiagram::fitGaussianModel()
{
    mean_pairs_ = original_diagram_.getPairs();
    const auto &pairs = original_diagram_.getPairs();
    covariance_matrix_ = {{0.0, 0.0}, {0.0, 0.0}};
    for (const auto &pair : pairs)
    {
        detail::requireValidPair(pair, "persistence pair");
    }
    if (pairs.size() < 2)
    {
        model_type_ = "gaussian";
        model_parameters_ = {0.0, detail::kEpsilon};
        return;
    }

    std::vector<double> births;
    std::vector<double> deaths;
    births.reserve(pairs.size());
    deaths.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        births.push_back(pair.birth);
        deaths.push_back(pair.isInfinite() ? pair.birth : pair.death);
    }
    const double birth_mean = detail::mean(births);
    const double death_mean = detail::mean(deaths);
    double birth_var = 0.0;
    double death_var = 0.0;
    double covariance = 0.0;
    for (Size i = 0; i < births.size(); ++i)
    {
        const double birth_delta = births[i] - birth_mean;
        const double death_delta = deaths[i] - death_mean;
        birth_var += birth_delta * birth_delta;
        death_var += death_delta * death_delta;
        covariance += birth_delta * death_delta;
    }
    const double denom = static_cast<double>(pairs.size() - 1);
    covariance_matrix_ = {
        {birth_var / denom, covariance / denom},
        {covariance / denom, death_var / denom},
    };
    model_type_ = "gaussian";
    model_parameters_ = {birth_mean, death_mean};
}

void ProbabilisticPersistenceDiagram::fitMixtureModel(Size num_components)
{
    if (num_components == 0)
    {
        throw std::invalid_argument("mixture component count must be positive");
    }
    const auto lifetimes = pairLifetimes(original_diagram_.getPairs());
    if (lifetimes.empty())
    {
        model_type_ = "mixture";
        model_parameters_.clear();
        return;
    }

    std::vector<double> sorted = lifetimes;
    std::sort(sorted.begin(), sorted.end());
    const Size components = std::min(num_components, sorted.size());
    model_parameters_.clear();
    model_parameters_.push_back(static_cast<double>(components));
    for (Size c = 0; c < components; ++c)
    {
        const Size begin = c * sorted.size() / components;
        const Size end = std::max(begin + 1, (c + 1) * sorted.size() / components);
        std::vector<double> bucket(sorted.begin() + static_cast<std::ptrdiff_t>(begin),
                                   sorted.begin() + static_cast<std::ptrdiff_t>(end));
        model_parameters_.push_back(static_cast<double>(bucket.size()) /
                                    static_cast<double>(sorted.size()));
        model_parameters_.push_back(detail::mean(bucket));
        model_parameters_.push_back(std::max(detail::variance(bucket), detail::kEpsilon));
    }
    mean_pairs_ = original_diagram_.getPairs();
    model_type_ = "mixture";
}

void ProbabilisticPersistenceDiagram::fitKernelModel(double bandwidth)
{
    if (!std::isfinite(bandwidth) || bandwidth <= 0.0)
    {
        throw std::invalid_argument("kernel bandwidth must be positive");
    }
    mean_pairs_ = original_diagram_.getPairs();
    model_type_ = "kernel";
    model_parameters_ = {bandwidth};
}

Diagram ProbabilisticPersistenceDiagram::sampleDiagram() const
{
    Diagram sample;
    if (mean_pairs_.empty())
    {
        return sample;
    }

    const double birth_sigma =
        covariance_matrix_.empty() ? 0.0 : std::sqrt(std::max(0.0, covariance_matrix_[0][0]));
    const double death_sigma = covariance_matrix_.size() < 2
                                   ? birth_sigma
                                   : std::sqrt(std::max(0.0, covariance_matrix_[1][1]));
    for (auto pair : mean_pairs_)
    {
        if (model_type_ == "gaussian")
        {
            if (birth_sigma > detail::kEpsilon)
            {
                std::normal_distribution<double> birth_noise(0.0, birth_sigma);
                pair.birth += birth_noise(random_generator_);
            }
            if (!pair.isInfinite())
            {
                if (death_sigma > detail::kEpsilon)
                {
                    std::normal_distribution<double> death_noise(0.0, death_sigma);
                    pair.death += death_noise(random_generator_);
                }
            }
        }
        else if (model_type_ == "kernel" && !model_parameters_.empty())
        {
            std::normal_distribution<double> kernel_noise(0.0, model_parameters_.front());
            pair.birth += kernel_noise(random_generator_);
            if (!pair.isInfinite())
            {
                pair.death += kernel_noise(random_generator_);
            }
        }
        if (!pair.isInfinite() && pair.death < pair.birth)
        {
            std::swap(pair.birth, pair.death);
        }
        sample.addPair(pair);
    }
    return sample;
}

std::vector<Diagram> ProbabilisticPersistenceDiagram::sampleDiagrams(Size num_samples) const
{
    std::vector<Diagram> samples;
    samples.reserve(num_samples);
    for (Size i = 0; i < num_samples; ++i)
    {
        samples.push_back(sampleDiagram());
    }
    return samples;
}

double ProbabilisticPersistenceDiagram::computeProbability(const Pair &pair) const
{
    if (model_type_ == "mixture")
    {
        return mixtureProbability(pair);
    }
    if (model_type_ == "kernel")
    {
        return kernelProbability(pair);
    }
    return gaussianProbability(pair);
}

double ProbabilisticPersistenceDiagram::computeLikelihood(const Diagram &diagram) const
{
    if (diagram.isEmpty())
    {
        return 0.0;
    }
    double log_sum = 0.0;
    for (const auto &pair : diagram.getPairs())
    {
        log_sum += std::log(std::max(computeProbability(pair), detail::kEpsilon));
    }
    return std::exp(log_sum / static_cast<double>(diagram.getPairs().size()));
}

std::vector<std::vector<double>> ProbabilisticPersistenceDiagram::computeCovarianceMatrix() const
{
    return covariance_matrix_;
}

std::vector<double> ProbabilisticPersistenceDiagram::computeMarginalVariances() const
{
    std::vector<double> variances;
    for (Size i = 0; i < covariance_matrix_.size(); ++i)
    {
        if (i < covariance_matrix_[i].size())
        {
            variances.push_back(covariance_matrix_[i][i]);
        }
    }
    return variances;
}

double ProbabilisticPersistenceDiagram::gaussianProbability(const Pair &pair) const
{
    const auto lifetimes = pairLifetimes(mean_pairs_);
    const double lifetime_mean = detail::mean(lifetimes);
    const double lifetime_var = std::max(detail::variance(lifetimes), detail::kEpsilon);
    return normalDensity(detail::finiteLifetime(pair), lifetime_mean, lifetime_var);
}

double ProbabilisticPersistenceDiagram::mixtureProbability(const Pair &pair) const
{
    if (model_parameters_.empty())
    {
        return gaussianProbability(pair);
    }
    const Size components = static_cast<Size>(model_parameters_.front());
    const double lifetime = detail::finiteLifetime(pair);
    double probability = 0.0;
    for (Size c = 0; c < components; ++c)
    {
        const Size offset = 1 + c * 3;
        probability +=
            model_parameters_[offset] *
            normalDensity(lifetime, model_parameters_[offset + 1], model_parameters_[offset + 2]);
    }
    return probability;
}

double ProbabilisticPersistenceDiagram::kernelProbability(const Pair &pair) const
{
    const double bandwidth = model_parameters_.empty() ? 1.0 : model_parameters_.front();
    const double lifetime = detail::finiteLifetime(pair);
    double density = 0.0;
    for (const auto &source : mean_pairs_)
    {
        density += normalDensity(lifetime, detail::finiteLifetime(source), bandwidth * bandwidth);
    }
    return mean_pairs_.empty() ? 0.0 : density / static_cast<double>(mean_pairs_.size());
}

} // namespace nerve::probabilistic
