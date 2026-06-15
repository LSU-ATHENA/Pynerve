#include "probabilistic_detail.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nerve::probabilistic
{
namespace
{
constexpr double kLogPriorImpossible = -1.0e300;

std::vector<std::vector<double>> perturbPoints(const std::vector<std::vector<double>> &points,
                                               double scale, std::mt19937 &generator)
{
    detail::pointDimension(points);
    auto perturbed = points;
    if (scale <= detail::kEpsilon)
    {
        return perturbed;
    }
    std::normal_distribution<double> noise(0.0, scale);
    for (auto &point : perturbed)
    {
        for (double &value : point)
        {
            value += noise(generator);
        }
    }
    return perturbed;
}

Diagram averageByIndex(const std::vector<Diagram> &diagrams)
{
    Diagram output;
    if (diagrams.empty())
    {
        return output;
    }
    Size pair_count = diagrams.front().getPairs().size();
    for (const auto &diagram : diagrams)
    {
        pair_count = std::min(pair_count, diagram.getPairs().size());
    }
    for (Size i = 0; i < pair_count; ++i)
    {
        Pair pair{};
        pair.dimension = diagrams.front().getPairs()[i].dimension;
        for (const auto &diagram : diagrams)
        {
            const Pair &source = diagram.getPairs()[i];
            detail::requireValidPair(source, "persistence pair");
            pair.birth += source.birth;
            pair.death += source.isInfinite() ? source.birth : source.death;
        }
        pair.birth /= static_cast<double>(diagrams.size());
        pair.death /= static_cast<double>(diagrams.size());
        if (pair.death < pair.birth)
        {
            std::swap(pair.birth, pair.death);
        }
        output.addPair(pair);
    }
    return output;
}

} // namespace

std::vector<Diagram>
MonteCarloPersistence::monteCarloSample(const std::vector<std::vector<double>> &points,
                                        Size num_samples, const std::string &sampling_method) const
{
    if (num_samples == 0)
    {
        throw std::invalid_argument("sample count must be positive");
    }
    detail::pointDimension(points);
    auto generator = detail::seededGenerator(points, 17u);
    std::vector<Diagram> samples;
    samples.reserve(num_samples);
    const double scale = detail::maxPairwiseDistance(points) * 0.02;
    for (Size i = 0; i < num_samples; ++i)
    {
        if (sampling_method == "uniform")
        {
            samples.push_back(detail::diagramFromPoints(points, 2));
        }
        else if (sampling_method == "gaussian")
        {
            samples.push_back(
                detail::diagramFromPoints(perturbPoints(points, scale, generator), 2));
        }
        else
        {
            throw std::invalid_argument("unsupported sampling method");
        }
    }
    return samples;
}

std::vector<Diagram>
MonteCarloPersistence::mcmcSample(const std::vector<std::vector<double>> &points, Size num_samples,
                                  double step_size) const
{
    if (num_samples == 0)
    {
        throw std::invalid_argument("sample count must be positive");
    }
    if (!std::isfinite(step_size) || step_size < 0.0)
    {
        throw std::invalid_argument("step size must be finite and non-negative");
    }
    detail::pointDimension(points);
    random_generator_ = detail::seededGenerator(points, 23u);
    Diagram current = detail::diagramFromPoints(points, 2);
    std::uniform_real_distribution<double> accept(0.0, 1.0);
    std::vector<Diagram> samples;
    samples.reserve(num_samples);

    for (Size i = 0; i < burn_in_ + num_samples; ++i)
    {
        Diagram proposal = proposeSample(current);
        const double current_target = computeTargetProbability(current);
        const double proposal_target = computeTargetProbability(proposal);
        const double ratio =
            current_target <= detail::kEpsilon ? 1.0 : proposal_target / current_target;
        if (accept(random_generator_) <= std::min(1.0, ratio))
        {
            current = proposal;
        }
        if (i >= burn_in_)
        {
            samples.push_back(current);
        }
    }
    return samples;
}

std::vector<Diagram>
MonteCarloPersistence::importanceSample(const std::vector<std::vector<double>> &points,
                                        Size num_samples,
                                        const std::vector<double> &proposal_weights) const
{
    if (num_samples == 0)
    {
        throw std::invalid_argument("sample count must be positive");
    }
    if (proposal_weights.size() != points.size())
    {
        throw std::invalid_argument("proposal weights must match point count");
    }
    double weight_sum = 0.0;
    for (double weight : proposal_weights)
    {
        if (!std::isfinite(weight) || weight < 0.0)
        {
            throw std::invalid_argument("proposal weights must be finite and non-negative");
        }
        weight_sum += weight;
    }
    if (weight_sum <= detail::kEpsilon)
    {
        throw std::invalid_argument("proposal weights must contain positive mass");
    }
    detail::pointDimension(points);
    auto generator = detail::seededGenerator(points, 29u);
    std::discrete_distribution<Size> pick(proposal_weights.begin(), proposal_weights.end());
    std::vector<Diagram> samples;
    samples.reserve(num_samples);
    for (Size i = 0; i < num_samples; ++i)
    {
        std::vector<std::vector<double>> selected;
        selected.reserve(points.size());
        for (Size j = 0; j < points.size(); ++j)
        {
            selected.push_back(points[pick(generator)]);
        }
        samples.push_back(detail::diagramFromPoints(selected, 2));
    }
    return samples;
}

double MonteCarloPersistence::computeGelmanRubinStatistic(const std::vector<Diagram> &chains) const
{
    if (chains.size() < 2)
    {
        return 1.0;
    }
    std::vector<double> totals;
    totals.reserve(chains.size());
    for (const auto &chain : chains)
    {
        totals.push_back(detail::totalLifetime(chain));
    }
    const double between = detail::variance(totals);
    const double within = std::max(detail::mean(totals), detail::kEpsilon);
    return std::sqrt(std::max(1.0, between / within));
}

double MonteCarloPersistence::computeEffectiveSampleSize(const std::vector<Diagram> &samples) const
{
    if (samples.empty())
    {
        return 0.0;
    }
    std::vector<double> totals;
    totals.reserve(samples.size());
    for (const auto &sample : samples)
    {
        totals.push_back(detail::totalLifetime(sample));
    }
    const double var = detail::variance(totals);
    if (var <= detail::kEpsilon || samples.size() < 3)
    {
        return static_cast<double>(samples.size());
    }
    double rho_sum = 0.0;
    const double avg = detail::mean(totals);
    for (Size lag = 1; lag < samples.size(); ++lag)
    {
        double covariance = 0.0;
        for (Size i = 0; i + lag < samples.size(); ++i)
        {
            covariance += (totals[i] - avg) * (totals[i + lag] - avg);
        }
        const double rho = covariance / (static_cast<double>(samples.size() - lag) * var);
        if (rho <= 0.0)
        {
            break;
        }
        rho_sum += rho;
    }
    return static_cast<double>(samples.size()) / (1.0 + 2.0 * rho_sum);
}

void MonteCarloPersistence::setSamplingMethod(const std::string &method)
{
    if (method != "uniform" && method != "gaussian" && method != "mcmc")
    {
        throw std::invalid_argument("unsupported sampling method");
    }
    sampling_method_ = method;
}

void MonteCarloPersistence::setStepSize(double step_size)
{
    if (!std::isfinite(step_size) || step_size < 0.0)
    {
        throw std::invalid_argument("step size must be finite and non-negative");
    }
    step_size_ = step_size;
}

void MonteCarloPersistence::setBurnIn(Size burn_in)
{
    burn_in_ = burn_in;
}

Diagram MonteCarloPersistence::proposeSample(const Diagram &current) const
{
    Diagram proposal;
    if (step_size_ <= detail::kEpsilon)
    {
        for (const auto &pair : current.getPairs())
        {
            proposal.addPair(pair);
        }
        return proposal;
    }
    std::normal_distribution<double> noise(0.0, step_size_);
    for (auto pair : current.getPairs())
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
        proposal.addPair(pair);
    }
    return proposal;
}

double MonteCarloPersistence::computeProposalProbability(const Diagram &from,
                                                         const Diagram &to) const
{
    const Size count = std::min(from.getPairs().size(), to.getPairs().size());
    if (count == 0)
    {
        return 0.0;
    }
    double squared = 0.0;
    for (Size i = 0; i < count; ++i)
    {
        const double delta =
            detail::finiteLifetime(from.getPairs()[i]) - detail::finiteLifetime(to.getPairs()[i]);
        squared += delta * delta;
    }
    const double variance = std::max(step_size_ * step_size_, detail::kEpsilon);
    return std::exp(-squared / (2.0 * variance));
}

double MonteCarloPersistence::computeTargetProbability(const Diagram &diagram) const
{
    return std::exp(-detail::totalLifetime(diagram));
}

void BayesianPersistence::setGaussianPrior(const std::vector<double> &mean,
                                           const std::vector<double> &std)
{
    if (mean.size() != std.size())
    {
        throw std::invalid_argument("gaussian prior mean and std sizes must match");
    }
    prior_type_ = "gaussian";
    prior_parameters_ = mean;
    prior_parameters_.insert(prior_parameters_.end(), std.begin(), std.end());
}

void BayesianPersistence::setUniformPrior(const std::vector<double> &lower,
                                          const std::vector<double> &upper)
{
    if (lower.size() != upper.size())
    {
        throw std::invalid_argument("uniform prior bound sizes must match");
    }
    prior_type_ = "uniform";
    prior_parameters_ = lower;
    prior_parameters_.insert(prior_parameters_.end(), upper.begin(), upper.end());
}

void BayesianPersistence::setInformativePrior(const std::vector<std::vector<double>> &prior_samples)
{
    if (prior_samples.empty())
    {
        throw std::invalid_argument("informative prior samples must not be empty");
    }
    prior_type_ = "informative";
    prior_parameters_.clear();
    for (const auto &sample : prior_samples)
    {
        for (double value : sample)
        {
            detail::requireFinite(value, "informative prior");
            prior_parameters_.push_back(value);
        }
    }
}

std::vector<Diagram>
BayesianPersistence::samplePosterior(const std::vector<std::vector<double>> &data, Size num_samples,
                                     const std::string &sampler) const
{
    MonteCarloPersistence mc;
    std::vector<Diagram> proposals = sampler == "mcmc"
                                         ? mc.mcmcSample(data, num_samples, 0.05)
                                         : mc.monteCarloSample(data, num_samples, "gaussian");
    std::sort(proposals.begin(), proposals.end(), [&](const Diagram &a, const Diagram &b) {
        return computeLogPrior(a) + computeLogLikelihood(a, data) >
               computeLogPrior(b) + computeLogLikelihood(b, data);
    });
    posterior_samples_ = proposals;
    return posterior_samples_;
}

double
BayesianPersistence::computeMarginalLikelihood(const std::vector<std::vector<double>> &data) const
{
    const Diagram observed = detail::diagramFromPoints(data, 2);
    return std::exp(computeLogPrior(observed) + computeLogLikelihood(observed, data));
}

double BayesianPersistence::computeBayesFactor(const std::vector<std::vector<double>> &data1,
                                               const std::vector<std::vector<double>> &data2) const
{
    const double denominator = std::max(computeMarginalLikelihood(data2), detail::kEpsilon);
    return computeMarginalLikelihood(data1) / denominator;
}

Diagram BayesianPersistence::computePosteriorMean() const
{
    return averageByIndex(posterior_samples_);
}

std::vector<std::vector<double>> BayesianPersistence::computePosteriorCovariance() const
{
    std::vector<double> totals;
    totals.reserve(posterior_samples_.size());
    for (const auto &sample : posterior_samples_)
    {
        totals.push_back(detail::totalLifetime(sample));
    }
    return {{detail::variance(totals)}};
}

std::vector<std::pair<double, double>>
BayesianPersistence::computeCredibleIntervals(double level) const
{
    detail::confidenceZ(level);
    std::vector<double> totals;
    totals.reserve(posterior_samples_.size());
    for (const auto &sample : posterior_samples_)
    {
        totals.push_back(detail::totalLifetime(sample));
    }
    return {{detail::quantile(totals, (1.0 - level) * 0.5),
             detail::quantile(totals, 1.0 - (1.0 - level) * 0.5)}};
}

double BayesianPersistence::computeLogPrior(const Diagram &diagram) const
{
    const double total = detail::totalLifetime(diagram);
    if (prior_type_ == "gaussian" && prior_parameters_.size() >= 2)
    {
        const auto half =
            static_cast<std::vector<double>::difference_type>(prior_parameters_.size() / 2);
        const auto midpoint = std::next(prior_parameters_.begin(), half);
        const std::vector<double> means(prior_parameters_.begin(), midpoint);
        const std::vector<double> stddevs(midpoint, prior_parameters_.end());
        const double center = detail::mean(means);
        const double stddev = std::max(detail::mean(stddevs), detail::kEpsilon);
        const double z = (total - center) / stddev;
        return -0.5 * z * z;
    }
    if (prior_type_ == "uniform" && prior_parameters_.size() >= 2)
    {
        const auto half =
            static_cast<std::vector<double>::difference_type>(prior_parameters_.size() / 2);
        const auto midpoint = std::next(prior_parameters_.begin(), half);
        const double lower = *std::min_element(prior_parameters_.begin(), midpoint);
        const double upper = *std::max_element(midpoint, prior_parameters_.end());
        return (total >= lower && total <= upper) ? 0.0 : kLogPriorImpossible;
    }
    return -std::log1p(total);
}

double BayesianPersistence::computeLogLikelihood(const Diagram &diagram,
                                                 const std::vector<std::vector<double>> &data) const
{
    const double observed = detail::totalLifetime(detail::diagramFromPoints(data, 2));
    const double residual = detail::totalLifetime(diagram) - observed;
    return -0.5 * residual * residual;
}

Diagram BayesianPersistence::mcmcStep(const Diagram &current,
                                      const std::vector<std::vector<double>> &data) const
{
    Diagram proposal;
    std::normal_distribution<double> noise(0.0, 0.05);
    for (auto pair : current.getPairs())
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
        proposal.addPair(pair);
    }
    const double current_score = computeLogPrior(current) + computeLogLikelihood(current, data);
    const double proposal_score = computeLogPrior(proposal) + computeLogLikelihood(proposal, data);
    return proposal_score >= current_score ? proposal : current;
}

} // namespace nerve::probabilistic
