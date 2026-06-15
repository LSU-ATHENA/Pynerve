#include "probabilistic_detail.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nerve::probabilistic
{
namespace
{
constexpr double kDegenerateWaldStatistic = 1.0e12;

std::vector<double> sortedLifetimes(const Diagram &diagram)
{
    auto values = detail::finiteLifetimes(diagram);
    std::sort(values.begin(), values.end());
    return values;
}

double pooledStd(const std::vector<double> &left, const std::vector<double> &right)
{
    const double left_var = detail::variance(left);
    const double right_var = detail::variance(right);
    const double denom = static_cast<double>(left.size() + right.size() - 2);
    if (denom <= 0.0)
    {
        return 0.0;
    }
    const double pooled = (static_cast<double>(left.size() - 1) * left_var +
                           static_cast<double>(right.size() - 1) * right_var) /
                          denom;
    return std::sqrt(std::max(0.0, pooled));
}

double finiteDiagramScore(const Diagram &diagram)
{
    const auto lifetimes = detail::finiteLifetimes(diagram);
    return -std::accumulate(lifetimes.begin(), lifetimes.end(), 0.0);
}

} // namespace

double StatisticalPersistence::permutationTest(const Diagram &diagram1, const Diagram &diagram2,
                                               Size num_permutations) const
{
    const auto sample1 = extractPersistenceValues(diagram1);
    const auto sample2 = extractPersistenceValues(diagram2);
    if (sample1.empty() || sample2.empty())
    {
        return 1.0;
    }
    const double observed = computeTestStatistic(sample1, sample2);
    std::vector<double> combined = sample1;
    combined.insert(combined.end(), sample2.begin(), sample2.end());
    std::mt19937 generator(5489u);
    Size extreme = 0;
    for (Size i = 0; i < num_permutations; ++i)
    {
        std::shuffle(combined.begin(), combined.end(), generator);
        const auto split = std::next(
            combined.begin(), static_cast<std::vector<double>::difference_type>(sample1.size()));
        std::vector<double> left(combined.begin(), split);
        std::vector<double> right(split, combined.end());
        if (computeTestStatistic(left, right) >= observed)
        {
            ++extreme;
        }
    }
    return static_cast<double>(extreme + 1) / static_cast<double>(num_permutations + 1);
}

double StatisticalPersistence::kolmogorovSmirnovTest(const Diagram &diagram1,
                                                     const Diagram &diagram2) const
{
    const auto left = sortedLifetimes(diagram1);
    const auto right = sortedLifetimes(diagram2);
    if (left.empty() || right.empty())
    {
        return 0.0;
    }
    Size i = 0;
    Size j = 0;
    double max_gap = 0.0;
    while (i < left.size() || j < right.size())
    {
        const double value =
            j == right.size() || (i < left.size() && left[i] <= right[j]) ? left[i] : right[j];
        while (i < left.size() && left[i] <= value)
        {
            ++i;
        }
        while (j < right.size() && right[j] <= value)
        {
            ++j;
        }
        const double left_cdf = static_cast<double>(i) / static_cast<double>(left.size());
        const double right_cdf = static_cast<double>(j) / static_cast<double>(right.size());
        max_gap = std::max(max_gap, std::abs(left_cdf - right_cdf));
    }
    return max_gap;
}

double StatisticalPersistence::waldTest(const Diagram &diagram,
                                        const std::vector<double> &null_hypothesis) const
{
    const auto values = extractPersistenceValues(diagram);
    if (values.empty() || null_hypothesis.empty())
    {
        return 0.0;
    }
    for (double value : null_hypothesis)
    {
        detail::requireFinite(value, "null hypothesis");
    }
    const double estimate = detail::mean(values);
    const double null_value = detail::mean(null_hypothesis);
    const double std_error =
        std::sqrt(detail::variance(values) / static_cast<double>(std::max<Size>(values.size(), 1)));
    if (std_error <= detail::kEpsilon)
    {
        return estimate == null_value ? 0.0 : kDegenerateWaldStatistic;
    }
    const double z = (estimate - null_value) / std_error;
    return z * z;
}

std::vector<std::pair<double, double>>
StatisticalPersistence::bootstrapConfidenceIntervals(const Diagram &diagram,
                                                     double confidence_level) const
{
    BootstrapPersistence bootstrap(diagram);
    bootstrap.setRandomSeed(5489u);
    bootstrap.setNumBootstrapSamples(200);
    const auto intervals = bootstrap.bootstrapPersistenceIntervals(confidence_level);
    std::vector<std::pair<double, double>> output;
    output.reserve(intervals.size());
    for (const auto &interval : intervals)
    {
        output.emplace_back(interval[0], interval[1]);
    }
    return output;
}

std::vector<std::pair<double, double>>
StatisticalPersistence::bayesianCredibleIntervals(const Diagram &diagram,
                                                  double credibility_level) const
{
    ProbabilisticPersistenceDiagram model(diagram);
    model.fitKernelModel(0.05);
    const auto samples = model.sampleDiagrams(200);
    std::vector<double> totals;
    totals.reserve(samples.size());
    for (const auto &sample : samples)
    {
        totals.push_back(detail::totalLifetime(sample));
    }
    return {{detail::quantile(totals, (1.0 - credibility_level) * 0.5),
             detail::quantile(totals, 1.0 - (1.0 - credibility_level) * 0.5)}};
}

double StatisticalPersistence::computeCohensD(const Diagram &diagram1,
                                              const Diagram &diagram2) const
{
    const auto left = extractPersistenceValues(diagram1);
    const auto right = extractPersistenceValues(diagram2);
    const double pooled = pooledStd(left, right);
    if (pooled <= detail::kEpsilon)
    {
        return 0.0;
    }
    return (detail::mean(left) - detail::mean(right)) / pooled;
}

double StatisticalPersistence::computeEffectSize(const Diagram &diagram1,
                                                 const Diagram &diagram2) const
{
    return std::abs(computeCohensD(diagram1, diagram2));
}

std::vector<double> StatisticalPersistence::extractPersistenceValues(const Diagram &diagram) const
{
    return detail::finiteLifetimes(diagram);
}

double StatisticalPersistence::computeTestStatistic(const std::vector<double> &sample1,
                                                    const std::vector<double> &sample2) const
{
    return std::abs(detail::mean(sample1) - detail::mean(sample2));
}

std::vector<double> StatisticalPersistence::permuteSample(const std::vector<double> &sample) const
{
    auto output = sample;
    std::mt19937 generator(5489u);
    std::shuffle(output.begin(), output.end(), generator);
    return output;
}

Diagram RobustPersistence::computeRobustPersistence(const std::vector<std::vector<double>> &points,
                                                    const std::string &method) const
{
    if (method != "huber" && method != "tukey")
    {
        throw std::invalid_argument("unsupported robust persistence method");
    }
    return mEstimatePersistence(points, method);
}

std::vector<bool> RobustPersistence::detectOutliers(const Diagram &diagram, double threshold) const
{
    if (!std::isfinite(threshold) || threshold < 0.0)
    {
        throw std::invalid_argument("outlier threshold must be non-negative");
    }
    const auto values = detail::finiteLifetimes(diagram);
    const double center = detail::median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values)
    {
        deviations.push_back(std::abs(value - center));
    }
    const double mad = std::max(detail::median(deviations) * 1.4826, detail::kEpsilon);
    std::vector<bool> flags;
    flags.reserve(diagram.getPairs().size());
    for (const auto &pair : diagram.getPairs())
    {
        flags.push_back(std::abs(detail::finiteLifetime(pair) - center) > threshold * mad);
    }
    return flags;
}

std::vector<Pair> RobustPersistence::removeOutliers(const Diagram &diagram, double threshold) const
{
    const auto flags = detectOutliers(diagram, threshold);
    std::vector<Pair> pairs;
    for (Size i = 0; i < diagram.getPairs().size(); ++i)
    {
        if (i >= flags.size() || !flags[i])
        {
            pairs.push_back(diagram.getPairs()[i]);
        }
    }
    return pairs;
}

std::vector<double> RobustPersistence::computeRobustMean(const Diagram &diagram) const
{
    return {detail::median(detail::finiteLifetimes(diagram))};
}

std::vector<double> RobustPersistence::computeRobustStd(const Diagram &diagram) const
{
    const auto values = detail::finiteLifetimes(diagram);
    const double center = detail::median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values)
    {
        deviations.push_back(std::abs(value - center));
    }
    return {detail::median(deviations) * 1.4826};
}

Diagram RobustPersistence::mEstimatePersistence(const std::vector<std::vector<double>> &points,
                                                const std::string &loss_function) const
{
    const Diagram diagram = detail::diagramFromPoints(points, 2);
    const auto lifetimes = detail::finiteLifetimes(diagram);
    const double center = computeMEstimator(lifetimes, loss_function);
    Diagram output;
    for (const auto &pair : diagram.getPairs())
    {
        if (std::abs(detail::finiteLifetime(pair) - center) <= std::max(center, detail::kEpsilon))
        {
            output.addPair(pair);
        }
    }
    return output.isEmpty() ? diagram : output;
}

double RobustPersistence::huberLoss(double residual, double threshold) const
{
    const double magnitude = std::abs(residual);
    return magnitude <= threshold ? 0.5 * residual * residual
                                  : threshold * (magnitude - 0.5 * threshold);
}

double RobustPersistence::tukeyLoss(double residual, double threshold) const
{
    const double ratio = residual / std::max(threshold, detail::kEpsilon);
    if (std::abs(ratio) >= 1.0)
    {
        return threshold * threshold / 6.0;
    }
    const double term = 1.0 - ratio * ratio;
    return threshold * threshold * (1.0 - term * term * term) / 6.0;
}

double RobustPersistence::computeMEstimator(const std::vector<double> &residuals,
                                            const std::string &loss_function) const
{
    if (residuals.empty())
    {
        return 0.0;
    }
    double best = residuals.front();
    double best_loss = std::numeric_limits<double>::max();
    for (double candidate : residuals)
    {
        double loss = 0.0;
        for (double value : residuals)
        {
            const double residual = value - candidate;
            loss +=
                loss_function == "tukey" ? tukeyLoss(residual, 4.685) : huberLoss(residual, 1.345);
        }
        if (loss < best_loss)
        {
            best = candidate;
            best_loss = loss;
        }
    }
    return best;
}

std::vector<std::vector<double>>
ProbabilisticTDA::samplePointsUniform(const std::vector<std::vector<double>> &points,
                                      Size sample_size)
{
    auto sampled = points;
    std::mt19937 generator(5489u);
    std::shuffle(sampled.begin(), sampled.end(), generator);
    sampled.resize(std::min(sample_size, sampled.size()));
    return sampled;
}

std::vector<std::vector<double>>
ProbabilisticTDA::samplePointsStratified(const std::vector<std::vector<double>> &points,
                                         Size sample_size, Size num_strata)
{
    if (num_strata == 0)
    {
        throw std::invalid_argument("stratum count must be positive");
    }
    auto sorted = points;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        return (!a.empty() ? a.front() : 0.0) < (!b.empty() ? b.front() : 0.0);
    });
    std::vector<std::vector<double>> output;
    for (Size stratum = 0; stratum < num_strata && output.size() < sample_size; ++stratum)
    {
        const Size begin = stratum * sorted.size() / num_strata;
        const Size end = (stratum + 1) * sorted.size() / num_strata;
        if (begin < end)
        {
            output.push_back(sorted[begin]);
        }
    }
    return output;
}

std::vector<std::vector<double>>
ProbabilisticTDA::addGaussianNoise(const std::vector<std::vector<double>> &points, double std_dev)
{
    if (!std::isfinite(std_dev) || std_dev < 0.0)
    {
        throw std::invalid_argument("gaussian noise standard deviation must be non-negative");
    }
    auto output = points;
    if (std_dev <= detail::kEpsilon)
    {
        return output;
    }
    auto generator = detail::seededGenerator(points, 41u);
    std::normal_distribution<double> noise(0.0, std_dev);
    for (auto &point : output)
    {
        for (double &value : point)
        {
            value += noise(generator);
        }
    }
    return output;
}

std::vector<std::vector<double>>
ProbabilisticTDA::addUniformNoise(const std::vector<std::vector<double>> &points, double range)
{
    if (!std::isfinite(range) || range < 0.0)
    {
        throw std::invalid_argument("uniform noise range must be non-negative");
    }
    auto generator = detail::seededGenerator(points, 43u);
    std::uniform_real_distribution<double> noise(-range, range);
    auto output = points;
    for (auto &point : output)
    {
        for (double &value : point)
        {
            value += noise(generator);
        }
    }
    return output;
}

std::vector<double> ProbabilisticTDA::computeBootstrapCi(const std::vector<double> &sample,
                                                         double confidence_level)
{
    return {
        detail::quantile(sample, (1.0 - confidence_level) * 0.5),
        detail::quantile(sample, 1.0 - (1.0 - confidence_level) * 0.5),
    };
}

double ProbabilisticTDA::computeEffectSize(const std::vector<double> &sample1,
                                           const std::vector<double> &sample2)
{
    const double pooled = pooledStd(sample1, sample2);
    return pooled <= detail::kEpsilon ? 0.0
                                      : (detail::mean(sample1) - detail::mean(sample2)) / pooled;
}

double ProbabilisticTDA::computeAic(const std::vector<Diagram> &models)
{
    const double log_likelihood =
        std::accumulate(models.begin(), models.end(), 0.0, [](double sum, const Diagram &diagram) {
            return sum + finiteDiagramScore(diagram);
        });
    return 2.0 * static_cast<double>(models.size()) - 2.0 * log_likelihood;
}

double ProbabilisticTDA::computeBic(const std::vector<Diagram> &models)
{
    const double n = static_cast<double>(std::max<Size>(models.size(), 1));
    const double log_likelihood =
        std::accumulate(models.begin(), models.end(), 0.0, [](double sum, const Diagram &diagram) {
            return sum + finiteDiagramScore(diagram);
        });
    return std::log(n) * static_cast<double>(models.size()) - 2.0 * log_likelihood;
}

double ProbabilisticTDA::computeCrossValidationScore(const std::vector<Diagram> &models,
                                                     Size k_folds)
{
    if (k_folds == 0)
    {
        throw std::invalid_argument("fold count must be positive");
    }
    if (models.empty())
    {
        return 0.0;
    }
    double score = 0.0;
    for (Size fold = 0; fold < k_folds; ++fold)
    {
        for (Size i = fold; i < models.size(); i += k_folds)
        {
            score += finiteDiagramScore(models[i]);
        }
    }
    return score / static_cast<double>(models.size());
}

} // namespace nerve::probabilistic
