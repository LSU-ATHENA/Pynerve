#pragma once

#include <vector>

namespace nerve::probabilistic
{
struct DistributionParams
{
    double mean = 0.0;
    double variance = 1.0;
    double skewness = 0.0;
    double kurtosis = 3.0;
    bool isValid() const;
};

class ProbabilisticModel
{
public:
    explicit ProbabilisticModel(const DistributionParams &params);
    DistributionParams getParams() const;
};

class Sampler
{
public:
    explicit Sampler(uint64_t seed);
    std::vector<double> sample(size_t count, const DistributionParams &params);
};

class Statistics
{
public:
    static double mean(const std::vector<double> &data);
    static double variance(const std::vector<double> &data);
    static double waldTest(const std::vector<std::pair<int, double>> &diagram,
                           const std::vector<double> &null_hypothesis);
    static double kolmogorovSmirnovTest(const std::vector<std::pair<int, double>> &d1,
                                        const std::vector<std::pair<int, double>> &d2);
};

namespace tda_utils
{
double computePersistenceEntropy(const std::vector<std::pair<int, double>> &diagram);
std::vector<double> computeBirthDistribution(const std::vector<std::pair<int, double>> &diagram);
} // namespace tda_utils
} // namespace nerve::probabilistic
