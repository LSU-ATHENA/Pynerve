#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/probabilistic/detail/probabilistic_detail.hpp"
#include "nerve/probabilistic/probabilistic_models.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::Diagram;
using nerve::persistence::Pair;
using nerve::probabilistic::ProbabilisticPersistenceDiagram;
using nerve::probabilistic::ProbabilisticTDA;
using nerve::probabilistic::RobustPersistence;
using nerve::probabilistic::StatisticalPersistence;

bool check_model_construction()
{
    Diagram diagram;
    diagram.addPair(Pair{0.0, 1.0, 0});
    diagram.addPair(Pair{0.5, 2.0, 0});
    diagram.addPair(Pair{1.0, 3.0, 0});

    ProbabilisticPersistenceDiagram model(diagram);
    model.fitGaussianModel();
    auto cov = model.computeCovarianceMatrix();
    if (cov.empty())
    {
        std::cerr << "covariance matrix is empty after fitting Gaussian model\n";
        return false;
    }
    auto variances = model.computeMarginalVariances();
    if (variances.empty())
    {
        std::cerr << "marginal variances are empty\n";
        return false;
    }
    return true;
}

bool check_distribution_parameters_valid()
{
    Diagram diagram;
    diagram.addPair(Pair{0.1, 0.5, 0});
    diagram.addPair(Pair{0.2, 1.0, 0});
    diagram.addPair(Pair{0.3, 2.0, 0});
    diagram.addPair(Pair{0.4, 3.0, 0});

    ProbabilisticPersistenceDiagram model(diagram);
    model.fitMixtureModel(2);
    auto cov = model.computeCovarianceMatrix();
    for (const auto &row : cov)
    {
        for (double v : row)
        {
            if (!std::isfinite(v))
            {
                std::cerr << "covariance matrix contains non-finite values\n";
                return false;
            }
        }
    }
    auto variances = model.computeMarginalVariances();
    for (double v : variances)
    {
        if (v < 0.0)
        {
            std::cerr << "variance is negative: " << v << "\n";
            return false;
        }
    }
    return true;
}

bool check_sampling_produces_finite_values()
{
    Diagram diagram;
    diagram.addPair(Pair{0.0, 1.0, 0});
    diagram.addPair(Pair{0.2, 1.5, 0});
    diagram.addPair(Pair{0.5, 2.0, 0});

    const double bandwidth = 0.1;
    ProbabilisticPersistenceDiagram model(diagram);
    model.fitKernelModel(bandwidth);
    auto samples = model.sampleDiagrams(10);
    if (samples.empty())
    {
        std::cerr << "sampling produced no diagrams\n";
        return false;
    }
    for (const auto &sample : samples)
    {
        for (const auto &pair : sample.getPairs())
        {
            if (!std::isfinite(pair.birth) || !std::isfinite(pair.death))
            {
                std::cerr << "sampled pair contains non-finite values\n";
                return false;
            }
        }
    }
    return true;
}

bool check_basic_statistics_match_known()
{
    Diagram diagram;
    diagram.addPair(Pair{0.0, 2.0, 0});
    diagram.addPair(Pair{1.0, 3.0, 0});
    diagram.addPair(Pair{2.0, 4.0, 0});
    diagram.addPair(Pair{3.0, 7.0, 0});

    StatisticalPersistence stats;
    std::vector<double> null_hyp = {2.0};
    double wald = stats.waldTest(diagram, null_hyp);
    if (!std::isfinite(wald))
    {
        std::cerr << "Wald statistic is not finite\n";
        return false;
    }
    if (wald < 0.0)
    {
        std::cerr << "Wald statistic is negative\n";
        return false;
    }

    double cd = stats.computeCohensD(diagram, diagram);
    if (std::abs(cd) > 1e-10)
    {
        std::cerr << "Cohen's d between identical diagrams should be zero\n";
        return false;
    }

    double ks = stats.kolmogorovSmirnovTest(diagram, diagram);
    if (std::abs(ks) > 1e-10)
    {
        std::cerr << "KS statistic between identical diagrams should be zero\n";
        return false;
    }
    return true;
}

bool check_probabilistic_tda_utils()
{
    std::vector<std::vector<double>> points = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    auto sampled = ProbabilisticTDA::samplePointsUniform(points, 2);
    if (sampled.size() != 2)
    {
        std::cerr << "uniform sampling did not return requested count\n";
        return false;
    }
    auto noisy = ProbabilisticTDA::addGaussianNoise(points, 0.1);
    if (noisy.size() != points.size())
    {
        std::cerr << "Gaussian noise changed point count\n";
        return false;
    }
    for (const auto &pt : noisy)
    {
        for (double v : pt)
        {
            if (!std::isfinite(v))
            {
                std::cerr << "noisy point contains non-finite value\n";
                return false;
            }
        }
    }
    std::vector<double> ci = ProbabilisticTDA::computeBootstrapCi({1.0, 2.0, 3.0, 4.0, 5.0}, 0.95);
    if (ci.size() != 2 || !std::isfinite(ci[0]) || !std::isfinite(ci[1]))
    {
        std::cerr << "bootstrap CI produced invalid result\n";
        return false;
    }
    if (ci[0] > ci[1])
    {
        std::cerr << "bootstrap CI lower bound exceeds upper bound\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_model_construction())
    {
        std::cerr << "FAIL: model construction\n";
        return 1;
    }
    if (!check_distribution_parameters_valid())
    {
        std::cerr << "FAIL: distribution parameters valid\n";
        return 1;
    }
    if (!check_sampling_produces_finite_values())
    {
        std::cerr << "FAIL: sampling produces finite values\n";
        return 1;
    }
    if (!check_basic_statistics_match_known())
    {
        std::cerr << "FAIL: basic statistics match known\n";
        return 1;
    }
    if (!check_probabilistic_tda_utils())
    {
        std::cerr << "FAIL: probabilistic TDA utils\n";
        return 1;
    }
    return 0;
}
