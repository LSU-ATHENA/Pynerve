#include "nerve/core_types.hpp"
#include "nerve/probabilistic/probabilistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::Diagram;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

bool check_probabilistic_model_construction()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitGaussianModel();
    double prob = model.computeProbability(Pair{0.0, 1.0, 0});
    if (prob <= 0.0 || prob > 1.0)
    {
        std::cerr << "gaussian probability out of range: " << prob << "\n";
        return false;
    }
    return true;
}

bool check_probabilistic_model_mixture()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}, {0.5, 2.0, 1}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitMixtureModel(2);
    double prob = model.computeProbability(Pair{0.25, 0.85, 0});
    if (prob <= 0.0 || prob > 1.0)
    {
        std::cerr << "mixture probability out of range: " << prob << "\n";
        return false;
    }
    return true;
}

bool check_probabilistic_model_kernel()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitKernelModel(0.5);
    double prob = model.computeProbability(Pair{0.0, 1.0, 0});
    if (prob <= 0.0)
    {
        std::cerr << "kernel probability should be positive: " << prob << "\n";
        return false;
    }
    return true;
}

bool check_probabilistic_model_sampling()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitGaussianModel();
    auto sample = model.sampleDiagram();
    if (sample.count() == 0)
    {
        std::cerr << "sampled diagram empty\n";
        return false;
    }
    auto samples = model.sampleDiagrams(5);
    if (samples.size() != 5)
    {
        std::cerr << "expected 5 samples, got " << samples.size() << "\n";
        return false;
    }
    return true;
}

bool check_probabilistic_model_likelihood()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitGaussianModel();
    double ll = model.computeLikelihood(d);
    if (!std::isfinite(ll))
    {
        std::cerr << "log-likelihood non-finite\n";
        return false;
    }
    return true;
}

bool check_probabilistic_model_covariance()
{
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram model(d);
    model.fitGaussianModel();
    auto cov = model.computeCovarianceMatrix();
    if (cov.empty())
    {
        std::cerr << "covariance matrix empty\n";
        return false;
    }
    auto vars = model.computeMarginalVariances();
    if (vars.empty())
    {
        std::cerr << "marginal variances empty\n";
        return false;
    }
    return true;
}

bool check_monte_carlo_uniform()
{
    nerve::probabilistic::MonteCarloPersistence mc;
    std::vector<std::vector<double>> points{{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    auto samples = mc.monteCarloSample(points, 3, "uniform");
    if (samples.empty())
    {
        std::cerr << "MC samples empty\n";
        return false;
    }
    return true;
}

bool check_bayesian_persistence()
{
    nerve::probabilistic::BayesianPersistence bp;
    std::vector<double> lower{0.0, 0.0};
    std::vector<double> upper{1.0, 1.0};
    bp.setUniformPrior(lower, upper);
    std::vector<std::vector<double>> data{{0.0, 0.0}, {0.5, 0.5}, {1.0, 0.0}};
    auto posterior = bp.samplePosterior(data, 3, "mcmc");
    if (!posterior.empty())
    {
        auto post_mean = bp.computePosteriorMean();
        (void)post_mean;
    }
    return true;
}

bool check_noise_aware_persistence()
{
    nerve::probabilistic::NoiseAwarePersistence nap;
    std::vector<std::vector<double>> points{{0.0, 0.0}, {1.0, 0.0}};
    nap.addGaussianNoise(points, 0.01);
    nap.setNoiseModel("gaussian");
    nap.setRobustnessMethod("median_diagram");
    return true;
}

bool check_robust_persistence()
{
    nerve::probabilistic::RobustPersistence rp;
    std::vector<std::vector<double>> points{{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    Diagram result = rp.computeRobustPersistence(points, "huber");
    (void)result;
    Diagram d = make_diagram({{0.0, 1.0, 0}, {0.2, 0.7, 0}});
    auto outliers = rp.detectOutliers(d, 2.0);
    if (outliers.size() != d.count())
    {
        std::cerr << "outlier detection size mismatch\n";
        return false;
    }
    return true;
}

bool check_scalar_vs_model_equivalence()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 1.0, 0}});
    nerve::probabilistic::ProbabilisticPersistenceDiagram m1(d1);
    nerve::probabilistic::ProbabilisticPersistenceDiagram m2(d2);
    m1.fitGaussianModel();
    m2.fitGaussianModel();
    double p1 = m1.computeProbability(Pair{0.0, 1.0, 0});
    double p2 = m2.computeProbability(Pair{0.0, 1.0, 0});
    if (std::abs(p1 - p2) > kTol)
    {
        std::cerr << "identical models gave different probabilities\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_probabilistic_model_construction())
    {
        std::cerr << "FAIL: model construction\n";
        return 1;
    }
    if (!check_probabilistic_model_mixture())
    {
        std::cerr << "FAIL: mixture model\n";
        return 1;
    }
    if (!check_probabilistic_model_kernel())
    {
        std::cerr << "FAIL: kernel model\n";
        return 1;
    }
    if (!check_probabilistic_model_sampling())
    {
        std::cerr << "FAIL: model sampling\n";
        return 1;
    }
    if (!check_probabilistic_model_likelihood())
    {
        std::cerr << "FAIL: model likelihood\n";
        return 1;
    }
    if (!check_probabilistic_model_covariance())
    {
        std::cerr << "FAIL: model covariance\n";
        return 1;
    }
    if (!check_monte_carlo_uniform())
    {
        std::cerr << "FAIL: Monte Carlo\n";
        return 1;
    }
    if (!check_bayesian_persistence())
    {
        std::cerr << "FAIL: Bayesian\n";
        return 1;
    }
    if (!check_noise_aware_persistence())
    {
        std::cerr << "FAIL: noise aware\n";
        return 1;
    }
    if (!check_robust_persistence())
    {
        std::cerr << "FAIL: robust\n";
        return 1;
    }
    if (!check_scalar_vs_model_equivalence())
    {
        std::cerr << "FAIL: scalar equivalence\n";
        return 1;
    }
    return 0;
}
