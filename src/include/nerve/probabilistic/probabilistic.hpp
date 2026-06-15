
#pragma once
#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <complex>
#include <memory>
#include <random>
#include <string>
#include <vector>
namespace nerve::probabilistic
{
using algebra::Simplex;
using persistence::Diagram;
using persistence::Pair;
class BootstrapPersistence
{
public:
    BootstrapPersistence() = default;
    explicit BootstrapPersistence(const Diagram &diagram);
    std::vector<Diagram> bootstrapResample(Size num_samples) const;
    std::vector<std::vector<double>>
    bootstrapPersistenceIntervals(double confidence_level = 0.95) const;
    std::vector<double> computeMeanPersistence() const;
    std::vector<double> computeStdPersistence() const;
    std::vector<double> computeConfidenceIntervals(double confidence_level = 0.95) const;
    void setRandomSeed(unsigned int seed);
    void setBootstrapMethod(const std::string &method);
    void setNumBootstrapSamples(Size num_samples);

private:
    Diagram original_diagram_;
    mutable std::mt19937 random_generator_{5489u};
    std::string bootstrap_method_ = "with_replacement";
    Size num_bootstrap_samples_ = 1000;
    Diagram resampleWithReplacement() const;
    Diagram resampleWithoutReplacement() const;
    Diagram resampleGaussianNoise(double noise_level = 0.1) const;
    std::vector<Pair> samplePairsWithReplacement() const;
};
class NoiseAwarePersistence
{
public:
    NoiseAwarePersistence() = default;
    void addGaussianNoise(const std::vector<std::vector<double>> &points, double std_dev);
    void addUniformNoise(const std::vector<std::vector<double>> &points, double range);
    void addPoissonNoise(const std::vector<std::vector<double>> &points, double lambda);
    Diagram computeRobustPersistence(const std::vector<std::vector<double>> &points,
                                     Size num_bootstrap_samples = 100) const;
    std::vector<std::vector<double>> computePersistenceUncertainty(const Diagram &diagram,
                                                                   Size num_samples = 100) const;
    std::vector<std::pair<double, double>>
    computeConfidenceBands(const Diagram &diagram, double confidence_level = 0.95) const;
    void setNoiseModel(const std::string &model);
    void setRobustnessMethod(const std::string &method);

private:
    std::string noise_model_ = "gaussian";
    std::string robustness_method_ = "median_diagram";
    double noise_parameter_ = 0.1;
    mutable std::mt19937 random_generator_{5489u};
    std::vector<std::vector<double>>
    applyNoiseModel(const std::vector<std::vector<double>> &points) const;
    std::vector<Diagram> computePersistenceEnsemble(const std::vector<std::vector<double>> &points,
                                                    Size num_samples) const;
};
class ProbabilisticPersistenceDiagram
{
public:
    ProbabilisticPersistenceDiagram() = default;
    explicit ProbabilisticPersistenceDiagram(const Diagram &diagram);
    void fitGaussianModel();
    void fitMixtureModel(Size num_components);
    void fitKernelModel(double bandwidth);
    Diagram sampleDiagram() const;
    std::vector<Diagram> sampleDiagrams(Size num_samples) const;
    double computeProbability(const Pair &pair) const;
    double computeLikelihood(const Diagram &diagram) const;
    std::vector<std::vector<double>> computeCovarianceMatrix() const;
    std::vector<double> computeMarginalVariances() const;

private:
    Diagram original_diagram_;
    std::vector<Pair> mean_pairs_;
    std::vector<std::vector<double>> covariance_matrix_;
    std::string model_type_ = "empirical";
    std::vector<double> model_parameters_;
    mutable std::mt19937 random_generator_{5489u};
    double gaussianProbability(const Pair &pair) const;
    double mixtureProbability(const Pair &pair) const;
    double kernelProbability(const Pair &pair) const;
};
class MonteCarloPersistence
{
public:
    MonteCarloPersistence() = default;
    std::vector<Diagram> monteCarloSample(const std::vector<std::vector<double>> &points,
                                          Size num_samples,
                                          const std::string &sampling_method = "uniform") const;
    std::vector<Diagram> mcmcSample(const std::vector<std::vector<double>> &points,
                                    Size num_samples, double step_size = 0.1) const;
    std::vector<Diagram> importanceSample(const std::vector<std::vector<double>> &points,
                                          Size num_samples,
                                          const std::vector<double> &proposal_weights) const;
    double computeGelmanRubinStatistic(const std::vector<Diagram> &chains) const;
    double computeEffectiveSampleSize(const std::vector<Diagram> &samples) const;
    void setSamplingMethod(const std::string &method);
    void setStepSize(double step_size);
    void setBurnIn(Size burn_in);

private:
    std::string sampling_method_ = "uniform";
    double step_size_ = 0.1;
    Size burn_in_ = 0;
    mutable std::mt19937 random_generator_{5489u};
    Diagram proposeSample(const Diagram &current) const;
    double computeProposalProbability(const Diagram &from, const Diagram &to) const;
    double computeTargetProbability(const Diagram &diagram) const;
};
class BayesianPersistence
{
public:
    BayesianPersistence() = default;
    void setGaussianPrior(const std::vector<double> &mean, const std::vector<double> &std);
    void setUniformPrior(const std::vector<double> &lower, const std::vector<double> &upper);
    void setInformativePrior(const std::vector<std::vector<double>> &prior_samples);
    std::vector<Diagram> samplePosterior(const std::vector<std::vector<double>> &data,
                                         Size num_samples,
                                         const std::string &sampler = "mcmc") const;
    double computeMarginalLikelihood(const std::vector<std::vector<double>> &data) const;
    double computeBayesFactor(const std::vector<std::vector<double>> &data1,
                              const std::vector<std::vector<double>> &data2) const;
    Diagram computePosteriorMean() const;
    std::vector<std::vector<double>> computePosteriorCovariance() const;
    std::vector<std::pair<double, double>> computeCredibleIntervals(double level = 0.95) const;

private:
    std::string prior_type_ = "uniform";
    std::vector<double> prior_parameters_;
    mutable std::vector<Diagram> posterior_samples_;
    mutable std::mt19937 random_generator_{5489u};
    double computeLogPrior(const Diagram &diagram) const;
    double computeLogLikelihood(const Diagram &diagram,
                                const std::vector<std::vector<double>> &data) const;
    Diagram mcmcStep(const Diagram &current, const std::vector<std::vector<double>> &data) const;
};
class StatisticalPersistence
{
public:
    StatisticalPersistence() = default;
    double permutationTest(const Diagram &diagram1, const Diagram &diagram2,
                           Size num_permutations = 1000) const;
    double kolmogorovSmirnovTest(const Diagram &diagram1, const Diagram &diagram2) const;
    double waldTest(const Diagram &diagram, const std::vector<double> &null_hypothesis) const;
    std::vector<std::pair<double, double>>
    bootstrapConfidenceIntervals(const Diagram &diagram, double confidence_level = 0.95) const;
    std::vector<std::pair<double, double>>
    bayesianCredibleIntervals(const Diagram &diagram, double credibility_level = 0.95) const;
    double computeCohensD(const Diagram &diagram1, const Diagram &diagram2) const;
    double computeEffectSize(const Diagram &diagram1, const Diagram &diagram2) const;

private:
    std::vector<double> extractPersistenceValues(const Diagram &diagram) const;
    double computeTestStatistic(const std::vector<double> &sample1,
                                const std::vector<double> &sample2) const;
    std::vector<double> permuteSample(const std::vector<double> &sample) const;
};
class RobustPersistence
{
public:
    RobustPersistence() = default;
    Diagram computeRobustPersistence(const std::vector<std::vector<double>> &points,
                                     const std::string &method = "huber") const;
    std::vector<bool> detectOutliers(const Diagram &diagram, double threshold = 2.0) const;
    std::vector<Pair> removeOutliers(const Diagram &diagram, double threshold = 2.0) const;
    std::vector<double> computeRobustMean(const Diagram &diagram) const;
    std::vector<double> computeRobustStd(const Diagram &diagram) const;
    Diagram mEstimatePersistence(const std::vector<std::vector<double>> &points,
                                 const std::string &loss_function = "huber") const;

private:
    double huberLoss(double residual, double threshold) const;
    double tukeyLoss(double residual, double threshold) const;
    double computeMEstimator(const std::vector<double> &residuals,
                             const std::string &loss_function) const;
};
class ProbabilisticTDA
{
public:
    static std::vector<std::vector<double>>
    samplePointsUniform(const std::vector<std::vector<double>> &points, Size sample_size);
    static std::vector<std::vector<double>>
    samplePointsStratified(const std::vector<std::vector<double>> &points, Size sample_size,
                           Size num_strata);
    static std::vector<std::vector<double>>
    addGaussianNoise(const std::vector<std::vector<double>> &points, double std_dev);
    static std::vector<std::vector<double>>
    addUniformNoise(const std::vector<std::vector<double>> &points, double range);
    static std::vector<double> computeBootstrapCi(const std::vector<double> &sample,
                                                  double confidence_level = 0.95);
    static double computeEffectSize(const std::vector<double> &sample1,
                                    const std::vector<double> &sample2);
    static double computeAic(const std::vector<Diagram> &models);
    static double computeBic(const std::vector<Diagram> &models);
    static double computeCrossValidationScore(const std::vector<Diagram> &models, Size k_folds = 5);
};
} // namespace nerve::probabilistic
