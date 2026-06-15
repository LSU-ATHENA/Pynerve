# Probabilistic

## Quick start

```python
from pynerve.probabilistic import sample_homology
import numpy as np

# Monte Carlo sampling of persistent homology
points = np.random.randn(100, 3)

# Sample persistence diagrams under noise
diagrams = sample_homology(points, method="monte_carlo",
                           num_samples=100, noise_std=0.05)
# -> list of 100 diagrams, each a list of (birth, death) pairs
```

Monte Carlo sampling for persistent homology under noise, bootstrap-based
confidence intervals, Bayesian persistence models, and noise-aware persistence
computation. GPU random number generation (cuRAND Philox) and SIMD probability
density functions (AVX-512 exp, log) accelerate sampling.


## API

```python
def sample_homology(points, method="monte_carlo", num_samples=100,
                    noise_std=0.05, seed=42) -> list[PersistenceDiagram]: ...
```

```cpp
#include <nerve/probabilistic/sampling.hpp>
#include <nerve/probabilistic/bootstrap.hpp>
#include <nerve/probabilistic/bayesian.hpp>

namespace nerve::probabilistic {

// Monte Carlo sampling
class MonteCarloPersistence {
    std::vector<Diagram> monteCarloSample(
        const std::vector<std::vector<double>>& points, Size n_samples,
        const std::string& noise_model = "uniform") const;
    std::vector<Diagram> mcmcSample(
        const std::vector<std::vector<double>>& points, Size n_samples,
        double step_size = 0.1) const;
    std::vector<Diagram> importanceSample(
        const std::vector<std::vector<double>>& points, Size n_samples,
        const std::vector<double>& importance_weights) const;
    double computeGelmanRubinStatistic(const std::vector<Diagram>&) const;
    double computeEffectiveSampleSize(const std::vector<Diagram>&) const;
    void setSamplingMethod(const std::string&);
    void setBurnIn(Size n);
};

// Bootstrap persistence
class BootstrapPersistence {
    explicit BootstrapPersistence(const Diagram& dgm);
    std::vector<Diagram> bootstrapResample(Size n_samples) const;
    std::vector<std::vector<double>> bootstrapPersistenceIntervals(
        double confidence_level = 0.95) const;
    std::vector<double> computeMeanPersistence() const;
    std::vector<double> computeStdPersistence() const;
    void setRandomSeed(unsigned int);
    void setNumBootstrapSamples(Size n);
};

// Noisy observation model
class NoiseAwarePersistence {
    void addGaussianNoise(const std::vector<std::vector<double>>&, double std);
    void addUniformNoise(const std::vector<std::vector<double>>&, double range);
    void addPoissonNoise(const std::vector<std::vector<double>>&, double lambda);
    Diagram computeRobustPersistence(
        const std::vector<std::vector<double>>&, Size n_iter = 100) const;
    std::vector<std::vector<double>> computePersistenceUncertainty(
        const Diagram&, Size n_iter = 100) const;
};

// Probabilistic persistence diagram (generative model)
class ProbabilisticPersistenceDiagram {
    explicit ProbabilisticPersistenceDiagram(const Diagram&);
    void fitGaussianModel();
    void fitMixtureModel(Size n_components);
    void fitKernelModel(double bandwidth);
    Diagram sampleDiagram() const;
    std::vector<Diagram> sampleDiagrams(Size n) const;
    double computeProbability(const Pair&) const;
};

// Bayesian persistence
class BayesianPersistence {
    void setGaussianPrior(const std::vector<double>& mean,
                          const std::vector<double>& std);
    void setUniformPrior(const std::vector<double>& lo,
                         const std::vector<double>& hi);
    std::vector<Diagram> samplePosterior(
        const std::vector<std::vector<double>>& points, Size n_samples,
        const std::string& method = "mcmc") const;
    double computeMarginalLikelihood(
        const std::vector<std::vector<double>>& points) const;
    double computeBayesFactor(
        const std::vector<std::vector<double>>& model_a,
        const std::vector<std::vector<double>>& model_b) const;
    Diagram computePosteriorMean() const;
    std::vector<std::pair<double, double>> computeCredibleIntervals(
        double level = 0.95) const;
};

// Statistical hypothesis tests for diagrams
class StatisticalPersistence {
    double permutationTest(const Diagram& a, const Diagram& b,
                           Size n_perm = 1000) const;
    double kolmogorovSmirnovTest(const Diagram& a, const Diagram& b) const;
    double computeCohensD(const Diagram& a, const Diagram& b) const;
};

// Robust persistence (outlier-robust estimation)
class RobustPersistence {
    Diagram computeRobustPersistence(
        const std::vector<std::vector<double>>& points,
        const std::string& estimator = "huber") const;
    std::vector<bool> detectOutliers(const Diagram&, double threshold = 2.0) const;
    std::vector<Pair> removeOutliers(const Diagram&, double threshold = 2.0) const;
};

// Utility functions
class ProbabilisticTDA {
    static std::vector<std::vector<double>> samplePointsUniform(
        const std::vector<std::vector<double>>& bounds, Size n);
    static std::vector<std::vector<double>> samplePointsStratified(
        const std::vector<std::vector<double>>& bounds, Size n, Size strata);
    static std::vector<std::vector<double>> addGaussianNoise(
        const std::vector<std::vector<double>>& data, double std);
    static double computeAic(const std::vector<Diagram>&);
    static double computeBic(const std::vector<Diagram>&);
    static double computeCrossValidationScore(
        const std::vector<Diagram>&, Size n_folds = 5);
};

}
```


## Monte Carlo sampling

`MonteCarloPersistence` generates samples of persistence diagrams under
various noise models and sampling strategies.

```cpp
MonteCarloPersistence mcp;
mcp.setSamplingMethod("monte_carlo");
mcp.setBurnIn(50);

auto samples = mcp.monteCarloSample(points, 1000, "gaussian");
// -> 1000 perturbed diagrams

auto rhat = mcp.computeGelmanRubinStatistic(samples);
// -> potential scale reduction factor (should be < 1.1 for convergence)
```

Three sampling methods are available: `monte_carlo` adds noise to points and recomputes persistence; `mcmc` applies Metropolis-Hastings sampling on point positions; `importance` performs importance-weighted noise sampling.


## Bootstrap

`BootstrapPersistence` computes confidence intervals for persistence
statistics via resampling.

```cpp
BootstrapPersistence bp(diagram);
bp.setNumBootstrapSamples(5000);

auto intervals = bp.bootstrapPersistenceIntervals(0.95);
// -> [[birth_lo, birth_hi], [death_lo, death_hi], ...]

auto mean_pers = bp.computeMeanPersistence();
auto std_pers = bp.computeStdPersistence();
```

Resampling strategy: sample pairs with replacement from the original diagram,
recompute persistence statistics on each resample, compute empirical
percentiles.


## GPU acceleration

`src/probabilistic/probabilistic_gpu.cu`:

```cpp
namespace nerve::probabilistic::gpu {

// cuRAND Philox-based GPU random number generation
// Used for batched noise addition and MCMC proposals
__global__ void probabilistic_sampling_kernel(
    float* points, float* noise, size_t n, size_t dim,
    uint64_t seed, uint64_t offset);

struct ProbabilisticBenchmark {
    double cpu_time_ms, gpu_time_ms, speedup;
    int num_samples;
    float accuracy;
};
ProbabilisticBenchmark benchmarkProbabilistic(int num_samples);

}
```

GPU kernels use `curandPhilox4_32_10_t` for parallel random number generation.
Each thread computes noise for a batch of points, reducing the overhead of
CPU-to-GPU transfers for large sample counts.


## SIMD probability density functions

`src/probabilistic/simd/probabilistic_simd_ops.cpp` provides SIMD probability density functions using AVX-512 intrinsics: the Normal PDF uses `_mm512_exp_pd`, `_mm512_mul_pd`, and `_mm512_sub_pd`; the Normal log PDF uses `_mm512_log_pd` and `_mm512_fmadd_pd`; the Uniform PDF uses `_mm512_cmp_pd` with blends; the Gamma PDF uses `_mm512_exp_pd`, `_mm512_log_pd`, and `_mm512_pow_pd`. These accelerate the likelihood computations used in MCMC acceptance ratios and Bayesian posterior sampling.


## Bayesian persistence

`BayesianPersistence` implements posterior inference for persistence diagrams
using MCMC (Metropolis-Hastings) with configurable priors.

```cpp
BayesianPersistence bp;
bp.setGaussianPrior({0.0, 0.0}, {1.0, 1.0});
auto posterior = bp.samplePosterior(points, 5000, "mcmc");
auto post_mean = bp.computePosteriorMean();
auto credible = bp.computeCredibleIntervals(0.95);
```

`computeBayesFactor` compares two models (e.g., with and without a
topological feature) using the marginal likelihood ratio. Values > 10
indicate strong evidence for model A.


## FAQ

**Q: How many samples do I typically need?**
A: For stable confidence intervals, 1000 bootstrap samples or 500 MCMC samples are usually sufficient. For high-dimensional data (dimension greater than 10), consider increasing to 5000 samples to account for the larger parameter space.

**Q: Which noise model should I choose?**
A: Use Gaussian noise when the measurement noise is symmetric and unbounded. Use uniform noise when the error is bounded (e.g., sensor precision limits). Use Poisson noise for count data. The `monte_carlo` method is the safest default.

**Q: How do I interpret the Gelman-Rubin statistic?**
A: Values below 1.1 indicate that multiple MCMC chains have converged to the same distribution. Values above 1.1 suggest poor mixing or non-convergence; increase the number of samples or adjust the step size.
