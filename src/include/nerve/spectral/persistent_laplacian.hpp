
#pragma once
#include "nerve/config.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
namespace nerve
{
namespace spectral
{
struct SpectralConfig
{
    size_t num_eigenpairs = 50;
    double convergence_tolerance = 1e-8;
    size_t max_iterations = 1000;
    double spectral_shift = 0.0;
};
struct Eigenpair
{
    double eigenvalue;
    Eigen::VectorXd eigenvector;
    double error_estimate;
    uint32_t iterations;
    bool is_harmonic;
    double spectral_gap;
    double participation_ratio;
    std::complex<double> complex_eigenvalue;
    Eigen::VectorXcd complex_eigenvector;
    bool isValid() const
    {
        return eigenvector.size() > 0 && std::isfinite(eigenvalue) && eigenvalue >= 0.0 &&
               std::isfinite(error_estimate) && error_estimate >= 0.0 &&
               std::isfinite(spectral_gap) && spectral_gap >= 0.0 &&
               std::isfinite(participation_ratio) && participation_ratio >= 0.0 &&
               std::isfinite(complex_eigenvalue.real()) &&
               std::isfinite(complex_eigenvalue.imag()) && eigenvector.allFinite() &&
               complex_eigenvector.allFinite();
    }
};
struct SpectralDecomposition
{
    std::vector<Eigenpair> eigenpairs;
    std::vector<Eigenpair> harmonic_eigenpairs;
    std::vector<Eigenpair> nonharmonic_eigenpairs;
    size_t matrix_size;
    size_t num_harmonic;
    double spectral_radius;
    double trace;
    double frobenius_norm;
    double computation_time_ms;
    std::string solver_used;
    bool converged;
    std::vector<double> convergence_history;
    double orthogonality_error;
    double residual_norm;
    double condition_number;
    std::vector<double> eigenvalues;
};
class PersistentLaplacianSolver
{
public:
    explicit PersistentLaplacianSolver(const SpectralConfig &config);
    SpectralDecomposition computeSpectrum(const Eigen::SparseMatrix<double> &laplacian);
    SpectralDecomposition
    computeSpectrumWithWarmStart(const Eigen::SparseMatrix<double> &laplacian,
                                 const SpectralDecomposition &previous_result);
    SpectralDecomposition computeHarmonicSpectrum(const Eigen::SparseMatrix<double> &laplacian);
    SpectralDecomposition computeNonharmonicSpectrum(const Eigen::SparseMatrix<double> &laplacian);
    static Eigen::SparseMatrix<double>
    buildPersistentLaplacian(const std::vector<std::vector<uint32_t>> &boundary_matrix,
                             const std::vector<double> &filtration_values);
    static Eigen::SparseMatrix<double>
    buildDiracOperator(const std::vector<std::vector<uint32_t>> &boundary_matrix);
    std::vector<Eigenpair> extractWarmStartBasis(const SpectralDecomposition &previous_result,
                                                 const Eigen::SparseMatrix<double> &new_laplacian);
    bool validateDecomposition(const SpectralDecomposition &result,
                               const Eigen::SparseMatrix<double> &laplacian);

private:
    SpectralConfig config_;
    SpectralDecomposition solveDirect(const Eigen::SparseMatrix<double> &laplacian);
    SpectralDecomposition solveArnoldi(const Eigen::SparseMatrix<double> &laplacian,
                                       const std::vector<Eigenpair> &warm_start = {});
    SpectralDecomposition solveLanczos(const Eigen::SparseMatrix<double> &laplacian,
                                       const std::vector<Eigenpair> &warm_start = {});
    std::vector<size_t> findHarmonicForms(const Eigen::SparseMatrix<double> &laplacian);
    double computeOrthogonalityError(const std::vector<Eigenpair> &eigenpairs);
    double computeResidualNorm(const Eigen::SparseMatrix<double> &laplacian,
                               const Eigenpair &eigenpair);
};
class PersistentLaplacianSolverGPU
{
public:
    explicit PersistentLaplacianSolverGPU(const SpectralConfig &config);
    ~PersistentLaplacianSolverGPU();
    errors::ErrorResult<SpectralDecomposition>
    computeSpectrumGpu(const Eigen::SparseMatrix<double> &laplacian);
    errors::ErrorResult<SpectralDecomposition>
    computeSpectrumGpuWithWarmStart(const Eigen::SparseMatrix<double> &laplacian,
                                    const SpectralDecomposition &previous_result);
    bool isAvailable() const;
    std::string getGpuInfo() const;
    void setGpuMemoryLimit(size_t limit_mb);
    size_t getGpuMemoryUsage() const;

private:
    SpectralConfig config_;
    void *gpu_context_;
    void *gpu_memory_pool_;
    bool initializeGPU();
    void cleanupGPU();
    errors::ErrorResult<SpectralDecomposition>
    gpuArnoldi(const Eigen::SparseMatrix<double> &laplacian,
               const std::vector<Eigenpair> &warm_start = {});
    errors::ErrorResult<SpectralDecomposition>
    gpuLanczos(const Eigen::SparseMatrix<double> &laplacian,
               const std::vector<Eigenpair> &warm_start = {});
};
class SpectralFeatureExtractor
{
public:
    struct FeatureConfig
    {
        bool use_eigenvalues = true;
        bool use_eigenvectors = true;
        bool use_spectral_gaps = true;
        bool use_participation_ratios = true;
        bool normalize_features = true;
        size_t max_features = 100;
    };
    explicit SpectralFeatureExtractor(const FeatureConfig &config);
    std::vector<float> extractFeatures(const SpectralDecomposition &spectrum);
    std::vector<float> extractHarmonicFeatures(const SpectralDecomposition &spectrum);
    std::vector<float> extractNonharmonicFeatures(const SpectralDecomposition &spectrum);
    std::vector<float> extractSpectralStatistics(const SpectralDecomposition &spectrum);
    std::vector<float> extractTopologicalFeatures(const SpectralDecomposition &spectrum);
    std::vector<std::vector<float>>
    extractFeaturesBatch(const std::vector<SpectralDecomposition> &spectra);

private:
    FeatureConfig config_;
    std::vector<float> normalizeFeatureVector(const std::vector<float> &features);
    std::vector<float> computeSpectralGaps(const std::vector<Eigenpair> &eigenpairs);
    std::vector<float> computeParticipationRatios(const std::vector<Eigenpair> &eigenpairs);
};
class SpectralAnomalyDetector
{
public:
    struct AnomalyConfig
    {
        double anomaly_threshold = 3.0;
        size_t reference_window_size = 100;
        bool use_eigenvalue_anomalies = true;
        bool use_eigenvector_anomalies = true;
        bool enable_adaptive_threshold = true;
    };
    explicit SpectralAnomalyDetector(const AnomalyConfig &config);
    struct AnomalyResult
    {
        bool is_anomaly;
        double anomaly_score;
        std::vector<size_t> anomalous_eigenpairs;
        std::vector<double> eigenvalue_deviations;
        std::vector<double> eigenvector_deviations;
        std::string anomaly_description;
    };
    AnomalyResult detectAnomaly(const SpectralDecomposition &current_spectrum);
    void updateReference(const std::vector<SpectralDecomposition> &reference_spectra);
    void resetReference();
    void updateAdaptiveThreshold(const std::vector<AnomalyResult> &recent_results);

private:
    AnomalyConfig config_;
    std::vector<SpectralDecomposition> reference_spectra_;
    double adaptive_threshold_;
    std::vector<double>
    computeEigenvalueStatistics(const std::vector<SpectralDecomposition> &spectra);
    std::vector<double>
    computeEigenvectorStatistics(const std::vector<SpectralDecomposition> &spectra);
    double computeEigenvalueAnomalyScore(const SpectralDecomposition &spectrum,
                                         const std::vector<double> &reference_stats);
    double computeEigenvectorAnomalyScore(const SpectralDecomposition &spectrum,
                                          const std::vector<double> &reference_stats);
};
class SpectralStackManager
{
public:
    static SpectralStackManager &instance();
    void setConfig(const SpectralConfig &config);
    SpectralConfig getConfig() const;
    std::shared_ptr<PersistentLaplacianSolver> getCpuSolver();
    std::shared_ptr<PersistentLaplacianSolverGPU> getGpuSolver();
    SpectralDecomposition computeSpectrum(const Eigen::SparseMatrix<double> &laplacian);
    SpectralDecomposition
    computeSpectrumWithWarmStart(const Eigen::SparseMatrix<double> &laplacian,
                                 const SpectralDecomposition &previous_result);
    std::shared_ptr<SpectralFeatureExtractor> getFeatureExtractor();
    std::vector<float> extractFeatures(const SpectralDecomposition &spectrum);
    std::shared_ptr<SpectralAnomalyDetector> getAnomalyDetector();
    SpectralAnomalyDetector::AnomalyResult detectAnomaly(const SpectralDecomposition &spectrum);
    std::vector<float> prepareGnnFeatures(const SpectralDecomposition &spectrum);
    struct PerformanceStats
    {
        double average_computation_time_ms;
        size_t total_computations;
        size_t gpu_computations;
        size_t cpu_computations;
        double average_gpu_time_ms;
        double average_cpu_time_ms;
        float gpu_speedup;
    };
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    SpectralStackManager() = default;
    SpectralConfig config_;
    std::shared_ptr<PersistentLaplacianSolver> cpu_solver_;
    std::shared_ptr<PersistentLaplacianSolverGPU> gpu_solver_;
    std::shared_ptr<SpectralFeatureExtractor> feature_extractor_;
    std::shared_ptr<SpectralAnomalyDetector> anomaly_detector_;
    mutable std::shared_mutex mutex_;
    mutable PerformanceStats performance_stats_;
    void updatePerformanceStats(bool used_gpu, double computation_time_ms);
};
} // namespace spectral

#else
namespace nerve
{
namespace spectral
{
struct SpectralConfig
{
    size_t num_eigenpairs = 50;
    double convergence_tolerance = 1e-8;
    size_t max_iterations = 1000;
    double spectral_shift = 0.0;
};

struct Eigenpair
{
    double eigenvalue = 0.0;
    std::vector<double> eigenvector;
    double error_estimate = 0.0;
    uint32_t iterations = 0;
    bool is_harmonic = false;
    double spectral_gap = 0.0;
    double participation_ratio = 0.0;
    std::complex<double> complex_eigenvalue{};
    std::vector<std::complex<double>> complex_eigenvector;

    bool isValid() const
    {
        return !eigenvector.empty() && std::isfinite(eigenvalue) && eigenvalue >= 0.0 &&
               std::isfinite(error_estimate) && error_estimate >= 0.0 &&
               std::isfinite(spectral_gap) && spectral_gap >= 0.0 &&
               std::isfinite(participation_ratio) && participation_ratio >= 0.0 &&
               std::isfinite(complex_eigenvalue.real()) &&
               std::isfinite(complex_eigenvalue.imag()) &&
               std::all_of(eigenvector.begin(), eigenvector.end(),
                           [](double value) { return std::isfinite(value); }) &&
               std::all_of(complex_eigenvector.begin(), complex_eigenvector.end(),
                           [](const auto &value) {
                               return std::isfinite(value.real()) && std::isfinite(value.imag());
                           });
    }
};
struct SpectralDecomposition
{
    std::vector<Eigenpair> eigenpairs;
    std::vector<Eigenpair> harmonic_eigenpairs;
    std::vector<Eigenpair> nonharmonic_eigenpairs;
    size_t matrix_size = 0;
    size_t num_harmonic = 0;
    double spectral_radius = 0.0;
    double trace = 0.0;
    double frobenius_norm = 0.0;
    double computation_time_ms = 0.0;
    std::string solver_used;
    bool converged = false;
    std::vector<double> convergence_history;
    double orthogonality_error = 0.0;
    double residual_norm = 0.0;
    double condition_number = 0.0;
    std::vector<double> eigenvalues;
};

class PersistentLaplacianSolver;
class PersistentLaplacianSolverGPU;
class SpectralFeatureExtractor;
class SpectralAnomalyDetector;
class SpectralStackManager;
} // namespace spectral
#endif

} // namespace nerve
