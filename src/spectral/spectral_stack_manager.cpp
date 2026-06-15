
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"

#include <algorithm>
#include <chrono>
#include <shared_mutex>

namespace nerve
{
namespace spectral
{

// SpectralStackManager Implementation

SpectralStackManager &SpectralStackManager::instance()
{
    static SpectralStackManager instance;
    return instance;
}

void SpectralStackManager::setConfig(const SpectralConfig &config)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_ = config;

    // Recreate solvers with new config
    cpu_solver_ = nullptr;
    gpu_solver_ = nullptr;
}

SpectralConfig SpectralStackManager::getConfig() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return config_;
}

std::shared_ptr<PersistentLaplacianSolver> SpectralStackManager::getCpuSolver()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!cpu_solver_)
    {
        cpu_solver_ = std::make_shared<PersistentLaplacianSolver>(config_);
    }

    return cpu_solver_;
}

std::shared_ptr<PersistentLaplacianSolverGPU> SpectralStackManager::getGpuSolver()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!gpu_solver_)
    {
        auto candidate = std::make_shared<PersistentLaplacianSolverGPU>(config_);
        if (candidate->isAvailable())
        {
            gpu_solver_ = std::move(candidate);
        }
    }

    return gpu_solver_;
}

SpectralDecomposition
SpectralStackManager::computeSpectrum(const Eigen::SparseMatrix<double> &laplacian)
{
    auto start = std::chrono::high_resolution_clock::now();

    SpectralDecomposition result;
    bool used_gpu = false;
    bool enable_gpu = true;

    if (enable_gpu)
    {
        auto gpu_solver = getGpuSolver();
        if (gpu_solver)
        {
            auto gpu_result = gpu_solver->computeSpectrumGpu(laplacian);
            if (gpu_result.isSuccess())
            {
                result = gpu_result.moveValue();
                used_gpu = result.solver_used.rfind("gpu:", 0) == 0;
            }
        }
    }

    if (!used_gpu)
    {
        auto cpu_solver = getCpuSolver();
        result = cpu_solver->computeSpectrum(laplacian);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    updatePerformanceStats(used_gpu, static_cast<double>(duration.count()));

    return result;
}

SpectralDecomposition
SpectralStackManager::computeSpectrumWithWarmStart(const Eigen::SparseMatrix<double> &laplacian,
                                                   const SpectralDecomposition &previous_result)
{
    auto start = std::chrono::high_resolution_clock::now();

    SpectralDecomposition result;
    bool used_gpu = false;
    bool enable_gpu = true;

    if (enable_gpu)
    {
        auto gpu_solver = getGpuSolver();
        if (gpu_solver)
        {
            auto gpu_result =
                gpu_solver->computeSpectrumGpuWithWarmStart(laplacian, previous_result);
            if (gpu_result.isSuccess())
            {
                result = gpu_result.moveValue();
                used_gpu = result.solver_used.rfind("gpu:", 0) == 0;
            }
        }
    }

    if (!used_gpu)
    {
        auto cpu_solver = getCpuSolver();
        result = cpu_solver->computeSpectrumWithWarmStart(laplacian, previous_result);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    updatePerformanceStats(used_gpu, static_cast<double>(duration.count()));

    return result;
}

std::shared_ptr<SpectralFeatureExtractor> SpectralStackManager::getFeatureExtractor()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!feature_extractor_)
    {
        SpectralFeatureExtractor::FeatureConfig feat_config;
        feature_extractor_ = std::make_shared<SpectralFeatureExtractor>(feat_config);
    }

    return feature_extractor_;
}

std::vector<float> SpectralStackManager::extractFeatures(const SpectralDecomposition &spectrum)
{
    auto extractor = getFeatureExtractor();
    return extractor->extractFeatures(spectrum);
}

std::shared_ptr<SpectralAnomalyDetector> SpectralStackManager::getAnomalyDetector()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!anomaly_detector_)
    {
        SpectralAnomalyDetector::AnomalyConfig anomaly_config;
        anomaly_detector_ = std::make_shared<SpectralAnomalyDetector>(anomaly_config);
    }

    return anomaly_detector_;
}

SpectralAnomalyDetector::AnomalyResult
SpectralStackManager::detectAnomaly(const SpectralDecomposition &spectrum)
{
    auto detector = getAnomalyDetector();
    return detector->detectAnomaly(spectrum);
}

std::vector<float> SpectralStackManager::prepareGnnFeatures(const SpectralDecomposition &spectrum)
{
    // Extract features suitable for GNN input
    auto features = extractFeatures(spectrum);

    // Ensure fixed size for GNN
    const size_t gnn_input_size = 64;

    if (features.size() < gnn_input_size)
    {
        // Pad with zeros
        features.resize(gnn_input_size, 0.0f);
    }
    else if (features.size() > gnn_input_size)
    {
        // Truncate
        features.resize(gnn_input_size);
    }

    return features;
}

SpectralStackManager::PerformanceStats SpectralStackManager::getPerformanceStats() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return performance_stats_;
}

void SpectralStackManager::resetPerformanceStats()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    performance_stats_ = PerformanceStats{};
}

void SpectralStackManager::updatePerformanceStats(bool used_gpu, double computation_time_ms)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    ++performance_stats_.total_computations;

    if (used_gpu)
    {
        ++performance_stats_.gpu_computations;
        // Update average GPU time
        double old_avg = performance_stats_.average_gpu_time_ms;
        double n = static_cast<double>(performance_stats_.gpu_computations);
        performance_stats_.average_gpu_time_ms = (old_avg * (n - 1) + computation_time_ms) / n;
    }
    else
    {
        ++performance_stats_.cpu_computations;
        // Update average CPU time
        double old_avg = performance_stats_.average_cpu_time_ms;
        double n = static_cast<double>(performance_stats_.cpu_computations);
        performance_stats_.average_cpu_time_ms = (old_avg * (n - 1) + computation_time_ms) / n;
    }

    // Update overall average
    double old_avg = performance_stats_.average_computation_time_ms;
    double n = static_cast<double>(performance_stats_.total_computations);
    performance_stats_.average_computation_time_ms = (old_avg * (n - 1) + computation_time_ms) / n;

    // Compute speedup
    if (performance_stats_.average_cpu_time_ms > 0 && performance_stats_.average_gpu_time_ms > 0)
    {
        performance_stats_.gpu_speedup = static_cast<float>(performance_stats_.average_cpu_time_ms /
                                                            performance_stats_.average_gpu_time_ms);
    }
}

} // namespace spectral
} // namespace nerve
