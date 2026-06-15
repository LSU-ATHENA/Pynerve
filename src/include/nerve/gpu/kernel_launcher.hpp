
#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/detail/error_result.hpp"
#include "nerve/gpu/tensor_core_kernels.cuh"
#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "nerve/spectral/laplacian.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::streaming
{

struct Slide
{
    std::vector<double> new_point;
    std::vector<double> old_point;
    std::optional<double> max_radius;
};

struct Window
{
    std::vector<std::vector<double>> points;
    Size max_size = 0;
    Size max_dimension = 1;
    double max_radius = 0.0;
    std::optional<Slide> pending_slide;
    std::vector<int> last_affected_indices;
};

} // namespace nerve::streaming

namespace nerve::gpu
{

// Operation types that can be GPU accelerated
enum class OperationType
{
    kDistanceMatrix,
    kLaplacianConstruction,
    kBoundaryReduction,
    kCohomologyReduction,
    kSimplexBoundary,
    kDiagramDistance,
    kColumnSymmetricDifference,
    kStreamingPersistence,
    kEigenvalueComputation,
    kCount // Keep last for array sizing
};

// GPU-accelerated configuration for all modules
struct Config
{
    // Global settings
    bool enable_gpu = true;
    size_t threshold = 1000; // Minimum problem size to use GPU
    bool auto_select = true; // Auto choose GPU vs CPU
    bool enable_streaming = true;
    size_t chunk_size = 4096;

    // Module-specific overrides
    std::unordered_map<OperationType, bool> enabled;
    std::unordered_map<OperationType, size_t> thresholds;

    // Performance tuning
    size_t max_memory_mb = 0; // 0 = auto-detect
    size_t num_streams = 4;
    bool tensor_cores = true;
    bool mixed_precision = false;
    bool profiling = false;

    Config()
    {
        // Enable all operations by default
        for (int i = 0; i < static_cast<int>(OperationType::kCount); ++i)
        {
            enabled[static_cast<OperationType>(i)] = true;
            thresholds[static_cast<OperationType>(i)] = threshold;
        }
    }
};

// Performance profile for an operation
struct PerformanceProfile
{
    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double speedup = 1.0;
    size_t problem_size = 0;
    size_t memory_used_mb = 0;
    bool used_gpu = false;
    std::string operation_name;
    std::string kernel_name;
    std::chrono::system_clock::time_point timestamp;
};

// Execution strategy chosen by manager
enum class Strategy
{
    kCPUOnly,
    kGPUAccelerated,
    kHybrid
};

// Forward declarations
template <typename T>
class Buffer;

class ComputeManager
{
public:
    static ComputeManager &getInstance();

    void initialize(const Config &config);
    void initialize();

    void shutdown();

    bool isAvailable() const;
    bool isSupported(OperationType operation) const;

    void setConfig(const Config &config);
    Config getConfig() const;

    errors::ErrorResult<void>
    computeDistanceMatrix(const std::vector<std::vector<double>> &points,
                          std::vector<std::vector<double>> &out_distances);

    errors::ErrorResult<void> constructLaplacian(const std::vector<std::vector<double>> &points,
                                                 const std::vector<std::vector<int>> &edges,
                                                 std::vector<std::vector<double>> &out_laplacian);

    errors::ErrorResult<void> symmetricDifferenceColumns(const std::vector<int> &col_i,
                                                         const std::vector<int> &col_j,
                                                         std::vector<int> &out_result);

    errors::ErrorResult<void>
    computeDiagramCostMatrix(const persistence::Diagram &diagram1,
                             const persistence::Diagram &diagram2,
                             std::vector<std::vector<double>> &out_cost_matrix);

    errors::ErrorResult<void> computeReduction(persistence::Reducer &reducer,
                                               const algebra::BoundaryMatrix &matrix);

    errors::ErrorResult<void> computeCohomology(persistence::Reducer &reducer,
                                                const algebra::BoundaryMatrix &matrix);

    errors::ErrorResult<void> processWindowSlide(streaming::Window &window,
                                                 persistence::Diagram &out_diagram);

    errors::ErrorResult<void> detectAffectedRegion(const streaming::Window &window,
                                                   const streaming::Slide &slide,
                                                   std::vector<int> &affected_indices);

    struct VRSimplex
    {
        std::vector<int> vertices;
        double filtration_value;
        int dimension;
    };

    errors::ErrorResult<void> buildVRComplex(const std::vector<std::vector<double>> &points,
                                             double max_radius, int max_dimension,
                                             std::vector<VRSimplex> &out_simplices);

    struct CechSimplex
    {
        std::vector<int> vertices;
        double filtration_value;
        double alpha_value;
        int dimension;
        bool isValid;
    };

    errors::ErrorResult<void> buildCechComplex(const std::vector<std::vector<double>> &points,
                                               double max_radius, int max_dimension,
                                               std::vector<CechSimplex> &out_simplices);

    errors::ErrorResult<void> buildAlphaComplex(const std::vector<std::vector<double>> &points,
                                                double max_alpha, int max_dimension,
                                                std::vector<CechSimplex> &out_simplices);

    errors::ErrorResult<double> solveAssignment(const std::vector<std::vector<double>> &cost_matrix,
                                                std::vector<std::pair<int, int>> &out_assignment);

    errors::ErrorResult<double> solveBottleneck(const std::vector<std::vector<double>> &cost_matrix,
                                                std::vector<std::pair<int, int>> &out_assignment);

    struct ClearingResult
    {
        std::vector<bool> positive_simplices;
        std::vector<bool> columns_to_clear;
        size_t operations_saved;
    };

    errors::ErrorResult<void> applyClearing(const algebra::BoundaryMatrix &boundary_matrix,
                                            const std::vector<int> &simplex_dimensions,
                                            const std::vector<double> &filtration_values,
                                            int target_dimension, double max_filtration,
                                            ClearingResult &out_result);

    // Performance profiling
    void recordPerformance(const PerformanceProfile &profile);
    std::vector<PerformanceProfile> getPerformanceHistory() const;
    void clearPerformanceHistory();

    // Get aggregated stats
    struct Stats
    {
        size_t total_operations = 0;
        size_t gpu_operations = 0;
        double average_speedup = 1.0;
        double total_time_saved_ms = 0.0;
    };

    Stats getStats() const;
    void clearStats();

    // Get last error
    std::string getLastError() const;

    // Memory management
    size_t getAvailableMemoryMB() const;
    void setMemoryLimitMB(size_t limit);

    // Optimized kernel control
    void setUseOptimizedKernels(bool enable);
    bool getUseOptimizedKernels() const;

    // Auto-tuning
    void runAutoTune();
    void loadTunedConfiguration(const std::string &filename);
    void saveTunedConfiguration(const std::string &filename);

    // CUDA Graph support
    void setUseCudaGraphs(bool enable);
    bool getUseCudaGraphs() const;

    // GPU architecture info
    std::string getGPUArchitectureName() const;
    int getComputeCapability() const;

private:
    ComputeManager() = default;
    ~ComputeManager();

    // Prevent copying
    ComputeManager(const ComputeManager &) = delete;
    ComputeManager &operator=(const ComputeManager &) = delete;

    // Internal state
    mutable std::shared_mutex mutex_;
    Config config_;
    bool initialized_ = false;
    bool available_ = false;
    bool useOptimizedKernels_ = true;
    bool useCudaGraphs_ = false;

    // GPU architecture
    int gpuComputeCapability_ = 0;
    int gpuMultiProcessorCount_ = 0;
    size_t gpuSharedMemPerBlock_ = 0;

    // Performance tracking
    std::vector<PerformanceProfile> history_;
    Stats stats_;

    // Last error message
    std::string last_error_;

    // Execution strategy selection
    Strategy selectStrategy(OperationType operation, size_t problem_size) const;

    // Helper methods
    void updateStats(const PerformanceProfile &profile);
    void recordSuccess(const std::string &operation, double speedup);
    void recordFailure(const std::string &operation, const std::string &reason);
    void detectGPUArchitecture();

    // RAII scope for performance tracking
    class OperationScope;

    // CUDA Graph management
    class GraphManager;
    std::unique_ptr<GraphManager> graphManager_;
};

// High-level convenience API for common GPU operations
namespace compute
{
// Initialize with default config
inline void initialize()
{
    ComputeManager::getInstance().initialize();
}

// Initialize with custom config
inline void initialize(const Config &config)
{
    ComputeManager::getInstance().initialize(config);
}

// Check availability
inline bool isAvailable()
{
    return ComputeManager::getInstance().isAvailable();
}
} // namespace compute

} // namespace nerve::gpu
