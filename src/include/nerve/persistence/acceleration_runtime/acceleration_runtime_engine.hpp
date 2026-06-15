
#pragma once

#include "acceleration_runtime_capabilities.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nerve::persistence::acceleration_runtime
{

// Constants for magic numbers
constexpr double MIN_COMPUTATION_TIME_MS = 1e-9;
constexpr size_t DEFAULT_CHUNK_SIZE = 8192;
constexpr double DEFAULT_SPEEDUP_FACTOR = 1.0;

// Use forward declarations - enums are defined in fast_vr.hpp
enum class AlgorithmType;
enum class ApproximationLevel;

/**
 * @brief Performance statistics for accelerated operations
 */
struct AccelerationRuntimeStats
{
    double computation_time_ms = 0.0;
    size_t memory_used_bytes = 0;
    size_t operations_performed = 0;
    double speedup_factor = 1.0;
    std::string algorithm_used;
    std::string optimization_details;
};

/**
 * @brief Compiled accelerated VR engine
 */
class AccelerationRuntimeVrEngine
{
public:
    /**
     * @brief Create accelerated VR engine
     * @param config Configuration for optimization acceleration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
    create(const VRConfig &config);

    /**
     * @brief Compute Vietoris-Rips persistence with accelerated acceleration
     * @param points Input points
     * @param point_dim Point dimension
     * @param config VR configuration
     * @return Persistence pairs or error
     */
    errors::ErrorResult<std::vector<Pair>> computeVrPersistence(const std::vector<double> &points,
                                                                size_t point_dim,
                                                                const VRConfig &config);

    /**
     * @brief Get comprehensive performance statistics
     * @return Complete performance statistics
     */
    AccelerationRuntimeStats getPerformanceStats() const;

    /**
     * @brief Get system capabilities
     * @return System capabilities information
     */
    const SystemCapabilities &getSystemCapabilities() const { return system_; }

    virtual ~AccelerationRuntimeVrEngine() = default;

private:
    AccelerationRuntimeVrEngine(const VRConfig &config);

    // Core components
    VRConfig config_;
    SystemCapabilities system_;

    // Performance tracking with thread safety
    mutable std::mutex performance_stats_mutex_;
    AccelerationRuntimeStats performance_stats_;
};

/**
 * @brief Accelerated engine factory methods
 */
class AccelerationRuntimeEngineFactory
{
public:
    /**
     * @brief Create accelerated engine with optimal configuration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>> createOptimal();

    /**
     * @brief Create accelerated engine for specific use case
     * @param use_case Target use case
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
    createForUseCase(const std::string &use_case);

    /**
     * @brief Create accelerated engine with custom configuration
     * @param config Custom configuration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
    createWithConfig(const VRConfig &config);
};

/**
 * @brief Accelerated system validator
 */
class AccelerationRuntimeSystemValidator
{
public:
    /**
     * @brief Validate system capabilities
     * @return Validation result
     */
    static errors::ErrorResult<bool> validateSystem();

    /**
     * @brief Validate GPU capabilities
     * @return Validation result
     */
    static errors::ErrorResult<bool> validateGpu();

    /**
     * @brief Validate threading capabilities
     * @return Validation result
     */
    static errors::ErrorResult<bool> validateThreading();

    /**
     * @brief Validate memory capabilities
     * @return Validation result
     */
    static errors::ErrorResult<bool> validateMemory();

    /**
     * @brief Validate all capabilities
     * @return Validation result
     */
    static errors::ErrorResult<bool> validateAll();
};

} // namespace nerve::persistence::acceleration_runtime
