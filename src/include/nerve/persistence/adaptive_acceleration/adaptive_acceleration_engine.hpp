
#pragma once

#include "adaptive_acceleration_system_capabilities.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nerve::persistence
{

namespace adaptive_acceleration
{

using AlgorithmType = ::nerve::persistence::AlgorithmType;
using ApproximationLevel = ::nerve::persistence::ApproximationLevel;

/**
 * @brief Performance statistics for Adaptive Acceleration operations
 */
struct AdaptiveAccelerationStats
{
    double computation_time_ms = 0.0;
    size_t memory_used_bytes = 0;
    size_t operations_performed = 0;
    double speedup_factor = 1.0;
    std::string algorithm_used;
    std::string optimization_details;
};

/**
 * @brief Compiled Adaptive Acceleration VR engine
 */
class AdaptiveAccelerationVrEngine
{
public:
    /**
     * @brief Create Adaptive Acceleration VR engine
     * @param config Configuration for Adaptive Acceleration acceleration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
    create(const VRConfig &config);

    /**
     * @brief Compute Vietoris-Rips persistence with Adaptive Acceleration acceleration
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
    const AdaptiveAccelerationStats &getPerformanceStats() const { return performance_stats_; }

    /**
     * @brief Get system capabilities
     * @return System capabilities information
     */
    const SystemCapabilities &getSystemCapabilities() const { return system_; }

    virtual ~AdaptiveAccelerationVrEngine() = default;

private:
    AdaptiveAccelerationVrEngine(const VRConfig &config);

    // Core components
    VRConfig config_;
    SystemCapabilities system_;

    // Performance tracking
    AdaptiveAccelerationStats performance_stats_;
};

/**
 * @brief Adaptive Acceleration engine factory methods
 */
class AdaptiveAccelerationEngineFactory
{
public:
    /**
     * @brief Create Adaptive Acceleration engine with optimal configuration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>> createOptimal();

    /**
     * @brief Create Adaptive Acceleration engine for specific use case
     * @param use_case Target use case
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
    createForUseCase(const std::string &use_case);

    /**
     * @brief Create Adaptive Acceleration engine with custom configuration
     * @param config Custom configuration
     * @return Engine instance or error
     */
    static errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
    createWithConfig(const VRConfig &config);
};

/**
 * @brief Adaptive Acceleration system validator
 */
class AdaptiveAccelerationSystemValidator
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

} // namespace adaptive_acceleration
} // namespace nerve::persistence
