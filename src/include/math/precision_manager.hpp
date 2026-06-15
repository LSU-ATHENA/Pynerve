
#pragma once
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace nerve::math
{

// Adaptive precision management for numerical stability
class PrecisionManager
{
public:
    enum class PrecisionLevel
    {
        SINGLE,   // float (32-bit)
        DOUBLE,   // double (64-bit)
        HIGH,     // long double (80-bit or 128-bit)
        ARBITRARY // Arbitrary precision (if available)
    };

    enum class OperationType
    {
        MATRIX_REDUCTION,
        DISTANCE_COMPUTATION,
        FIELD_OPERATION,
        NUMERICAL_INTEGRATION,
        kEigenvalueComputation,
        OPTIMIZATION
    };

    struct PrecisionConfig
    {
        PrecisionLevel level;
        double tolerance;
        double epsilon;
        std::string description;

        PrecisionConfig(PrecisionLevel lvl, double tol, double eps, const std::string &desc)
            : level(lvl)
            , tolerance(tol)
            , epsilon(eps)
            , description(desc)
        {}
    };

private:
    inline static const std::array<PrecisionConfig, 4> precision_configs_ = {
        {PrecisionConfig(PrecisionLevel::SINGLE, 1e-6, 1e-7, "Single precision (float)"),
         PrecisionConfig(PrecisionLevel::DOUBLE, 1e-12, 1e-13, "Double precision (double)"),
         PrecisionConfig(PrecisionLevel::HIGH, 1e-15, 1e-16, "High precision (long double)"),
         PrecisionConfig(PrecisionLevel::ARBITRARY, 1e-20, 1e-21, "Arbitrary precision")}};

public:
    // Determine appropriate precision level based on operation and input characteristics
    static PrecisionLevel determinePrecision(OperationType operation, double input_magnitude = 1.0,
                                             size_t problem_size = 1000)
    {
        switch (operation)
        {
            case OperationType::MATRIX_REDUCTION:
                return determineMatrixReductionPrecision(input_magnitude, problem_size);

            case OperationType::DISTANCE_COMPUTATION:
                return determineDistancePrecision(input_magnitude, problem_size);

            case OperationType::FIELD_OPERATION:
                return determineFieldPrecision(problem_size);

            case OperationType::NUMERICAL_INTEGRATION:
                return determineIntegrationPrecision(input_magnitude, problem_size);

            case OperationType::kEigenvalueComputation:
                return determineEigenvaluePrecision(input_magnitude, problem_size);

            case OperationType::OPTIMIZATION:
                return determineOptimizationPrecision(input_magnitude, problem_size);

            default:
                return PrecisionLevel::DOUBLE;
        }
    }

    // Get tolerance for given precision level
    static double getTolerance(PrecisionLevel level)
    {
        return precision_configs_[static_cast<size_t>(level)].tolerance;
    }

    // Get epsilon for given precision level
    static double getEpsilon(PrecisionLevel level)
    {
        return precision_configs_[static_cast<size_t>(level)].epsilon;
    }

    // Get description for given precision level
    static const std::string &getDescription(PrecisionLevel level)
    {
        return precision_configs_[static_cast<size_t>(level)].description;
    }

    // Check if value is effectively zero
    static bool isZero(double value, PrecisionLevel level)
    {
        double tol = getTolerance(level);
        return std::abs(value) < tol;
    }

    // Check if value is effectively zero (automatic precision detection)
    static bool isZero(double value, OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        PrecisionLevel level = determinePrecision(operation);
        return isZero(value, level);
    }

    // Check if two values are effectively equal
    static bool areEqual(double a, double b, PrecisionLevel level)
    {
        double tol = getTolerance(level);
        return std::abs(a - b) < tol;
    }

    // Check if two values are effectively equal (automatic precision detection)
    static bool areEqual(double a, double b,
                         OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        PrecisionLevel level = determinePrecision(operation);
        return areEqual(a, b, level);
    }

    // Check if value is within epsilon of zero
    static bool isNearZero(double value, PrecisionLevel level)
    {
        double eps = getEpsilon(level);
        return std::abs(value) < eps;
    }

    // Check if value is within epsilon of zero (automatic precision detection)
    static bool isNearZero(double value, OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        PrecisionLevel level = determinePrecision(operation);
        return isNearZero(value, level);
    }

    // Get relative tolerance for comparisons
    static double getRelativeTolerance(PrecisionLevel level)
    {
        double base_tol = getTolerance(level);
        return base_tol * 10.0; // 10x base tolerance for relative comparisons
    }

    // Check if two values are relatively equal
    static bool areRelativelyEqual(double a, double b, PrecisionLevel level)
    {
        double rel_tol = getRelativeTolerance(level);
        double abs_diff = std::abs(a - b);
        double scale = std::max(std::abs(a), std::abs(b));
        return abs_diff < rel_tol * scale;
    }

    // Adaptive tolerance based on input magnitude
    static double getAdaptiveTolerance(double input_magnitude, PrecisionLevel level)
    {
        double base_tol = getTolerance(level);
        double magnitude_factor = std::log10(std::max(1.0, std::abs(input_magnitude)));
        return base_tol * (1.0 + magnitude_factor * 0.1);
    }

    // Check if value is zero with adaptive tolerance
    static bool isAdaptiveZero(double value, double input_magnitude, PrecisionLevel level)
    {
        double adaptive_tol = getAdaptiveTolerance(input_magnitude, level);
        return std::abs(value) < adaptive_tol;
    }

    // Validate numerical stability
    static bool isNumericallyStable(double value, OperationType operation)
    {
        PrecisionLevel level = determinePrecision(operation);

        // Check for common numerical issues
        if (std::isnan(value) || std::isinf(value))
        {
            return false;
        }

        // Reject values outside the stable dynamic range used by this policy.
        const double abs_value = std::abs(value);
        if (abs_value < getEpsilon(level))
        {
            return false;
        }

        if (abs_value > 1e100)
        {
            return false;
        }

        return true;
    }

    // Safe division with zero check
    static error::Result<double>
    safeDivide(double numerator, double denominator,
               OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        PrecisionLevel level = determinePrecision(operation);
        double tol = getTolerance(level);

        if (std::abs(denominator) < tol)
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Division by zero or near-zero value");
        }

        double result = numerator / denominator;

        if (!isNumericallyStable(result, operation))
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Division result is numerically unstable");
        }

        return error::Result<double>::ok(result);
    }

    // Safe square root
    static error::Result<double> safeSqrt(double value,
                                          OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        if (value < 0.0)
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Square root of negative number");
        }

        PrecisionLevel level = determinePrecision(operation);
        double tol = getTolerance(level);

        if (std::abs(value) < tol)
        {
            return error::Result<double>::ok(0.0);
        }

        double result = std::sqrt(value);

        if (!isNumericallyStable(result, operation))
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Square root result is numerically unstable");
        }

        return error::Result<double>::ok(result);
    }

    // Safe power operation
    static error::Result<double> safePow(double base, double exponent,
                                         OperationType operation = OperationType::MATRIX_REDUCTION)
    {
        if (base < 0.0 && std::floor(exponent) != exponent)
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Power of negative base with non-integer exponent");
        }

        PrecisionLevel level = determinePrecision(operation);
        double tol = getTolerance(level);

        if (std::abs(base) < tol && exponent < 0.0)
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Near-zero base with negative exponent");
        }

        double result = std::pow(base, exponent);

        if (!isNumericallyStable(result, operation))
        {
            return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                              "Power result is numerically unstable");
        }

        return error::Result<double>::ok(result);
    }

private:
    // Determine precision for matrix reduction
    static PrecisionLevel determineMatrixReductionPrecision(double input_magnitude,
                                                            size_t problem_size)
    {
        if (problem_size > 100000)
        {
            return PrecisionLevel::HIGH; // Large matrices need higher precision
        }
        else if (input_magnitude > 1e6 || input_magnitude < 1e-6)
        {
            return PrecisionLevel::HIGH; // Large magnitude values need higher precision
        }
        else if (problem_size > 10000)
        {
            return PrecisionLevel::DOUBLE;
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }

    // Determine precision for distance computation
    static PrecisionLevel determineDistancePrecision(double input_magnitude, size_t problem_size)
    {
        if (problem_size > 10000)
        {
            return PrecisionLevel::DOUBLE; // Distance computations typically don't need ultra-high
                                           // precision
        }
        else if (input_magnitude > 1e9 || input_magnitude < 1e-9)
        {
            return PrecisionLevel::HIGH; // Extreme values need higher precision
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }

    // Determine precision for field operations
    static PrecisionLevel determineFieldPrecision(size_t problem_size)
    {
        if (problem_size > 1000000)
        {
            return PrecisionLevel::HIGH; // Large field computations need higher precision
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }

    // Determine precision for numerical integration
    static PrecisionLevel determineIntegrationPrecision(double input_magnitude, size_t problem_size)
    {
        const double magnitude = std::abs(input_magnitude);
        if (!std::isfinite(input_magnitude) || magnitude > 1e8 ||
            (magnitude > 0.0 && magnitude < 1e-10))
        {
            return PrecisionLevel::HIGH;
        }
        else if (problem_size > 100000)
        {
            return PrecisionLevel::HIGH; // Integration can accumulate errors
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }

    // Determine precision for eigenvalue computation
    static PrecisionLevel determineEigenvaluePrecision(double input_magnitude, size_t problem_size)
    {
        const double magnitude = std::abs(input_magnitude);
        if (!std::isfinite(input_magnitude) || magnitude > 1e6 ||
            (magnitude > 0.0 && magnitude < 1e-12))
        {
            return PrecisionLevel::HIGH;
        }
        else if (problem_size > 1000)
        {
            return PrecisionLevel::HIGH; // Eigenvalue problems are sensitive
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }

    // Determine precision for optimization
    static PrecisionLevel determineOptimizationPrecision(double input_magnitude,
                                                         size_t problem_size)
    {
        const double magnitude = std::abs(input_magnitude);
        if (!std::isfinite(input_magnitude) || magnitude > 1e12 ||
            (magnitude > 0.0 && magnitude < 1e-12))
        {
            return PrecisionLevel::HIGH;
        }
        else if (problem_size > 10000)
        {
            return PrecisionLevel::DOUBLE; // Optimization typically uses double precision
        }
        else
        {
            return PrecisionLevel::DOUBLE;
        }
    }
};

// Convenience functions for common operations
namespace precision
{

inline bool isZero(double value, PrecisionManager::PrecisionLevel level)
{
    return PrecisionManager::isZero(value, level);
}

inline bool isZero(double value, PrecisionManager::OperationType operation)
{
    return PrecisionManager::isZero(value, operation);
}

inline bool areEqual(double a, double b, PrecisionManager::PrecisionLevel level)
{
    return PrecisionManager::areEqual(a, b, level);
}

inline bool areEqual(double a, double b, PrecisionManager::OperationType operation)
{
    return PrecisionManager::areEqual(a, b, operation);
}

inline error::Result<double> safeDivide(double numerator, double denominator,
                                        PrecisionManager::OperationType operation)
{
    return PrecisionManager::safeDivide(numerator, denominator, operation);
}

inline error::Result<double> safeSqrt(double value, PrecisionManager::OperationType operation)
{
    return PrecisionManager::safeSqrt(value, operation);
}

inline error::Result<double> safePow(double base, double exponent,
                                     PrecisionManager::OperationType operation)
{
    return PrecisionManager::safePow(base, exponent, operation);
}

} // namespace precision

// Factory functions for precision configurations
inline error::Result<PrecisionManager::PrecisionConfig>
makePrecisionConfig(PrecisionManager::PrecisionLevel level)
{
    try
    {
        double tolerance = PrecisionManager::getTolerance(level);
        double epsilon = PrecisionManager::getEpsilon(level);
        const std::string &description = PrecisionManager::getDescription(level);

        return error::Result<PrecisionManager::PrecisionConfig>::ok(
            PrecisionManager::PrecisionConfig(level, tolerance, epsilon, description));
    }
    catch (const std::exception &e)
    {
        return error::Result<PrecisionManager::PrecisionConfig>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create precision config: ") + e.what());
    }
}

} // namespace nerve::math
