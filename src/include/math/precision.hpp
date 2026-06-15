#pragma once
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nerve::math
{

enum class PrecisionLevel
{
    FP16,  // ~0.1% relative error, 3 decimal digits
    FP32,  // ~1.2e-7 relative error, 7 decimal digits
    FP64,  // ~2.2e-16 relative error, 15 decimal digits
    EXACT, // integer / modular arithmetic, no floating point error
};

constexpr double machineEps(PrecisionLevel p) noexcept
{
    switch (p)
    {
        case PrecisionLevel::FP16:
            return 9.765625e-4; // 2^-10
        case PrecisionLevel::FP32:
            return 1.1920929e-7; // 2^-23
        case PrecisionLevel::FP64:
            return 2.2204460e-16; // 2^-52
        case PrecisionLevel::EXACT:
            return 0.0;
    }
    return 0.0;
}

template <typename T>
struct WithPrecision
{
    T value;
    PrecisionLevel precision;
    double error_bound; // absolute upper bound on |true - computed|

    bool isReliableFor(double min_feature) const noexcept
    {
        return error_bound < min_feature / 2.0;
    }

    std::string precisionWarning() const
    {
        if (precision == PrecisionLevel::FP16)
            return "WARNING: FP16 precision. Features with persistence < " +
                   std::to_string(error_bound * 2) +
                   " may be spurious. Use FP32 or FP64 for reliable results.";
        return "";
    }
};

struct FP16DistanceResult
{
    float distance;  // computed value (float, not half  --  already converted)
    float max_error; // |true_dist - computed_dist| <= max_error
    bool reliable;   // false = error_bound > significance_threshold

    // Conversion from raw FP16 computation
    // dist_fp16 = computed distance in FP16
    // true_scale = max possible distance in dataset (for error bound)
    static FP16DistanceResult fromFp16(float dist_computed, float true_scale_upper_bound) noexcept
    {
        // FP16 relative error: at most eps_half = 2^-10 approx 0.001
        // Each arithmetic operation introduces relative error eps_half/2
        // For Euclidean distance (dim additions + 1 sqrt):
        //   error <= (dim * eps_half/2 + eps_half/2) * true_dist
        //   conservative bound: 2 * eps_half * true_scale
        constexpr float EPS_HALF = 9.765625e-4f;
        float max_err = 2.0f * EPS_HALF * true_scale_upper_bound;

        return {
            .distance = dist_computed,
            .max_error = max_err,
            .reliable = (dist_computed > max_err * 10.0f) // SNR > 10
        };
    }
};

struct PrecisionPolicy
{
    // Distance computation precision
    PrecisionLevel distance_precision = PrecisionLevel::FP64;

    // Filtration value precision: ALWAYS double
    PrecisionLevel filtration_precision = PrecisionLevel::FP64;

    // Field arithmetic: ALWAYS exact
    PrecisionLevel field_precision = PrecisionLevel::EXACT;

    // Minimum feature size the user cares about (for precision validation)
    double min_feature_size = 0.0; // 0 = user doesn't know / don't check

    // Validate: is the chosen precision sufficient for the data?
    nerve::error::Result<void> validateForData(double data_scale, double min_persistence) const
    {
        double eps = machineEps(distance_precision);
        double distance_error = 2.0 * eps * data_scale;

        if (distance_error >= min_persistence / 2.0)
        {
            return nerve::error::Result<void>::err(
                nerve::error::TDAErrorCode::PrecisionInsufficient,
                "Distance precision " + std::to_string(distance_error) +
                    " is too coarse to reliably detect persistence pairs of size " +
                    std::to_string(min_persistence) +
                    ". Switch to FP64 or increase min_feature_size threshold.");
        }
        return nerve::error::Result<void>::ok();
    }

    // Quick check: is FP16 safe for this dataset?
    bool fp16IsSafe(double data_scale, double min_persistence) const noexcept
    {
        constexpr double EPS_HALF = 9.765625e-4;
        return (2.0 * EPS_HALF * data_scale) < (min_persistence / 2.0);
    }

    // Get error bounds for different precision levels
    double getErrorBound(PrecisionLevel precision, double data_scale) const noexcept
    {
        double eps = machineEps(precision);
        return 2.0 * eps * data_scale; // Conservative bound for distance computations
    }

    // Check if a persistence pair is reliable given current precision
    bool isPairReliable(double birth, double death, double data_scale) const noexcept
    {
        if (death == std::numeric_limits<double>::infinity())
            return true; // Essential

        double persistence = death - birth;
        double error_bound = getErrorBound(distance_precision, data_scale);

        return persistence > 2.0 * error_bound;
    }

    // Filter unreliable pairs from a diagram
    std::vector<nerve::Size> filterUnreliablePairs(const std::vector<nerve::Pair> &pairs,
                                                   double data_scale) const
    {
        std::vector<nerve::Size> unreliable;
        for (nerve::Size i = 0; i < pairs.size(); ++i)
        {
            if (!isPairReliable(pairs[i].birth, pairs[i].death, data_scale))
            {
                unreliable.push_back(i);
            }
        }
        return unreliable;
    }
};

class PrecisionAwareDistance
{
public:
    explicit PrecisionAwareDistance(const PrecisionPolicy &policy = PrecisionPolicy{})
        : policy_(policy)
    {}

    // Compute distance with precision annotation
    WithPrecision<double> computeDistance(const double *a, const double *b, nerve::Size dim,
                                          PrecisionLevel precision = PrecisionLevel::FP64) const
    {
        double dist_sq = 0.0;
        for (nerve::Size i = 0; i < dim; ++i)
        {
            double diff = a[i] - b[i];
            dist_sq += diff * diff;
        }
        double distance = std::sqrt(dist_sq);

        double error_bound = 0.0;
        switch (precision)
        {
            case PrecisionLevel::FP16:
                // FP16: accumulate in FP32, convert to FP16 at end
                // Error bound: 2 * eps_half * max_coord_diff * sqrt(dim)
                error_bound = 2.0 * machineEps(PrecisionLevel::FP16) * distance;
                break;
            case PrecisionLevel::FP32:
                error_bound = machineEps(PrecisionLevel::FP32) * distance;
                break;
            case PrecisionLevel::FP64:
                error_bound = machineEps(PrecisionLevel::FP64) * distance;
                break;
            case PrecisionLevel::EXACT:
                error_bound = 0.0;
                break;
        }

        return {distance, precision, error_bound};
    }

    // Compute distance matrix with precision validation
    nerve::error::Result<std::vector<double>>
    computeDistanceMatrix(const double *points, nerve::Size n_points, nerve::Size dim,
                          PrecisionLevel precision = PrecisionLevel::FP64) const
    {
        if (n_points == 0)
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::EmptyPointCloud, "point count must be non-zero");
        }
        if (dim == 0)
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::InvalidDimension, "point dimension must be non-zero");
        }
        nerve::Size coordinate_count = 0;
        if (!checkedMul(n_points, dim, &coordinate_count))
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::ResourceLimit, "point coordinate count overflows Size");
        }
        nerve::Size matrix_count = 0;
        if (!checkedMul(n_points, n_points, &matrix_count))
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "distance matrix element count overflows Size");
        }
        if (matrix_count > std::vector<double>().max_size())
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "distance matrix element count exceeds vector capacity");
        }
        if (points == nullptr)
        {
            return nerve::error::Result<std::vector<double>>::err(
                nerve::error::TDAErrorCode::InvalidInput, "points pointer must not be null");
        }
        for (nerve::Size i = 0; i < coordinate_count; ++i)
        {
            if (!std::isfinite(points[i]))
            {
                return nerve::error::Result<std::vector<double>>::err(
                    nerve::error::TDAErrorCode::NaNInInput, "point coordinates must be finite");
            }
        }

        // Validate precision for this data
        double data_scale = estimateDataScale(points, coordinate_count);
        if (policy_.min_feature_size > 0.0)
        {
            auto validation = policy_.validateForData(data_scale, policy_.min_feature_size);
            if (!validation.isOk())
            {
                return nerve::error::Result<std::vector<double>>::err(
                    static_cast<nerve::error::TDAErrorCode>(validation.error().value()),
                    std::string(validation.detail()), validation.where());
            }
        }

        std::vector<double> matrix(matrix_count, 0.0);

        for (nerve::Size i = 0; i < n_points; ++i)
        {
            for (nerve::Size j = i + 1; j < n_points; ++j)
            {
                auto result = computeDistance(points + i * dim, points + j * dim, dim, precision);

                matrix[i * n_points + j] = result.value;
                matrix[j * n_points + i] = result.value; // symmetric

                // Warn about precision issues
                if (!result.precisionWarning().empty())
                {
                    // Emit diagnostics without introducing a hard dependency
                    // on a global logging backend in this header-only path.
                    fprintf(stderr, "%s\n", result.precisionWarning().c_str());
                }
            }
        }

        return nerve::error::Result<std::vector<double>>::ok(std::move(matrix));
    }

private:
    PrecisionPolicy policy_;

    static bool checkedMul(nerve::Size lhs, nerve::Size rhs, nerve::Size *result) noexcept
    {
        if (lhs != 0 && rhs > std::numeric_limits<nerve::Size>::max() / lhs)
        {
            return false;
        }
        *result = lhs * rhs;
        return true;
    }

    static double estimateDataScale(const double *points, nerve::Size coordinate_count) noexcept
    {
        double max_coord = 0.0;
        for (nerve::Size i = 0; i < coordinate_count; ++i)
        {
            max_coord = std::max(max_coord, std::abs(points[i]));
        }
        return std::max(max_coord, 1.0);
    }
};

} // namespace nerve::math
