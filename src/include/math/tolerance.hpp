// math/tolerance.hpp
// Context-aware numeric tolerances for filtration, geometry, matrix,
// gradient, and ULP comparisons.
#pragma once
#include "nerve/core_types.hpp"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

namespace nerve::math
{

// Machine epsilon constants (compile-time).
template <typename T>
struct MachineEps;
template <>
struct MachineEps<float>
{
    static constexpr float value = FLT_EPSILON; // ~1.19e-7
    static constexpr int ulp_bits = 23;
};
template <>
struct MachineEps<double>
{
    static constexpr double value = DBL_EPSILON; // ~2.22e-16
    static constexpr int ulp_bits = 52;
};

// Context: what kind of comparison are we making?
// Different mathematical contexts need different tolerances.

// [1] Field element (Fp coefficients): exact integer arithmetic
struct FieldElementContext
{};

// [2] Filtration value: real-valued, relative tolerance
struct FiltrationContext
{
    double relative = 1e-9;  // 9 decimal digits, safe for double
    double absolute = 1e-12; // floor for near-zero values
};

// [3] Geometric distance: scale-dependent, relative
struct GeometricContext
{
    double data_scale = 1.0; // max coordinate magnitude in dataset
    double relative = 1e-9;
};

// [4] Matrix entry (over R): absolute small, but precision-aware
struct MatrixEntryContext
{
    double threshold = 0.0; // set to matrix_frobenius_norm * eps
};

// [5] Gradient / norm: relative to gradient magnitude
struct GradientContext
{
    double relative = 1e-7; // looser; gradients can have large condition number
};

// Tolerance computer: typed, context-driven.
class Tolerance
{
public:
    // Field elements: exact comparison.
    // Fp arithmetic is modular integer; never use floating point comparison.
    template <typename F>
        requires(!std::is_floating_point_v<F>)
    static bool isZero(F x, FieldElementContext) noexcept
    {
        return x == F::zero();
    }

    // Filtration values.
    static bool isZero(double x, FiltrationContext ctx) noexcept
    {
        return std::abs(x) <= ctx.absolute;
    }

    static bool nearlyEqual(double a, double b, FiltrationContext ctx) noexcept
    {
        if (a == b)
            return true;
        double scale = std::max({std::abs(a), std::abs(b), 1.0});
        return std::abs(a - b) <= std::max(ctx.relative * scale, ctx.absolute);
    }

    // a <= b with tolerance for filtration monotonicity checks.
    static bool leq(double a, double b, FiltrationContext ctx) noexcept
    {
        return a <= b + ctx.relative * std::max(std::abs(b), 1.0);
    }

    // Geometric distances.
    static bool isZero(double x, GeometricContext ctx) noexcept
    {
        return std::abs(x) <= ctx.relative * ctx.data_scale;
    }

    static bool nearlyEqual(double a, double b, GeometricContext ctx) noexcept
    {
        double scale =
            ctx.data_scale > 0 ? ctx.data_scale : std::max({std::abs(a), std::abs(b), 1.0});
        return std::abs(a - b) <= ctx.relative * scale;
    }

    // Matrix entries.
    static bool isNegligible(double x, MatrixEntryContext ctx) noexcept
    {
        return std::abs(x) <= ctx.threshold;
    }

    // Build a MatrixEntryContext from the matrix Frobenius norm
    static MatrixEntryContext forMatrix(double frobenius_norm) noexcept
    {
        return {frobenius_norm * MachineEps<double>::value * 10.0};
    }

    // Gradients and norms.
    static bool isZeroNorm(double norm, GradientContext ctx) noexcept
    {
        return norm <= ctx.relative;
    }

    // ULP comparison for low-level numeric checks.
    // Compares a and b within `ulps` Units in the Last Place.
    // Only valid for same-sign, non-NaN/Inf values.
    static bool withinUlps(double a, double b, int ulps = 4) noexcept
    {
        if (std::isnan(a) || std::isnan(b))
            return false;
        if (std::isinf(a) || std::isinf(b))
            return a == b;
        if (std::signbit(a) != std::signbit(b))
            return a == b;

        auto ia = std::bit_cast<int64_t>(a);
        auto ib = std::bit_cast<int64_t>(b);
        return std::abs(ia - ib) <= ulps;
    }

    // Data-scale estimation.
    // Call once per dataset to get the right GeometricContext
    static GeometricContext estimateScale(const double *points, nerve::Size n_points,
                                          nerve::Size dim) noexcept
    {
        if (points == nullptr || n_points == 0 || dim == 0)
        {
            return {.data_scale = 1.0};
        }
        if (n_points > std::numeric_limits<nerve::Size>::max() / dim)
        {
            return {.data_scale = 1.0};
        }

        const nerve::Size coordinate_count = n_points * dim;
        double max_coord = 0.0;
        for (nerve::Size i = 0; i < coordinate_count; ++i)
        {
            const double coord = points[i];
            if (!std::isfinite(coord))
            {
                return {.data_scale = 1.0};
            }
            max_coord = std::max(max_coord, std::abs(coord));
        }
        return {.data_scale = std::max(max_coord, 1.0)};
    }
};

} // namespace nerve::math
