#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace nerve::math
{

enum class Precision
{
    SINGLE, // float32 - ~7 decimal digits, error ~1e-7
    DOUBLE, // float64 - ~16 decimal digits, error ~2e-16
    HALF,   // float16 - ~3 decimal digits, error ~1e-3
    MIXED   // Use optimal precision per operation
};

enum class ErrorBound
{
    RELATIVE, // Error relative to magnitude
    ABSOLUTE, // Fixed absolute error
    SCALED    // Scaled by data magnitude
};

template <Precision P>
struct PrecisionTraits;

template <>
struct PrecisionTraits<Precision::SINGLE>
{
    using type = float;
    static constexpr double epsilon = 1e-7;
    static constexpr double relative_error = 1e-6;
    static constexpr const char *name = "float32";
};

template <>
struct PrecisionTraits<Precision::DOUBLE>
{
    using type = double;
    static constexpr double epsilon = 2e-16;
    static constexpr double relative_error = 1e-15;
    static constexpr const char *name = "float64";
};

template <>
struct PrecisionTraits<Precision::HALF>
{
    using type = uint16_t; // FP16 storage
    static constexpr double epsilon = 1e-3;
    static constexpr double relative_error = 1e-2;
    static constexpr const char *name = "float16";
};

struct Tolerance
{
    // Units in the Last Place (ULP) comparison
    // Good for: comparing values of similar magnitude
    template <typename T>
    static bool nearlyEqual(T a, T b, int ulps = 4) noexcept
    {
        static_assert(std::is_floating_point_v<T>);
        if (a == b)
            return true;
        if (std::isnan(a) || std::isnan(b))
            return false;
        if (std::signbit(a) != std::signbit(b))
            return false;

        auto ia = std::bit_cast<int64_t>(a);
        auto ib = std::bit_cast<int64_t>(b);
        return std::abs(ia - ib) <= ulps;
    }

    // Relative + absolute: max(rel*scale, abs_floor)
    // Good for: comparing distances where scale matters
    template <typename T>
    static bool nearlyEqualScaled(T a, T b, T relative = static_cast<T>(1e-9),
                                  T absolute = static_cast<T>(1e-12)) noexcept
    {
        static_assert(std::is_floating_point_v<T>);
        T scale = std::max({std::abs(a), std::abs(b), static_cast<T>(1.0)});
        return std::abs(a - b) <= std::max(relative * scale, absolute);
    }

    // For filtration comparison: is a <= b (with tolerance)?
    template <typename T>
    static bool filtrationLeq(T a, T b, T tol_factor = static_cast<T>(1e-9)) noexcept
    {
        static_assert(std::is_floating_point_v<T>);
        T scale = std::max(std::abs(b), static_cast<T>(1.0));
        return a <= b + tol_factor * scale;
    }

    // For field coefficient comparisons: exact equality only
    template <typename T>
    static bool fieldEqual(T a, T b) noexcept
    {
        // Fields are exact - no floating point tolerance
        return a == b;
    }
};

struct AlgorithmPrecision
{
    // Distance computation: always double for accuracy
    static constexpr Precision distance_precision = Precision::DOUBLE;

    // Filtration values: always double for mathematical correctness
    static constexpr Precision filtration_precision = Precision::DOUBLE;

    // Field coefficients: exact arithmetic (no floating point)
    template <typename F>
    static constexpr bool is_exact_field = std::is_integral_v<F> || requires(F f) {
        { F::zero() } -> std::same_as<F>;
        { F::one() } -> std::same_as<F>;
    };

    // FP16 distances: document error bounds
    static constexpr double fp16_error_bound = 1e-3; // O(epsilon_half * max_dist)

    // Accumulation precision for FP16: use float32
    static constexpr Precision fp16_accumulation = Precision::SINGLE;
};

template <typename From, typename To>
struct PrecisionBoundary
{
    static_assert(std::is_arithmetic_v<From> && std::is_arithmetic_v<To>);

    static To safeCast(From value)
    {
        if constexpr (std::is_same_v<From, To>)
        {
            return value;
        }
        else if constexpr (std::is_floating_point_v<From> && std::is_floating_point_v<To>)
        {
            // Check for overflow/underflow in floating point conversion
            if constexpr (sizeof(To) < sizeof(From))
            {
                // Downgrading precision (double -> float, float -> half)
                if (std::abs(value) > std::numeric_limits<To>::max())
                {
                    return std::copysign(std::numeric_limits<To>::max(), value);
                }
            }
            return static_cast<To>(value);
        }
        else
        {
            // Integer to floating point or vice versa
            return static_cast<To>(value);
        }
    }
};

template <typename T>
concept IsPrecisionValid = std::is_floating_point_v<T> || std::is_integral_v<T>;

template <typename T>
constexpr bool validatePrecision(T value)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return !std::isnan(value) && !std::isinf(value);
    }
    else
    {
        return true; // Integer types are always valid
    }
}

template <Precision P>
class ErrorTracker
{
public:
    using ValueType = typename PrecisionTraits<P>::type;

    ErrorTracker()
        : accumulated_error_(0.0)
    {}

    void addOperation(ValueType result, ValueType expected)
    {
        double error = std::abs(static_cast<double>(result) - static_cast<double>(expected));
        accumulated_error_ += error;
        operation_count_++;
    }

    double getMeanError() const
    {
        return operation_count_ > 0 ? accumulated_error_ / operation_count_ : 0.0;
    }

    double getMaxAllowedError() const { return PrecisionTraits<P>::relative_error; }

    bool isWithinTolerance() const { return getMeanError() <= getMaxAllowedError(); }

private:
    double accumulated_error_;
    std::size_t operation_count_;
};

} // namespace nerve::math
