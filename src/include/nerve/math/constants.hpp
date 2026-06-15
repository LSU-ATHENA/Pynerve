#pragma once

// Mathematical and numerical constants for Nerve
// Following PyTorch-style constexpr constants at file scope

#include <cmath>
#include <cstddef>
#include <limits>

namespace nerve::math
{

// Mathematical Constants (high precision)

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kE = 2.718281828459045235360287471352662498;
inline constexpr double kSqrt2 = 1.414213562373095048801688724209698079;
inline constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
inline constexpr double kLn2 = 0.693147180559945309417232121458176568;
inline constexpr double kLn10 = 2.302585092994045684017991454684364208;

// Numerical Precision Constants

template <typename T>
struct Epsilon
{
    static constexpr T value = std::numeric_limits<T>::epsilon();
};

template <>
struct Epsilon<float>
{
    static constexpr float value = 1e-6f; // Less strict for float
};

template <>
struct Epsilon<double>
{
    static constexpr double value = 1e-10; // Stricter for double
};

// Type-safe epsilon accessor
inline constexpr float kEpsilonFloat = Epsilon<float>::value;
inline constexpr double kEpsilonDouble = Epsilon<double>::value;

// Generic epsilon function
template <typename T>
inline constexpr T epsilon()
{
    return Epsilon<T>::value;
}

// Common Thresholds

inline constexpr double kDefaultTolerance = 1e-6;
inline constexpr double kMachinePrecision = std::numeric_limits<double>::epsilon();
inline constexpr float kMinFloat = std::numeric_limits<float>::min();
inline constexpr float kMaxFloat = std::numeric_limits<float>::max();
inline constexpr double kMinDouble = std::numeric_limits<double>::min();
inline constexpr double kMaxDouble = std::numeric_limits<double>::max();

// SIMD and Hardware Constants

namespace simd
{
// AVX-512
inline constexpr size_t kAvx512Bits = 512;
inline constexpr size_t kAvx512Bytes = kAvx512Bits / 8;
inline constexpr size_t kAvx512Floats = kAvx512Bits / 32;  // 16 floats
inline constexpr size_t kAvx512Doubles = kAvx512Bits / 64; // 8 doubles

inline constexpr size_t kAvx2Bits = 256;
inline constexpr size_t kAvx2Bytes = kAvx2Bits / 8;
inline constexpr size_t kAvx2Floats = kAvx2Bits / 32;  // 8 floats
inline constexpr size_t kAvx2Doubles = kAvx2Bits / 64; // 4 doubles

inline constexpr size_t kSseBits = 128;
inline constexpr size_t kSseBytes = kSseBits / 8;
inline constexpr size_t kSseFloats = kSseBits / 32;  // 4 floats
inline constexpr size_t kSseDoubles = kSseBits / 64; // 2 doubles
} // namespace simd

namespace cache
{
inline constexpr size_t kLineSize = 64; // Bytes per cache line
inline constexpr size_t kL1Size = size_t{32} * size_t{1024};
inline constexpr size_t kL2Size = size_t{256} * size_t{1024};
inline constexpr size_t kL3Size = size_t{8} * size_t{1024} * size_t{1024};
} // namespace cache

// Algorithm-Specific Constants

namespace distance
{
inline constexpr size_t kDefaultBlockSize = 64;
inline constexpr size_t kMinBlockSize = 16;
inline constexpr size_t kMaxBlockSize = 256;
inline constexpr double kDefaultMetricEpsilon = 1e-10;
} // namespace distance

namespace knn
{
inline constexpr size_t kDefaultK = 5;
inline constexpr size_t kMaxK = 1000;
inline constexpr size_t kMinK = 1;
} // namespace knn

namespace mapper
{
inline constexpr int kDefaultCoverResolution = 10;
inline constexpr float kDefaultOverlap = 0.25f;
inline constexpr float kMinOverlap = 0.0f;
inline constexpr float kMaxOverlap = 0.5f;
inline constexpr float kDefaultDbscanEps = 0.5f;
inline constexpr int kDefaultMinSamples = 5;
} // namespace mapper

namespace persistence
{
inline constexpr double kInfDeath = std::numeric_limits<double>::infinity();
inline constexpr float kDefaultSigma = 0.1f;
inline constexpr int kDefaultResolution = 100;
} // namespace persistence

namespace kernel
{
inline constexpr double kDefaultGaussianSigma = 1.0;
inline constexpr double kDefaultWassersteinP = 2.0;
inline constexpr int kDefaultNProjections = 100;
} // namespace kernel

namespace diagram
{
inline constexpr int kDefaultKernelSize = 5;
inline constexpr int kDefaultResolution = 20;
inline constexpr float kDefaultSigma = 0.1f;
inline constexpr int kDefaultNLayers = 5;
inline constexpr int kDefaultVectorDim = 64;
} // namespace diagram

// Type-Aware Constants

template <typename T>
struct Constants
{
    static constexpr T kZero = T(0);
    static constexpr T kOne = T(1);
    static constexpr T kTwo = T(2);
    static constexpr T kHalf = T(0.5);
    static constexpr T kEpsilon = epsilon<T>();
    static constexpr T kInf = std::numeric_limits<T>::infinity();
};

// Common type aliases
using FloatConstants = Constants<float>;
using DoubleConstants = Constants<double>;

} // namespace nerve::math
