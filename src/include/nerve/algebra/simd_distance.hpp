#pragma once
#include <cstddef>
#include <vector>

namespace nerve::algebra
{

/// SIMD-accelerated distance calculator.
///
/// All distance computations delegate to the global SIMD dispatch table
/// (nerve::simd::SIMD), which selects the best available ISA at runtime
/// via a single simd_init() call. No per-instance CPU feature detection
/// or inline intrinsics are needed.
class SIMDDistanceCalculator
{
public:
    SIMDDistanceCalculator();

    [[nodiscard]] double euclideanDistance(const double *a, const double *b, size_t dim);
    [[nodiscard]] double manhattanDistance(const double *a, const double *b, size_t dim);
    [[nodiscard]] double cosineDistance(const double *a, const double *b, size_t dim);

    [[nodiscard]] std::vector<double> batchEuclideanDistances(const double *points,
                                                              size_t num_points, size_t dim);

    // Runtime capability queries -- reflect what the dispatch table selected.
    [[nodiscard]] bool hasAvx512() const noexcept;
    [[nodiscard]] bool hasAvx2() const noexcept;
    [[nodiscard]] bool hasFma() const noexcept;

    // Pure scalar implementations (no SIMD) for deterministic comparison.
    [[nodiscard]] double euclideanDistanceScalar(const double *a, const double *b, size_t dim);
    [[nodiscard]] double manhattanDistanceScalar(const double *a, const double *b, size_t dim);
    [[nodiscard]] double cosineDistanceScalar(const double *a, const double *b, size_t dim);
};

} // namespace nerve::algebra
