
#pragma once
#include <cstddef>
#include <vector>

namespace nerve::algebra
{

class SIMDDistanceCalculator
{
public:
    SIMDDistanceCalculator();

    [[nodiscard]] double euclideanDistance(const double *a, const double *b, size_t dim);
    [[nodiscard]] double manhattanDistance(const double *a, const double *b, size_t dim);
    [[nodiscard]] double cosineDistance(const double *a, const double *b, size_t dim);

    [[nodiscard]] std::vector<double> batchEuclideanDistances(const double *points,
                                                              size_t num_points, size_t dim);

    [[nodiscard]] bool hasAvx512() const noexcept { return has_avx512_; }
    [[nodiscard]] bool hasAvx2() const noexcept { return has_avx2_; }
    [[nodiscard]] bool hasFma() const noexcept { return has_fma_; }

    [[nodiscard]] double euclideanDistanceScalar(const double *a, const double *b, size_t dim);
    [[nodiscard]] double manhattanDistanceScalar(const double *a, const double *b, size_t dim);
    [[nodiscard]] double cosineDistanceScalar(const double *a, const double *b, size_t dim);

private:
    bool has_avx512_;
    bool has_avx2_;
    bool has_fma_;

    void detectCpuFeatures();

    double euclideanDistanceAvx512(const double *a, const double *b, size_t dim);
    double euclideanDistanceAvx2(const double *a, const double *b, size_t dim);

    void batchEuclideanDistancesAvx512(const double *points, size_t num_points, size_t dim,
                                       std::vector<double> &distances);
    void batchEuclideanDistancesAvx2(const double *points, size_t num_points, size_t dim,
                                     std::vector<double> &distances);
    void batchEuclideanDistancesScalar(const double *points, size_t num_points, size_t dim,
                                       std::vector<double> &distances);

    double manhattanDistanceAvx512(const double *a, const double *b, size_t dim);

    double cosineDistanceAvx512(const double *a, const double *b, size_t dim);
};

} // namespace nerve::algebra
