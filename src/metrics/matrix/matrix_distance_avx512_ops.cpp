#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/metrics/gpu_distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::metrics::avx512
{
namespace
{

constexpr int kSimdFloatWidth = 16;
constexpr int kTileSize = 64;

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

std::size_t checkedMatrixSize(int n1, int n2)
{
    if (n1 < 0 || n2 < 0)
    {
        throw std::invalid_argument("diagram matrix sizes must be non-negative");
    }

    const auto rows = static_cast<std::size_t>(n1);
    const auto cols = static_cast<std::size_t>(n2);
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows)
    {
        throw std::length_error("diagram matrix size overflow");
    }

    return rows * cols;
}

void validateDiagram(const float *birth, const float *death, int count, const char *name)
{
    if (count == 0)
    {
        return;
    }

    if (birth == nullptr || death == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " diagram arrays must not be null");
    }

    const float safe_abs = std::numeric_limits<float>::max() / 4.0F;
    for (int i = 0; i < count; ++i)
    {
        if (!std::isfinite(birth[i]) || !std::isfinite(death[i]) || death[i] < birth[i] ||
            std::abs(birth[i]) > safe_abs || std::abs(death[i]) > safe_abs)
        {
            throw std::invalid_argument(std::string(name) +
                                        " diagram contains invalid persistence pairs");
        }
    }
}

std::size_t validateDiagramDistanceInputs(const float *birth1, const float *death1, int n1,
                                          const float *birth2, const float *death2, int n2,
                                          float *out_matrix)
{
    const std::size_t matrix_size = checkedMatrixSize(n1, n2);
    validateDiagram(birth1, death1, n1, "first");
    validateDiagram(birth2, death2, n2, "second");

    if (matrix_size != 0 && out_matrix == nullptr)
    {
        throw std::invalid_argument("output matrix must not be null");
    }

    return matrix_size;
}

std::size_t matrixOffset(int row, int cols, int col)
{
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
           static_cast<std::size_t>(col);
}

void computeDiagramDistanceMatrixScalar(const float *birth1, const float *death1, int n1,
                                        const float *birth2, const float *death2, int n2,
                                        float *out_matrix)
{
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            const float birth_diff = std::abs(birth1[i] - birth2[j]);
            const float death_diff = std::abs(death1[i] - death2[j]);
            out_matrix[matrixOffset(i, n2, j)] = std::max(birth_diff, death_diff);
        }
    }
}

void computeDiagramDistanceMatrixAvx512(const float *birth1, const float *death1, int n1,
                                        const float *birth2, const float *death2, int n2,
                                        float *out_matrix)
{
    for (int i_tile = 0; i_tile < n1; i_tile += kTileSize)
    {
        const int i_end = std::min(i_tile + kTileSize, n1);

        for (int j_tile = 0; j_tile < n2; j_tile += kTileSize)
        {
            const int j_end = std::min(j_tile + kTileSize, n2);
            for (int i = i_tile; i < i_end; ++i)
            {
                const __m512 birth_i = _mm512_set1_ps(birth1[i]);
                const __m512 death_i = _mm512_set1_ps(death1[i]);

                int j = j_tile;
                for (; j + kSimdFloatWidth <= j_end; j += kSimdFloatWidth)
                {
                    const __m512 birth_j = _mm512_loadu_ps(birth2 + j);
                    const __m512 death_j = _mm512_loadu_ps(death2 + j);
                    __m512 birth_diff = _mm512_sub_ps(birth_i, birth_j);
                    birth_diff = _mm512_abs_ps(birth_diff);
                    __m512 death_diff = _mm512_sub_ps(death_i, death_j);
                    death_diff = _mm512_abs_ps(death_diff);
                    const __m512 dist = _mm512_max_ps(birth_diff, death_diff);

                    _mm512_storeu_ps(out_matrix + matrixOffset(i, n2, j), dist);
                }

                for (; j < j_end; ++j)
                {
                    const float birth_diff = std::abs(birth1[i] - birth2[j]);
                    const float death_diff = std::abs(death1[i] - death2[j]);
                    out_matrix[matrixOffset(i, n2, j)] = std::max(birth_diff, death_diff);
                }
            }
        }
    }
}

void validateBenchmarkInputs(int n1, int n2, int dim, int iterations)
{
    if (dim < 0 || iterations < 0)
    {
        throw std::invalid_argument("benchmark dimension and iteration count must be non-negative");
    }
    checkedMatrixSize(n1, n2);
}

} // namespace

void avx512DiagramDistanceMatrix(const float *birth1, const float *death1, int n1,
                                 const float *birth2, const float *death2, int n2,
                                 float *out_matrix)
{
    validateDiagramDistanceInputs(birth1, death1, n1, birth2, death2, n2, out_matrix);
    computeDiagramDistanceMatrixAvx512(birth1, death1, n1, birth2, death2, n2, out_matrix);
}

AVX512Benchmark benchmarkAVX512(int n1, int n2, int dim, int iterations)
{
    validateBenchmarkInputs(n1, n2, dim, iterations);

    AVX512Benchmark bench{.scalar_time_ms = 0.0,
                          .avx512_time_ms = 0.0,
                          .speedup = 1.0,
                          .n1 = n1,
                          .n2 = n2,
                          .dim = dim};

    const std::size_t matrix_size = checkedMatrixSize(n1, n2);
    if (iterations == 0 || matrix_size == 0)
    {
        return bench;
    }

    std::vector<float> birth1(static_cast<std::size_t>(n1)), death1(static_cast<std::size_t>(n1));
    std::vector<float> birth2(static_cast<std::size_t>(n2)), death2(static_cast<std::size_t>(n2));
    std::vector<float> matrix(matrix_size);

    for (int i = 0; i < n1; ++i)
    {
        birth1[i] = static_cast<float>(i);
        death1[i] = static_cast<float>(i + 10);
    }
    for (int i = 0; i < n2; ++i)
    {
        birth2[i] = static_cast<float>(i);
        death2[i] = static_cast<float>(i + 10);
    }

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        computeDiagramDistanceMatrixScalar(birth1.data(), death1.data(), n1, birth2.data(),
                                           death2.data(), n2, matrix.data());
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    bench.scalar_time_ms =
        std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    auto start_avx = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        computeDiagramDistanceMatrixAvx512(birth1.data(), death1.data(), n1, birth2.data(),
                                           death2.data(), n2, matrix.data());
    }
    auto end_avx = std::chrono::high_resolution_clock::now();
    bench.avx512_time_ms = std::chrono::duration<double, std::milli>(end_avx - start_avx).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.scalar_time_ms, bench.avx512_time_ms);

    return bench;
}

bool isAVX512Available()
{
#ifdef __AVX512F__
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void computeDistanceMatrixAuto(const float *birth1, const float *death1, int n1,
                               const float *birth2, const float *death2, int n2, float *out_matrix)
{
    validateDiagramDistanceInputs(birth1, death1, n1, birth2, death2, n2, out_matrix);

    if (isAVX512Available() && n2 >= 16)
    {
        computeDiagramDistanceMatrixAvx512(birth1, death1, n1, birth2, death2, n2, out_matrix);
    }
    else
    {
        computeDiagramDistanceMatrixScalar(birth1, death1, n1, birth2, death2, n2, out_matrix);
    }
}

} // namespace nerve::metrics::avx512
