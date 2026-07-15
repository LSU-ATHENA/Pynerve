
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/gpu_compute.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/symmetric_eigendecomposition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nerve::spectral
{

using Size = ::nerve::Size;

namespace
{

double orientedFaceSign(Size removed_vertex_index)
{
    return (removed_vertex_index % 2 == 0) ? 1.0 : -1.0;
}

Size dimensionCount(int max_dimension) noexcept
{
    if (max_dimension < 0)
    {
        return 0;
    }
    return static_cast<Size>(max_dimension) + 1;
}

Size doubledSize(Size value, const char *context)
{
    if (value > std::numeric_limits<Size>::max() / 2)
    {
        throw std::length_error(context);
    }
    return value * 2;
}

int simplexDimensionAsInt(Size dimension, const char *context)
{
    if (dimension > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(dimension);
}

} // anonymous namespace

namespace detail
{
errors::ErrorResult<void>
computeAllLaplaciansGpu(const std::vector<std::vector<double>> &boundary_matrix,
                        const std::vector<int> &simplex_dimensions, int max_dimension,
                        std::vector<std::vector<std::vector<double>>> &out_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_up_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_down_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_hodge_laplacians);
} // namespace detail

Laplacian::Laplacian()
    : size_(0)
    , max_dimension_(0)
{}

Laplacian::Laplacian(const SimplicialComplex &complex)
    : size_(0)
    , max_dimension_(0)
{
    buildFromComplex(complex);
}

Laplacian::Laplacian(const CellularComplex &complex)
    : size_(0)
    , max_dimension_(0)
{
    buildFromCellular(complex);
}

Laplacian::Laplacian(const SimplicialComplex &complex, const LaplacianConfig &config)
    : config_(config)
{
    buildBoundaryMatrices(complex);
    if (size_ >= config_.threshold && nerve::gpu::isAvailable())
    {
        computeAllLaplaciansGpu();
    }
    else
    {
        computeAllLaplacians();
    }
}

Laplacian::Laplacian(const CellularComplex &complex, const LaplacianConfig &config)
    : config_(config)
{
    buildBoundaryMatrices(complex);
    if (size_ >= config_.threshold && nerve::gpu::isAvailable())
    {
        computeAllLaplaciansGpu();
    }
    else
    {
        computeAllLaplacians();
    }
}

void Laplacian::buildFromComplex(const SimplicialComplex &complex)
{
    buildBoundaryMatrices(complex);
    computeAllLaplacians();
}

void Laplacian::buildFromCellular(const CellularComplex &complex)
{
    buildBoundaryMatrices(complex);
    computeAllLaplacians();
}

Size Laplacian::size() const
{
    return size_;
}

int Laplacian::maxDimension() const
{
    return max_dimension_;
}

std::vector<std::vector<double>> Laplacian::getLaplacian(int dimension) const
{
    if (dimension < 0 || static_cast<Size>(dimension) >= laplacians_.size())
    {
        throw std::out_of_range("Dimension out of range");
    }
    return laplacians_[static_cast<Size>(dimension)];
}

std::vector<std::vector<double>> Laplacian::getUpLaplacian(int dimension) const
{
    if (dimension < 0 || static_cast<Size>(dimension) >= up_laplacians_.size())
    {
        throw std::out_of_range("Dimension out of range");
    }
    return up_laplacians_[static_cast<Size>(dimension)];
}

std::vector<std::vector<double>> Laplacian::getDownLaplacian(int dimension) const
{
    if (dimension < 0 || static_cast<Size>(dimension) >= down_laplacians_.size())
    {
        throw std::out_of_range("Dimension out of range");
    }
    return down_laplacians_[static_cast<Size>(dimension)];
}

std::vector<std::vector<double>> Laplacian::getHodgeLaplacian(int dimension) const
{
    if (dimension < 0 || static_cast<Size>(dimension) >= hodge_laplacians_.size())
    {
        throw std::out_of_range("Dimension out of range");
    }
    return hodge_laplacians_[static_cast<Size>(dimension)];
}

std::vector<double> Laplacian::eigenvalues(int dimension, Size k) const
{
    auto laplacian = getLaplacian(dimension);
    std::vector<double> values;
    computeEigendecomposition(laplacian, values);
    if (k > 0 && k < values.size())
    {
        values.resize(k);
    }
    return values;
}

std::vector<std::vector<double>> Laplacian::eigenvectors(int dimension, Size k) const
{
    auto laplacian = getLaplacian(dimension);
    std::vector<double> values;
    auto vectors = computeEigendecomposition(laplacian, values);
    if (k > 0 && k < vectors.size())
    {
        vectors.resize(k);
    }
    return vectors;
}

std::vector<double> Laplacian::spectrum(int dimension) const
{
    return eigenvalues(dimension);
}

std::vector<std::vector<double>> Laplacian::computeEmbedding(int dimension, Size target_dim) const
{
    auto vectors = eigenvectors(dimension, target_dim);
    const Size n = vectors.size();
    const Size m = vectors.empty() ? 0 : vectors[0].size();
    std::vector<std::vector<double>> embedding(m, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < m; ++j)
        {
            embedding[j][i] = vectors[i][j];
        }
    }
    return embedding;
}

std::vector<std::vector<double>> Laplacian::computeDiffusionMap(Size target_dim) const
{
    const auto heat = heatKernel(0, 1.0);
    const Size n = heat.size();
    std::vector<std::vector<double>> diffusion_map;
    for (Size i = 1; i <= target_dim && i < n; ++i)
    {
        std::vector<double> component;
        component.reserve(n);
        for (Size j = 0; j < n; ++j)
        {
            component.push_back(heat[j][i]);
        }
        diffusion_map.push_back(std::move(component));
    }
    return diffusion_map;
}

std::vector<std::vector<double>> Laplacian::heatKernel(int dimension, double t) const
{
    auto laplacian = getLaplacian(dimension);
    std::vector<double> values;
    const auto vectors = computeEigendecomposition(laplacian, values);
    const Size n = laplacian.size();
    std::vector<std::vector<double>> heat(n, std::vector<double>(n, 0.0));
    const Size num_ev = std::min(n, static_cast<Size>(values.size()));
    // Batched SIMD.exp: fill buffer with -t * eigenvalues, then apply exp once
    std::vector<double> exp_terms(num_ev);
    for (Size i = 0; i < num_ev; ++i)
    {
        exp_terms[i] = -t * values[i];
    }
    nerve::simd::simd_exp(exp_terms.data(), num_ev);
    for (Size i = 0; i < num_ev; ++i)
    {
        const double exp_term = exp_terms[i];
        for (Size j = 0; j < n; ++j)
        {
            for (Size k = 0; k < n; ++k)
            {
                heat[j][k] += exp_term * vectors[i][j] * vectors[i][k];
            }
        }
    }
    return heat;
}

std::vector<std::vector<double>> Laplacian::heatFlow(const std::vector<double> &initial,
                                                     int dimension, double t) const
{
    const auto heat = heatKernel(dimension, t);
    const Size n = heat.size();
    std::vector<std::vector<double>> flow(n, std::vector<double>(1, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n && j < initial.size(); ++j)
        {
            flow[i][0] += heat[i][j] * initial[j];
        }
    }
    return flow;
}

std::vector<double> Laplacian::computeSpectralGap(int dimension) const
{
    auto values = spectrum(dimension);
    if (values.size() < 2)
    {
        return {0.0};
    }
    return {values[1] - values[0]};
}

std::vector<double> Laplacian::computeCheegerConstants() const
{
    const Size count = dimensionCount(max_dimension_);
    std::vector<double> cheeger(count, 0.0);
    for (Size dim = 0; dim < count; ++dim)
    {
        const auto gap = computeSpectralGap(static_cast<int>(dim));
        cheeger[dim] = std::sqrt(std::max(0.0, gap[0] / 2.0));
    }
    return cheeger;
}

std::vector<int> Laplacian::computeMorseIndex(int dimension) const
{
    const auto values = spectrum(dimension);
    int morse_index = 0;
    for (const auto value : values)
    {
        if (value < 0.0)
        {
            ++morse_index;
        }
    }
    return {morse_index};
}

void Laplacian::buildBoundaryMatrices(const SimplicialComplex &complex)
{
    auto simplices = complex.getSimplices();
    std::sort(simplices.begin(), simplices.end(),
              [](const algebra::Simplex &a, const algebra::Simplex &b) {
                  if (a.dimension() != b.dimension())
                  {
                      return a.dimension() < b.dimension();
                  }
                  return a < b;
              });

    size_ = simplices.size();
    simplex_dimensions_.assign(size_, 0);
    boundary_matrix_.assign(size_, std::vector<double>(size_, 0.0));
    coboundary_matrix_.assign(size_, std::vector<double>(size_, 0.0));

    std::unordered_map<algebra::Simplex, Size, algebra::Simplex::Hash> simplex_to_index;
    simplex_to_index.reserve(doubledSize(size_, "Laplacian simplex index reserve overflows"));
    max_dimension_ = -1;
    for (Size i = 0; i < simplices.size(); ++i)
    {
        simplex_to_index.emplace(simplices[i], i);
        simplex_dimensions_[i] = simplexDimensionAsInt(
            simplices[i].dimension(), "Laplacian simplex dimension exceeds int range");
        max_dimension_ = std::max(max_dimension_, simplex_dimensions_[i]);
    }

    for (Size col = 0; col < simplices.size(); ++col)
    {
        const auto &simplex = simplices[col];
        if (simplex.dimension() == 0)
        {
            continue;
        }
        const auto &vertices = simplex.vertices();
        for (Size removed = 0; removed < vertices.size(); ++removed)
        {
            std::vector<Index> face_vertices;
            face_vertices.reserve(vertices.size() - 1);
            for (Size idx = 0; idx < vertices.size(); ++idx)
            {
                if (idx != removed)
                {
                    face_vertices.push_back(vertices[idx]);
                }
            }
            const algebra::Simplex face(face_vertices);
            const auto it = simplex_to_index.find(face);
            if (it == simplex_to_index.end())
            {
                continue;
            }
            boundary_matrix_[it->second][col] = orientedFaceSign(removed);
        }
    }

    coboundary_matrix_ = computeMatrixTranspose(boundary_matrix_);
}

void Laplacian::buildBoundaryMatrices(const CellularComplex &complex)
{
    boundary_matrix_ = complex.computeBoundaryMatrix();
    coboundary_matrix_ = complex.computeCoboundaryMatrix();
    size_ = complex.numCells();
    max_dimension_ = complex.maxDimension();
    simplex_dimensions_.assign(size_, 0);
    for (Size i = 0; i < size_; ++i)
    {
        simplex_dimensions_[i] = complex.getCell(static_cast<Index>(i)).dimension();
    }
}

void Laplacian::computeAllLaplacians()
{
    const Size count = dimensionCount(max_dimension_);
    laplacians_.assign(count, {});
    up_laplacians_.assign(count, {});
    down_laplacians_.assign(count, {});
    hodge_laplacians_.assign(count, {});

    std::vector<std::vector<Size>> indices(count);
    for (Size i = 0; i < simplex_dimensions_.size(); ++i)
    {
        const int dim = simplex_dimensions_[i];
        if (dim >= 0 && static_cast<Size>(dim) < count)
        {
            indices[static_cast<Size>(dim)].push_back(i);
        }
    }

    for (Size dim = 0; dim < count; ++dim)
    {
        const auto &d_indices = indices[dim];
        const Size n = d_indices.size();
        std::vector<std::vector<double>> up(n, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> down(n, std::vector<double>(n, 0.0));

        if (dim + 1 < count)
        {
            const auto &dp1_indices = indices[dim + 1];
            const Size dp1_start = dp1_indices.empty() ? 0 : dp1_indices[0];
            const Size dp1_count = dp1_indices.size();
            // Contiguous columns: boundary row dot product via SIMD.dot
            for (Size i = 0; i < n; ++i)
            {
                for (Size j = 0; j < n; ++j)
                {
                    up[i][j] = nerve::simd::simd_dot(&boundary_matrix_[d_indices[i]][dp1_start],
                                                     &boundary_matrix_[d_indices[j]][dp1_start],
                                                     dp1_count);
                }
            }
        }

        if (dim > 0)
        {
            const auto &dm1_indices = indices[dim - 1];
            const Size d_start = d_indices.empty() ? 0 : d_indices[0];
            // Loop-interchange: for each row, axpy onto down[i][j..]
            // down[i][j] += B[row][d_i] * B[row][d_j] over rows
            // B[row][d_j] is contiguous for consecutive d_j (same dimension)
            for (const auto row : dm1_indices)
            {
                const double *Brow = boundary_matrix_[row].data();
                for (Size i = 0; i < n; ++i)
                {
                    const double bi = Brow[d_indices[i]];
                    if (bi == 0.0)
                        continue;
                    nerve::simd::simd_axpy(bi, &Brow[d_start], &down[i][0], n);
                }
            }
        }

        std::vector<std::vector<double>> hodge(n, std::vector<double>(n, 0.0));
        for (Size i = 0; i < n; ++i)
        {
            for (Size j = 0; j < n; ++j)
            {
                hodge[i][j] = up[i][j] + down[i][j];
            }
        }

        up_laplacians_[dim] = std::move(up);
        down_laplacians_[dim] = std::move(down);
        hodge_laplacians_[dim] = hodge;
        laplacians_[dim] = std::move(hodge);
    }
}

void Laplacian::computeAllLaplaciansGpu()
{
#ifdef __CUDACC__
    auto result = detail::computeAllLaplaciansGpu(boundary_matrix_, simplex_dimensions_,
                                                  max_dimension_, laplacians_, up_laplacians_,
                                                  down_laplacians_, hodge_laplacians_);
    if (result.isError())
    {
        computeAllLaplacians();
    }
#else
    computeAllLaplacians();
#endif
}

std::vector<std::vector<double>>
Laplacian::computeMatrixProduct(const std::vector<std::vector<double>> &a,
                                const std::vector<std::vector<double>> &b) const
{
    const Size n = a.size();
    const Size m = b.empty() ? 0 : b[0].size();
    const Size p = a.empty() ? 0 : a[0].size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < m; ++j)
        {
            for (Size k = 0; k < p; ++k)
            {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>>
Laplacian::computeMatrixTranspose(const std::vector<std::vector<double>> &a) const
{
    const Size n = a.size();
    const Size m = a.empty() ? 0 : a[0].size();
    std::vector<std::vector<double>> transpose(m, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < m; ++j)
        {
            transpose[j][i] = a[i][j];
        }
    }
    return transpose;
}

std::vector<std::vector<double>>
Laplacian::computeEigendecomposition(const std::vector<std::vector<double>> &matrix,
                                     std::vector<double> &eigenvalues) const
{
    const auto result = detail::jacobiEigendecomposition(matrix);
    eigenvalues = result.eigenvalues;
    return result.eigenvectors;
}

} // namespace nerve::spectral
