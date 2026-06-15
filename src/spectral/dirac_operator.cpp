
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/symmetric_eigendecomposition.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nerve::spectral
{

namespace
{

double orientedFaceSign(Size removed_vertex_index)
{
    return (removed_vertex_index % 2 == 0) ? 1.0 : -1.0;
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

std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>> &matrix)
{
    const Size n = matrix.size();
    const Size m = matrix.empty() ? 0 : matrix[0].size();
    std::vector<std::vector<double>> transpose(m, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < m; ++j)
        {
            transpose[j][i] = matrix[i][j];
        }
    }
    return transpose;
}

} // namespace

DiracOperator::DiracOperator()
    : size_(0)
    , max_dimension_(0)
{}

DiracOperator::DiracOperator(const SimplicialComplex &complex)
    : size_(0)
    , max_dimension_(0)
{
    buildFromComplex(complex);
}

DiracOperator::DiracOperator(const CellularComplex &complex)
    : size_(0)
    , max_dimension_(0)
{
    buildFromCellular(complex);
}

void DiracOperator::buildFromComplex(const SimplicialComplex &complex)
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
    boundary_matrix_.assign(size_, std::vector<double>(size_, 0.0));

    std::unordered_map<algebra::Simplex, Size, algebra::Simplex::Hash> simplex_to_index;
    simplex_to_index.reserve(doubledSize(size_, "Dirac simplex index reserve overflows"));
    max_dimension_ = -1;
    for (Size i = 0; i < simplices.size(); ++i)
    {
        simplex_to_index.emplace(simplices[i], i);
        max_dimension_ = std::max(
            max_dimension_, simplexDimensionAsInt(simplices[i].dimension(),
                                                  "Dirac simplex dimension exceeds int range"));
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
            if (it != simplex_to_index.end())
            {
                boundary_matrix_[it->second][col] = orientedFaceSign(removed);
            }
        }
    }

    coboundary_matrix_ = transposeMatrix(boundary_matrix_);
    buildDiracMatrix();
}

void DiracOperator::buildFromCellular(const CellularComplex &complex)
{
    boundary_matrix_ = complex.computeBoundaryMatrix();
    coboundary_matrix_ = complex.computeCoboundaryMatrix();
    size_ = complex.numCells();
    max_dimension_ = complex.maxDimension();
    buildDiracMatrix();
}

std::vector<std::vector<double>> DiracOperator::getDirac() const
{
    return dirac_matrix_;
}

std::vector<std::vector<double>> DiracOperator::getDiracSquared() const
{
    return dirac_squared_matrix_;
}

std::vector<double> DiracOperator::eigenvalues(Size k) const
{
    std::vector<double> values;
    computeEigendecomposition(dirac_matrix_, values);
    if (k > 0 && k < values.size())
    {
        values.resize(k);
    }
    return values;
}

std::vector<std::vector<double>> DiracOperator::eigenvectors(Size k) const
{
    std::vector<double> values;
    auto vectors = computeEigendecomposition(dirac_matrix_, values);
    if (k > 0 && k < vectors.size())
    {
        vectors.resize(k);
    }
    return vectors;
}

std::vector<std::vector<std::complex<double>>> DiracOperator::getSpinorLaplacian() const
{
    const Size n = dirac_squared_matrix_.size();
    std::vector<std::vector<std::complex<double>>> spinorLaplacian(
        n, std::vector<std::complex<double>>(n, std::complex<double>(0.0, 0.0)));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            spinorLaplacian[i][j] = std::complex<double>(dirac_squared_matrix_[i][j], 0.0);
        }
    }
    return spinorLaplacian;
}

std::vector<std::vector<std::complex<double>>> DiracOperator::getChiralityOperator() const
{
    return buildChiralityOperator();
}

int DiracOperator::computeAtiyahSingerIndex() const
{
    const auto values = eigenvalues();
    int positive_count = 0;
    int negative_count = 0;
    constexpr double kTolerance = 1e-10;
    for (const auto value : values)
    {
        if (value > kTolerance)
        {
            ++positive_count;
        }
        else if (value < -kTolerance)
        {
            ++negative_count;
        }
    }
    return positive_count - negative_count;
}

std::vector<int> DiracOperator::computeAnalyticalIndex() const
{
    return {computeAtiyahSingerIndex()};
}

std::vector<int> DiracOperator::computeTopologicalIndex() const
{
    return {computeAtiyahSingerIndex()};
}

void DiracOperator::buildDiracMatrix()
{
    const Size n = boundary_matrix_.size();
    const Size m = doubledSize(n, "Dirac matrix dimension overflows");
    dirac_matrix_.assign(m, std::vector<double>(m, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            const double cob = (i < coboundary_matrix_.size() && j < coboundary_matrix_[i].size())
                                   ? coboundary_matrix_[i][j]
                                   : 0.0;
            const double bnd = (i < boundary_matrix_.size() && j < boundary_matrix_[i].size())
                                   ? boundary_matrix_[i][j]
                                   : 0.0;
            dirac_matrix_[i][n + j] = cob;
            dirac_matrix_[n + i][j] = bnd;
        }
    }
    dirac_squared_matrix_ = computeMatrixProduct(dirac_matrix_, dirac_matrix_);
}

std::vector<std::vector<double>>
DiracOperator::computeMatrixProduct(const std::vector<std::vector<double>> &a,
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
DiracOperator::computeEigendecomposition(const std::vector<std::vector<double>> &matrix,
                                         std::vector<double> &eigenvalues) const
{
    const auto result = detail::jacobiEigendecomposition(matrix);
    eigenvalues = result.eigenvalues;
    return result.eigenvectors;
}

std::vector<std::vector<std::complex<double>>> DiracOperator::buildChiralityOperator() const
{
    const Size n = boundary_matrix_.size();
    const Size m = doubledSize(n, "Dirac chirality matrix dimension overflows");
    std::vector<std::vector<std::complex<double>>> chirality(
        m, std::vector<std::complex<double>>(m, std::complex<double>(0.0, 0.0)));
    for (Size i = 0; i < n; ++i)
    {
        chirality[i][n + i] = std::complex<double>(1.0, 0.0);
        chirality[n + i][i] = std::complex<double>(1.0, 0.0);
    }
    return chirality;
}

} // namespace nerve::spectral
