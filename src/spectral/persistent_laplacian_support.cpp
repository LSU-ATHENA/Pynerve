#include "nerve/spectral/persistent_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

namespace nerve::spectral
{
namespace
{

constexpr double kDefaultZeroTolerance = 1e-10;

double zeroToleranceFor(const SpectralConfig &config)
{
    return std::max(config.convergence_tolerance, kDefaultZeroTolerance);
}

Eigen::VectorXd normalizedOrCanonical(const Eigen::VectorXd &input)
{
    Eigen::VectorXd out = input;
    const double norm = out.norm();
    if (norm <= std::numeric_limits<double>::epsilon())
    {
        if (out.size() > 0)
        {
            out.setZero();
            out[0] = 1.0;
        }
        return out;
    }
    out /= norm;
    return out;
}

} // namespace

Eigen::SparseMatrix<double> PersistentLaplacianSolver::buildPersistentLaplacian(
    const std::vector<std::vector<uint32_t>> &boundary_matrix,
    const std::vector<double> &filtration_values)
{
    const std::size_t n = std::max(boundary_matrix.size(), filtration_values.size());
    if (n == 0U)
    {
        return {};
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 4U);
    for (std::size_t row = 0; row < boundary_matrix.size(); ++row)
    {
        for (uint32_t col : boundary_matrix[row])
        {
            if (col < n)
            {
                triplets.emplace_back(static_cast<int>(row), static_cast<int>(col), 1.0);
            }
        }
    }

    Eigen::SparseMatrix<double> boundary(static_cast<int>(n), static_cast<int>(n));
    if (!triplets.empty())
    {
        boundary.setFromTriplets(triplets.begin(), triplets.end());
    }
    Eigen::SparseMatrix<double> laplacian = boundary.transpose() * boundary;

    if (!filtration_values.empty())
    {
        std::vector<Eigen::Triplet<double>> diag;
        diag.reserve(filtration_values.size());
        for (std::size_t i = 0; i < filtration_values.size() && i < n; ++i)
        {
            diag.emplace_back(static_cast<int>(i), static_cast<int>(i), filtration_values[i]);
        }
        Eigen::SparseMatrix<double> filtration_diag(static_cast<int>(n), static_cast<int>(n));
        filtration_diag.setFromTriplets(diag.begin(), diag.end());
        laplacian += filtration_diag;
    }
    laplacian.prune(0.0);
    return laplacian;
}

Eigen::SparseMatrix<double> PersistentLaplacianSolver::buildDiracOperator(
    const std::vector<std::vector<uint32_t>> &boundary_matrix)
{
    const std::size_t n = boundary_matrix.size();
    if (n == 0U)
    {
        return {};
    }

    std::vector<Eigen::Triplet<double>> boundary_triplets;
    boundary_triplets.reserve(n * 4U);
    for (std::size_t row = 0; row < boundary_matrix.size(); ++row)
    {
        for (uint32_t col : boundary_matrix[row])
        {
            if (col < n)
            {
                boundary_triplets.emplace_back(static_cast<int>(row), static_cast<int>(col), 1.0);
            }
        }
    }

    Eigen::SparseMatrix<double> boundary(static_cast<int>(n), static_cast<int>(n));
    if (!boundary_triplets.empty())
    {
        boundary.setFromTriplets(boundary_triplets.begin(), boundary_triplets.end());
    }

    std::vector<Eigen::Triplet<double>> dirac_triplets;
    dirac_triplets.reserve(boundary.nonZeros() * 2U);
    for (int outer = 0; outer < boundary.outerSize(); ++outer)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(boundary, outer); it; ++it)
        {
            dirac_triplets.emplace_back(it.row(), static_cast<int>(n) + it.col(), it.value());
            dirac_triplets.emplace_back(static_cast<int>(n) + it.col(), it.row(), it.value());
        }
    }

    Eigen::SparseMatrix<double> dirac(static_cast<int>(2U * n), static_cast<int>(2U * n));
    if (!dirac_triplets.empty())
    {
        dirac.setFromTriplets(dirac_triplets.begin(), dirac_triplets.end());
    }
    dirac.prune(0.0);
    return dirac;
}

std::vector<Eigenpair>
PersistentLaplacianSolver::extractWarmStartBasis(const SpectralDecomposition &previous_result,
                                                 const Eigen::SparseMatrix<double> &new_laplacian)
{
    std::vector<Eigenpair> basis;
    const Eigen::Index expected_size = new_laplacian.rows();
    if (expected_size <= 0)
    {
        return basis;
    }

    const std::size_t max_keep =
        std::min(config_.num_eigenpairs, previous_result.eigenpairs.size());
    basis.reserve(max_keep);
    for (const Eigenpair &pair : previous_result.eigenpairs)
    {
        if (basis.size() >= max_keep)
        {
            break;
        }
        if (pair.eigenvector.size() != expected_size)
        {
            continue;
        }
        Eigenpair normalized_pair = pair;
        normalized_pair.eigenvector = normalizedOrCanonical(pair.eigenvector);
        basis.push_back(std::move(normalized_pair));
    }
    return basis;
}

bool PersistentLaplacianSolver::validateDecomposition(const SpectralDecomposition &result,
                                                      const Eigen::SparseMatrix<double> &laplacian)
{
    if (result.eigenpairs.empty() || laplacian.rows() != laplacian.cols())
    {
        return false;
    }
    if (computeOrthogonalityError(result.eigenpairs) > 1e-2)
    {
        return false;
    }

    const double residual_limit = std::max(1e-6, config_.convergence_tolerance * 10.0);
    for (const Eigenpair &pair : result.eigenpairs)
    {
        if (pair.eigenvector.size() != laplacian.rows())
        {
            return false;
        }
        if (computeResidualNorm(laplacian, pair) > residual_limit)
        {
            return false;
        }
    }
    return true;
}

std::vector<size_t>
PersistentLaplacianSolver::findHarmonicForms(const Eigen::SparseMatrix<double> &laplacian)
{
    const SpectralDecomposition spectrum = solveDirect(laplacian);
    const double zero_tol = zeroToleranceFor(config_);
    std::vector<size_t> indices;
    for (std::size_t i = 0; i < spectrum.eigenpairs.size(); ++i)
    {
        if (std::abs(spectrum.eigenpairs[i].eigenvalue) <= zero_tol)
        {
            indices.push_back(i);
        }
    }
    return indices;
}

double
PersistentLaplacianSolver::computeOrthogonalityError(const std::vector<Eigenpair> &eigenpairs)
{
    if (eigenpairs.size() < 2)
    {
        return 0.0;
    }
    const Eigen::Index n = eigenpairs.front().eigenvector.size();
    if (n <= 0)
    {
        return 0.0;
    }

    Eigen::MatrixXd vectors(n, static_cast<Eigen::Index>(eigenpairs.size()));
    for (std::size_t i = 0; i < eigenpairs.size(); ++i)
    {
        if (eigenpairs[i].eigenvector.size() != n)
        {
            return std::numeric_limits<double>::infinity();
        }
        vectors.col(static_cast<Eigen::Index>(i)) =
            normalizedOrCanonical(eigenpairs[i].eigenvector);
    }

    const Eigen::MatrixXd gram = vectors.transpose() * vectors;
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(gram.rows(), gram.cols());
    return (gram - identity).norm();
}

double PersistentLaplacianSolver::computeResidualNorm(const Eigen::SparseMatrix<double> &laplacian,
                                                      const Eigenpair &eigenpair)
{
    if (eigenpair.eigenvector.size() != laplacian.rows() || eigenpair.eigenvector.size() == 0)
    {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::VectorXd residual =
        (laplacian * eigenpair.eigenvector) - (eigenpair.eigenvalue * eigenpair.eigenvector);
    const double denom =
        std::max(eigenpair.eigenvector.norm(), std::numeric_limits<double>::epsilon());
    return residual.norm() / denom;
}

} // namespace nerve::spectral
