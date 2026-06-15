#include "nerve/spectral/persistent_laplacian.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace nerve::spectral
{
namespace
{

constexpr double kDefaultZeroTolerance = 1e-10;
constexpr std::uint32_t kDeterministicLanczosSeed = 42U;

double zeroToleranceFor(const SpectralConfig &config)
{
    return std::max(config.convergence_tolerance, kDefaultZeroTolerance);
}

Eigen::VectorXd makeDeterministicUnitVector(std::int64_t size)
{
    Eigen::VectorXd out(size);
    std::mt19937_64 rng(kDeterministicLanczosSeed);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    for (std::int64_t i = 0; i < size; ++i)
    {
        out[i] = gaussian(rng);
    }
    const double norm = out.norm();
    if (norm <= std::numeric_limits<double>::epsilon())
    {
        out.setZero();
        if (size > 0)
        {
            out[0] = 1.0;
        }
        return out;
    }
    out /= norm;
    return out;
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

double sparseTrace(const Eigen::SparseMatrix<double> &matrix)
{
    double trace = 0.0;
    const std::int64_t n = std::min(matrix.rows(), matrix.cols());
    for (std::int64_t i = 0; i < n; ++i)
    {
        trace += matrix.coeff(i, i);
    }
    return trace;
}

double sparseFrobeniusNorm(const Eigen::SparseMatrix<double> &matrix)
{
    double norm_sq = 0.0;
    for (int outer = 0; outer < matrix.outerSize(); ++outer)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, outer); it; ++it)
        {
            norm_sq += it.value() * it.value();
        }
    }
    return std::sqrt(norm_sq);
}

double participationRatio(const Eigen::VectorXd &vector)
{
    if (vector.size() == 0)
    {
        return 0.0;
    }
    double sum_sq = 0.0;
    double sum_fourth = 0.0;
    for (Eigen::Index i = 0; i < vector.size(); ++i)
    {
        const double v_sq = vector[i] * vector[i];
        sum_sq += v_sq;
        sum_fourth += v_sq * v_sq;
    }
    if (sum_fourth <= std::numeric_limits<double>::epsilon())
    {
        return 0.0;
    }
    const double n = static_cast<double>(vector.size());
    return (sum_sq * sum_sq) / (n * sum_fourth);
}

void sortEigenpairs(std::vector<Eigenpair> &pairs)
{
    std::sort(pairs.begin(), pairs.end(),
              [](const Eigenpair &a, const Eigenpair &b) { return a.eigenvalue < b.eigenvalue; });
}

void markHarmonicAndGaps(std::vector<Eigenpair> &pairs, double zero_tol)
{
    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        pairs[i].is_harmonic = std::abs(pairs[i].eigenvalue) <= zero_tol;
        const double next_value =
            (i + 1 < pairs.size()) ? pairs[i + 1].eigenvalue : pairs[i].eigenvalue;
        pairs[i].spectral_gap = std::max(0.0, next_value - pairs[i].eigenvalue);
    }
}

} // namespace

PersistentLaplacianSolver::PersistentLaplacianSolver(const SpectralConfig &config)
    : config_(config)
{}

SpectralDecomposition
PersistentLaplacianSolver::computeSpectrum(const Eigen::SparseMatrix<double> &laplacian)
{
    const auto started_at = std::chrono::steady_clock::now();
    SpectralDecomposition result;
    result.matrix_size = static_cast<std::size_t>(laplacian.rows());
    result.trace = sparseTrace(laplacian);
    result.frobenius_norm = sparseFrobeniusNorm(laplacian);

    if (laplacian.rows() == 0 || laplacian.cols() == 0 || laplacian.rows() != laplacian.cols())
    {
        result.converged = false;
        result.solver_used = "cpu:invalid-input";
        return result;
    }

    result = solveLanczos(laplacian);

    sortEigenpairs(result.eigenpairs);
    const double zero_tol = zeroToleranceFor(config_);
    markHarmonicAndGaps(result.eigenpairs, zero_tol);
    result.num_harmonic = 0U;
    result.harmonic_eigenpairs.clear();
    result.nonharmonic_eigenpairs.clear();
    for (const Eigenpair &pair : result.eigenpairs)
    {
        if (std::abs(pair.eigenvalue) <= zero_tol)
        {
            ++result.num_harmonic;
            result.harmonic_eigenpairs.push_back(pair);
        }
        else
        {
            result.nonharmonic_eigenpairs.push_back(pair);
        }
    }

    result.spectral_radius = 0.0;
    double min_positive = std::numeric_limits<double>::infinity();
    for (const Eigenpair &pair : result.eigenpairs)
    {
        result.spectral_radius = std::max(result.spectral_radius, std::abs(pair.eigenvalue));
        if (pair.eigenvalue > zero_tol)
        {
            min_positive = std::min(min_positive, pair.eigenvalue);
        }
    }
    result.condition_number = std::isfinite(min_positive) && min_positive > 0.0
                                  ? result.spectral_radius / min_positive
                                  : 0.0;
    result.orthogonality_error = computeOrthogonalityError(result.eigenpairs);
    result.residual_norm = 0.0;
    for (const Eigenpair &pair : result.eigenpairs)
    {
        result.residual_norm = std::max(result.residual_norm, computeResidualNorm(laplacian, pair));
    }

    const auto finished_at = std::chrono::steady_clock::now();
    result.computation_time_ms =
        std::chrono::duration<double, std::milli>(finished_at - started_at).count();
    return result;
}

SpectralDecomposition PersistentLaplacianSolver::computeSpectrumWithWarmStart(
    const Eigen::SparseMatrix<double> &laplacian, const SpectralDecomposition &previous_result)
{
    const std::vector<Eigenpair> warm_start = extractWarmStartBasis(previous_result, laplacian);
    if (warm_start.empty())
    {
        return computeSpectrum(laplacian);
    }
    return solveLanczos(laplacian, warm_start);
}

SpectralDecomposition
PersistentLaplacianSolver::computeHarmonicSpectrum(const Eigen::SparseMatrix<double> &laplacian)
{
    SpectralDecomposition result = computeSpectrum(laplacian);
    result.eigenpairs = result.harmonic_eigenpairs;
    return result;
}

SpectralDecomposition
PersistentLaplacianSolver::computeNonharmonicSpectrum(const Eigen::SparseMatrix<double> &laplacian)
{
    SpectralDecomposition result = computeSpectrum(laplacian);
    result.eigenpairs = result.nonharmonic_eigenpairs;
    return result;
}

SpectralDecomposition
PersistentLaplacianSolver::solveDirect(const Eigen::SparseMatrix<double> &laplacian)
{
    SpectralDecomposition result;
    result.solver_used = "cpu:direct";
    const Eigen::Index n = laplacian.rows();
    if (n <= 0)
    {
        result.converged = false;
        return result;
    }

    const auto n_size = static_cast<std::size_t>(n);
    const int k =
        static_cast<int>(std::min({config_.num_eigenpairs, n_size,
                                   static_cast<std::size_t>(std::numeric_limits<int>::max())}));
    if (k <= 0)
    {
        result.converged = false;
        return result;
    }

    Eigen::MatrixXd dense = Eigen::MatrixXd(laplacian);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense);
    if (eigensolver.info() != Eigen::Success)
    {
        result.converged = false;
        return result;
    }

    result.eigenpairs.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i)
    {
        Eigenpair pair{};
        pair.eigenvalue = eigensolver.eigenvalues()[i];
        pair.eigenvector = eigensolver.eigenvectors().col(i);
        pair.error_estimate = computeResidualNorm(laplacian, pair);
        pair.iterations = 1U;
        pair.participation_ratio = participationRatio(pair.eigenvector);
        pair.complex_eigenvalue = std::complex<double>(pair.eigenvalue, 0.0);
        pair.complex_eigenvector = pair.eigenvector.cast<std::complex<double>>();
        result.eigenpairs.push_back(std::move(pair));
    }
    result.converged = true;
    return result;
}

SpectralDecomposition
PersistentLaplacianSolver::solveArnoldi(const Eigen::SparseMatrix<double> &laplacian,
                                        const std::vector<Eigenpair> &warm_start)
{
    // Laplacians are symmetric positive semidefinite; dispatch to Lanczos for
    // stable Hermitian Krylov behavior.
    SpectralDecomposition result = solveLanczos(laplacian, warm_start);
    result.solver_used = "cpu:arnoldi(lanczos-dispatch)";
    return result;
}

SpectralDecomposition
PersistentLaplacianSolver::solveLanczos(const Eigen::SparseMatrix<double> &laplacian,
                                        const std::vector<Eigenpair> &warm_start)
{
    SpectralDecomposition result;
    result.solver_used = "cpu:lanczos";

    const Eigen::Index n = laplacian.rows();
    if (n <= 0)
    {
        result.converged = false;
        return result;
    }

    const auto n_size = static_cast<std::size_t>(n);
    const int capped_n = static_cast<int>(
        std::min(n_size, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int max_iterations = static_cast<int>(std::min(
        config_.max_iterations, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int max_steps = std::max<int>(1, std::min(max_iterations, capped_n));
    const int target =
        static_cast<int>(std::min({config_.num_eigenpairs, n_size,
                                   static_cast<std::size_t>(std::numeric_limits<int>::max())}));
    if (target <= 0)
    {
        result.converged = false;
        return result;
    }

    std::vector<Eigen::VectorXd> basis;
    basis.reserve(static_cast<std::size_t>(max_steps + 1));
    if (!warm_start.empty() && warm_start.front().eigenvector.size() == n)
    {
        basis.push_back(normalizedOrCanonical(warm_start.front().eigenvector));
    }
    else
    {
        basis.push_back(makeDeterministicUnitVector(static_cast<std::int64_t>(n)));
    }

    std::vector<double> alpha;
    std::vector<double> beta;
    alpha.reserve(static_cast<std::size_t>(max_steps));
    beta.reserve(static_cast<std::size_t>(max_steps));
    const double tol = zeroToleranceFor(config_);

    for (int j = 0; j < max_steps; ++j)
    {
        Eigen::VectorXd w = laplacian * basis[static_cast<std::size_t>(j)];
        if (j > 0)
        {
            w -= beta[static_cast<std::size_t>(j - 1)] * basis[static_cast<std::size_t>(j - 1)];
        }
        const double a = basis[static_cast<std::size_t>(j)].dot(w);
        alpha.push_back(a);
        w -= a * basis[static_cast<std::size_t>(j)];

        for (int i = 0; i <= j; ++i)
        {
            const double correction = basis[static_cast<std::size_t>(i)].dot(w);
            w -= correction * basis[static_cast<std::size_t>(i)];
        }

        const double b = w.norm();
        result.convergence_history.push_back(b);
        if (b <= tol || (j + 1) >= max_steps)
        {
            break;
        }
        beta.push_back(b);
        basis.push_back(w / b);
    }

    const int kdim = static_cast<int>(alpha.size());
    if (kdim == 0)
    {
        result.converged = false;
        return result;
    }

    Eigen::MatrixXd tridiag = Eigen::MatrixXd::Zero(kdim, kdim);
    for (int i = 0; i < kdim; ++i)
    {
        tridiag(i, i) = alpha[static_cast<std::size_t>(i)];
        if (i + 1 < kdim && static_cast<std::size_t>(i) < beta.size())
        {
            tridiag(i, i + 1) = beta[static_cast<std::size_t>(i)];
            tridiag(i + 1, i) = beta[static_cast<std::size_t>(i)];
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(tridiag);
    if (eigensolver.info() != Eigen::Success)
    {
        result.converged = false;
        return result;
    }

    const int count = std::min(target, kdim);
    result.eigenpairs.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        Eigen::VectorXd ritz = Eigen::VectorXd::Zero(n);
        for (int j = 0; j < kdim; ++j)
        {
            ritz.noalias() += eigensolver.eigenvectors()(j, i) * basis[static_cast<std::size_t>(j)];
        }
        Eigenpair pair{};
        pair.eigenvalue = eigensolver.eigenvalues()[i];
        pair.eigenvector = normalizedOrCanonical(ritz);
        pair.error_estimate = computeResidualNorm(laplacian, pair);
        pair.iterations = static_cast<std::uint32_t>(kdim);
        pair.participation_ratio = participationRatio(pair.eigenvector);
        pair.complex_eigenvalue = std::complex<double>(pair.eigenvalue, 0.0);
        pair.complex_eigenvector = pair.eigenvector.cast<std::complex<double>>();
        result.eigenpairs.push_back(std::move(pair));
    }

    const bool early_converged =
        !result.convergence_history.empty() && result.convergence_history.back() <= tol;
    result.converged = early_converged || kdim < max_steps;
    return result;
}

} // namespace nerve::spectral
