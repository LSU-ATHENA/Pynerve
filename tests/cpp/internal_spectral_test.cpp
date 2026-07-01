#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/spectral/detail/spectral_detail.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

#if NERVE_HAS_EIGEN

using nerve::spectral::SpectralAnomalyDetector;
using nerve::spectral::SpectralFeatureExtractor;
using nerve::spectral::SpectralStackManager;

bool check_anomaly_detector_initialization()
{
    SpectralAnomalyDetector::AnomalyConfig config;
    config.anomaly_threshold = 2.5;
    config.reference_window_size = 50;
    config.use_eigenvalue_anomalies = true;
    config.use_eigenvector_anomalies = false;
    config.enable_adaptive_threshold = false;
    SpectralAnomalyDetector detector(config);

    SpectralAnomalyDetector::AnomalyResult result = detector.detectAnomaly({});
    if (result.anomaly_score != 0.0)
    {
        std::cerr << "anomaly detector returned non-zero score for empty spectrum\n";
        return false;
    }
    if (result.is_anomaly)
    {
        std::cerr << "anomaly detector flagged empty spectrum as anomalous\n";
        return false;
    }
    return true;
}

bool check_feature_extractor_config()
{
    SpectralFeatureExtractor::FeatureConfig feat_config;
    feat_config.use_eigenvalues = true;
    feat_config.use_eigenvectors = false;
    feat_config.use_spectral_gaps = false;
    feat_config.use_participation_ratios = false;
    feat_config.normalize_features = false;
    feat_config.max_features = 50;
    SpectralFeatureExtractor extractor(feat_config);
    return true;
}

#endif // NERVE_HAS_EIGEN

bool check_spectral_simd_basic_ops()
{
#if defined(NERVE_HAS_X86_INTRINSICS)
    std::vector<double> mat = {2.0, 0.0, 0.0, 2.0};
    std::vector<double> vec = {3.0, 4.0};
    std::vector<double> result(2, 0.0);
    nerve::spectral::matVecMulSimd(mat.data(), vec.data(), 2, result.data());
    if (std::abs(result[0] - 6.0) > 1e-12 || std::abs(result[1] - 8.0) > 1e-12)
    {
        std::cerr << "spectral SIMD mat-vec mul mismatch\n";
        return false;
    }
    double dp = nerve::spectral::dotProductSimd(vec.data(), vec.data(), 2);
    if (std::abs(dp - 25.0) > 1e-12)
    {
        std::cerr << "spectral SIMD dot product mismatch\n";
        return false;
    }
#else
    (void)0;
#endif
    return true;
}

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
using nerve::algebra::SimplicialComplex;

bool check_dirac_operator_construction()
{
    SimplicialComplex complex;
    complex.addSimplexWithFiltration({0}, 0.0);
    complex.addSimplexWithFiltration({1}, 0.0);
    complex.addSimplexWithFiltration({0, 1}, 1.0);
    nerve::spectral::DiracOperator dirac(complex);
    auto matrix = dirac.getDirac();
    if (matrix.empty())
    {
        std::cerr << "Dirac operator matrix is empty\n";
        return false;
    }
    if (matrix.size() != matrix[0].size())
    {
        std::cerr << "Dirac operator matrix is not square\n";
        return false;
    }
    if (matrix.size() != 6)
    {
        std::cerr << "Dirac operator matrix has unexpected size\n";
        return false;
    }
    return true;
}

bool check_persistent_laplacian_support()
{
    std::vector<std::vector<uint32_t>> boundary = {{}, {0}, {1}};
    std::vector<double> filtration = {0.0, 0.5, 1.0};
    nerve::spectral::SpectralConfig spectral_config;
    nerve::spectral::PersistentLaplacianSolver solver(spectral_config);
    auto laplacian =
        nerve::spectral::PersistentLaplacianSolver::buildPersistentLaplacian(boundary, filtration);
    if (laplacian.rows() == 0)
    {
        std::cerr << "persistent laplacian matrix has zero rows\n";
        return false;
    }
    if (laplacian.rows() != laplacian.cols())
    {
        std::cerr << "persistent laplacian matrix is not square\n";
        return false;
    }
    return true;
}
#endif

} // namespace

int main()
{
#if NERVE_HAS_EIGEN
    if (!check_anomaly_detector_initialization())
    {
        std::cerr << "FAIL: anomaly detector initialization\n";
        return 1;
    }
    if (!check_feature_extractor_config())
    {
        std::cerr << "FAIL: feature extractor config\n";
        return 1;
    }
#endif
    if (!check_spectral_simd_basic_ops())
    {
        std::cerr << "FAIL: spectral simd basic ops\n";
        return 1;
    }
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
    if (!check_dirac_operator_construction())
    {
        std::cerr << "FAIL: dirac operator construction\n";
        return 1;
    }
    if (!check_persistent_laplacian_support())
    {
        std::cerr << "FAIL: persistent laplacian support\n";
        return 1;
    }
#endif
    return 0;
}
