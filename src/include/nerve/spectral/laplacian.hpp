
#pragma once
#include "nerve/algebra/cellular.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/core_types.hpp"

#include <complex>
#include <memory>
#include <vector>
namespace nerve::spectral
{
using algebra::Cell;
using algebra::CellularComplex;
using algebra::SimplicialComplex;

// GPU acceleration configuration for Laplacian operations
struct LaplacianConfig
{
    bool enable_gpu = false;          // Enable GPU acceleration
    size_t threshold = 1000;          // Minimum size to use GPU
    bool prefer_tiled_kernels = true; // Use shared memory tiling
    size_t max_gpu_memory_mb = 0;     // 0 = auto-detect
};

class Laplacian
{
public:
    Laplacian();
    explicit Laplacian(const SimplicialComplex &complex);
    explicit Laplacian(const CellularComplex &complex);
    explicit Laplacian(const SimplicialComplex &complex, const LaplacianConfig &config);
    explicit Laplacian(const CellularComplex &complex, const LaplacianConfig &config);
    void buildFromComplex(const SimplicialComplex &complex);
    void buildFromCellular(const CellularComplex &complex);
    ::nerve::Size size() const;
    int maxDimension() const;
    std::vector<std::vector<double>> getLaplacian(int dimension) const;
    std::vector<std::vector<double>> getUpLaplacian(int dimension) const;
    std::vector<std::vector<double>> getDownLaplacian(int dimension) const;
    std::vector<std::vector<double>> getHodgeLaplacian(int dimension) const;
    std::vector<double> eigenvalues(int dimension, ::nerve::Size k = 0) const;
    std::vector<std::vector<double>> eigenvectors(int dimension, ::nerve::Size k = 0) const;
    std::vector<double> spectrum(int dimension) const;
    std::vector<std::vector<double>> computeEmbedding(int dimension,
                                                      ::nerve::Size target_dim = 2) const;
    std::vector<std::vector<double>> computeDiffusionMap(::nerve::Size target_dim = 2) const;
    std::vector<std::vector<double>> heatKernel(int dimension, double t) const;
    std::vector<std::vector<double>> heatFlow(const std::vector<double> &initial, int dimension,
                                              double t) const;
    std::vector<double> computeSpectralGap(int dimension) const;
    std::vector<double> computeCheegerConstants() const;
    std::vector<int> computeMorseIndex(int dimension) const;

private:
    std::vector<std::vector<double>> boundary_matrix_;
    std::vector<std::vector<double>> coboundary_matrix_;
    std::vector<std::vector<std::vector<double>>> laplacians_;
    std::vector<std::vector<std::vector<double>>> up_laplacians_;
    std::vector<std::vector<std::vector<double>>> down_laplacians_;
    std::vector<std::vector<std::vector<double>>> hodge_laplacians_;
    std::vector<int> simplex_dimensions_;
    ::nerve::Size size_;
    int max_dimension_;
    LaplacianConfig config_;
    void buildBoundaryMatrices(const SimplicialComplex &complex);
    void buildBoundaryMatrices(const CellularComplex &complex);
    void computeAllLaplacians();
    void computeAllLaplaciansGpu(); // GPU-accelerated version
    std::vector<std::vector<double>>
    computeMatrixProduct(const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<double>> &B) const;
    std::vector<std::vector<double>>
    computeMatrixTranspose(const std::vector<std::vector<double>> &A) const;
    std::vector<std::vector<double>>
    computeEigendecomposition(const std::vector<std::vector<double>> &matrix,
                              std::vector<double> &eigenvalues) const;
};
class DiracOperator
{
public:
    DiracOperator();
    explicit DiracOperator(const SimplicialComplex &complex);
    explicit DiracOperator(const CellularComplex &complex);
    void buildFromComplex(const SimplicialComplex &complex);
    void buildFromCellular(const CellularComplex &complex);
    std::vector<std::vector<double>> getDirac() const;
    std::vector<std::vector<double>> getDiracSquared() const;
    std::vector<double> eigenvalues(::nerve::Size k = 0) const;
    std::vector<std::vector<double>> eigenvectors(::nerve::Size k = 0) const;
    std::vector<std::vector<std::complex<double>>> getSpinorLaplacian() const;
    std::vector<std::vector<std::complex<double>>> getChiralityOperator() const;
    int computeAtiyahSingerIndex() const;
    std::vector<int> computeAnalyticalIndex() const;
    std::vector<int> computeTopologicalIndex() const;

private:
    std::vector<std::vector<double>> boundary_matrix_;
    std::vector<std::vector<double>> coboundary_matrix_;
    std::vector<std::vector<double>> dirac_matrix_;
    std::vector<std::vector<double>> dirac_squared_matrix_;
    ::nerve::Size size_;
    int max_dimension_;
    void buildDiracMatrix();
    std::vector<std::vector<double>>
    computeMatrixProduct(const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<double>> &B) const;
    std::vector<std::vector<double>>
    computeEigendecomposition(const std::vector<std::vector<double>> &matrix,
                              std::vector<double> &eigenvalues) const;
    std::vector<std::vector<std::complex<double>>> buildChiralityOperator() const;
};
class HodgeTheory
{
public:
    HodgeTheory();
    explicit HodgeTheory(const SimplicialComplex &complex);
    explicit HodgeTheory(const CellularComplex &complex);
    std::vector<std::vector<double>> computeHarmonicForms(int dimension) const;
    std::vector<std::vector<double>> computeExactForms(int dimension) const;
    std::vector<std::vector<double>> computeCohexactForms(int dimension) const;
    std::vector<std::vector<double>> getHodgeStar(int dimension) const;
    std::vector<std::vector<double>> getCodifferential(int dimension) const;
    std::vector<int> computeHodgeNumbers() const;
    std::vector<int> computeBettiNumbers() const;
    std::vector<std::vector<double>>
    wedgeProduct(const std::vector<std::vector<double>> &alpha,
                 const std::vector<std::vector<double>> &beta) const;
    std::vector<std::vector<double>>
    interiorProduct(const std::vector<std::vector<double>> &vector_field,
                    const std::vector<std::vector<double>> &form) const;

private:
    Laplacian laplacian_;
    std::vector<std::vector<std::vector<double>>> hodge_stars_;
    std::vector<std::vector<std::vector<double>>> codifferentials_;
    void buildHodgeOperators();
    std::vector<std::vector<double>> computeHodgeStarMatrix(int dimension) const;
};
class SpectralClustering
{
public:
    SpectralClustering();
    explicit SpectralClustering(const Laplacian &laplacian);
    std::vector<int> cluster(::nerve::Size k, int dimension = 0) const;
    std::vector<int> clusterNormalizedLaplacian(::nerve::Size k, int dimension = 0) const;
    std::vector<int> clusterFromEmbedding(::nerve::Size k,
                                          const std::vector<std::vector<double>> &embedding) const;
    std::vector<double> computeClusterQuality(const std::vector<int> &labels) const;
    std::vector<int> computeOptimalK(int max_k = 10) const;

private:
    const Laplacian *laplacian_;
    std::vector<int> kmeansClustering(const std::vector<std::vector<double>> &points,
                                      ::nerve::Size k) const;
    double computeSilhouetteScore(const std::vector<std::vector<double>> &points,
                                  const std::vector<int> &labels) const;
};
class ManifoldLearning
{
public:
    ManifoldLearning();
    explicit ManifoldLearning(const Laplacian &laplacian);
    std::vector<std::vector<double>> laplacianEigenmaps(::nerve::Size target_dim = 2,
                                                        int dimension = 0) const;
    std::vector<std::vector<double>> diffusionMaps(::nerve::Size target_dim = 2,
                                                   double t = 1.0) const;
    std::vector<std::vector<double>> isomap(::nerve::Size target_dim = 2) const;
    std::vector<std::vector<double>> lle(::nerve::Size target_dim = 2,
                                         ::nerve::Size k_neighbors = 12) const;
    std::vector<std::vector<double>> hessianEigenmaps(::nerve::Size target_dim = 2,
                                                      ::nerve::Size k_neighbors = 12) const;
    std::vector<double> computeIntrinsicDimensionality() const;
    std::vector<std::vector<double>>
    computeLocalTangentSpace(const std::vector<double> &point) const;

private:
    const Laplacian *laplacian_;
    std::vector<std::vector<double>>
    computeNearestNeighbors(const std::vector<std::vector<double>> &points, ::nerve::Size k) const;
    std::vector<std::vector<double>>
    computeReconstructionWeights(const std::vector<std::vector<double>> &points,
                                 ::nerve::Size k) const;
};
} // namespace nerve::spectral
