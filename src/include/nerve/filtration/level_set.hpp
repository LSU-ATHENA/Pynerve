
#pragma once
#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"
#ifdef USE_TORCH
#include <torch/torch.h>
#endif
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>
namespace nerve::filtration
{
class LevelSet
{
public:
    LevelSet() = default;
    explicit LevelSet(const std::vector<Size> &grid_shape);
    void setGridShape(const std::vector<Size> &shape);
    void setFiltrationType(const std::string &type);
    void setNumLevels(Size num_levels);
    void setAdaptiveLevels(bool adaptive);
    errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
    buildFiltration(const core::BufferView<const double> &scalar_field,
                    const core::DeterminismContract &contract = {});
#ifdef USE_TORCH
    errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
    buildFiltration(const at::Tensor &scalar_field, const core::DeterminismContract &contract = {});
#endif
    errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
    buildFiltration(const core::BufferView<const double> &scalar_field,
                    const std::vector<std::vector<Index>> &connectivity,
                    const core::DeterminismContract &contract = {});
    std::vector<std::pair<algebra::Simplex, double>>
    build2dFiltration(const core::BufferView<const double> &scalar_field, Size width, Size height,
                      const core::DeterminismContract &contract = {});
    std::vector<std::pair<algebra::Simplex, double>>
    build3dFiltration(const core::BufferView<const double> &scalar_field, Size width, Size height,
                      Size depth, const core::DeterminismContract &contract = {});
    std::vector<double> computeLevels(const core::BufferView<const double> &scalar_field) const;
    std::vector<double>
    computeAdaptiveLevels(const core::BufferView<const double> &scalar_field) const;
    std::vector<double> computeQuantileLevels(const core::BufferView<const double> &scalar_field,
                                              Size num_quantiles) const;
    Size getNumSimplices() const;
    Size getNumSimplicesOfDimension(Size dim) const;
    std::vector<double> getFiltrationValues() const;
    std::map<Size, std::vector<algebra::Simplex>> getSimplicesByLevel() const;
    std::vector<Index> findCriticalPoints(const std::vector<double> &scalar_field) const;
    std::vector<Index> findMinima(const std::vector<double> &scalar_field) const;
    std::vector<Index> findMaxima(const std::vector<double> &scalar_field) const;
    std::vector<Index> findSaddles(const std::vector<double> &scalar_field) const;
    double getComputationTime() const;
    std::vector<Index> getNeighbors(Index point_index) const;

private:
    std::vector<Size> grid_shape_;
    std::string filtration_type_ = "sublevel";
    Size num_levels_ = 100;
    bool adaptive_levels_ = true;
    bool use_mesh_connectivity_ = false;
    std::vector<std::vector<Index>> mesh_neighbors_;
    std::vector<std::pair<algebra::Simplex, double>> filtration_;
    std::vector<double> levels_;
    double computation_time_ = 0.0;
    void validateScalarField(const std::vector<double> &scalar_field) const;
#ifdef USE_TORCH
    void validateTensor(const core::Tensor &scalar_field) const;
#endif
    void buildGridConnectivity();
    void buildMeshConnectivity(const std::vector<std::vector<Index>> &connectivity);
    std::vector<Index> getGridNeighbors(Index point_index) const;
    std::vector<std::vector<Index>> getGridSimplices() const;
    Index gridIndexToLinear(const std::vector<Index> &grid_coords) const;
    std::vector<Index> linearIndexToGrid(Index linear_index) const;

public:
    void buildVertexSimplices(const std::vector<double> &scalar_field);
    void buildEdgeSimplices(const std::vector<double> &scalar_field);
    void buildTriangleSimplices(const std::vector<double> &scalar_field);
    void buildTetrahedronSimplices(const std::vector<double> &scalar_field);
    double assignFiltrationValue(const algebra::Simplex &simplex,
                                 const std::vector<double> &scalar_field) const;
    std::vector<Index> getSimplexVertices(const algebra::Simplex &simplex) const;
    void sortFiltration();
    Size getGridDimension() const;
    Size getTotalGridPoints() const;
};
class MorseSmale
{
public:
    MorseSmale() = default;
    explicit MorseSmale(const std::vector<Size> &grid_shape);
    void setGradientMethod(const std::string &method);
    void setSimplificationTolerance(double tolerance);
    void setPersistenceThreshold(double threshold);
    std::vector<std::pair<algebra::Simplex, double>>
    buildMorseSmaleComplex(const core::BufferView<const double> &scalar_field,
                           const core::DeterminismContract &contract = {});
    std::vector<std::vector<Index>>
    computeGradientFlow(const core::BufferView<const double> &scalar_field,
                        const core::DeterminismContract &contract = {}) const;
    std::map<Index, std::string>
    classifyCriticalPoints(const core::BufferView<const double> &scalar_field,
                           const core::DeterminismContract &contract = {}) const;
    std::vector<std::vector<Index>>
    computeSeparatrices(const core::BufferView<const double> &scalar_field) const;
    std::vector<std::vector<Index>>
    computeAttractorBasins(const core::BufferView<const double> &scalar_field) const;
    std::vector<std::vector<Index>>
    computeRepellerBasins(const core::BufferView<const double> &scalar_field) const;

private:
    std::vector<Size> grid_shape_;
    std::string gradient_method_ = "central";
    double simplification_tolerance_ = 1e-6;
    double persistence_threshold_ = 0.0;
    std::vector<std::vector<double>> computeGradient(const std::vector<double> &scalar_field) const;
    std::vector<Index> followGradientDescent(const std::vector<double> &scalar_field,
                                             const std::vector<std::vector<double>> &gradient,
                                             Index start_point) const;
    bool isCriticalPoint(const std::vector<double> &scalar_field, Index point_index) const;
    std::string classifyCriticalPointType(const std::vector<double> &scalar_field,
                                          Index point_index) const;
};
class Watershed
{
public:
    Watershed() = default;
    void setConnectivity(Size connectivity);
    void setMinRegionSize(Size min_size);
    void setMergeThreshold(double threshold);
    std::vector<std::pair<algebra::Simplex, double>>
    buildWatershedFiltration(const core::BufferView<const double> &scalar_field,
                             const std::vector<Size> &shape,
                             const core::DeterminismContract &contract = {});
    std::vector<std::vector<Index>>
    computeWatershedSegments(const core::BufferView<const double> &scalar_field,
                             const std::vector<Size> &shape,
                             const core::DeterminismContract &contract = {}) const;
    std::vector<std::vector<Index>>
    mergeSmallBasins(const std::vector<std::vector<Index>> &basins,
                     const core::BufferView<const double> &scalar_field,
                     const core::DeterminismContract &contract = {}) const;

private:
    Size connectivity_ = 8;
    Size min_region_size_ = 10;
    double merge_threshold_ = 0.1;
    std::vector<Index> findLocalMinima(const std::vector<double> &scalar_field,
                                       const std::vector<Size> &shape) const;
    std::vector<std::vector<Index>> growBasins(const std::vector<double> &scalar_field,
                                               const std::vector<Size> &shape,
                                               const std::vector<Index> &minima) const;
    std::vector<Index> getNeighbors(Index point_index, const std::vector<Size> &shape) const;
};
class DiscreteMorse
{
public:
    DiscreteMorse() = default;
    void setAcyclicMatching(bool acyclic);
    void setOptimizationLevel(Size level);
    std::vector<std::pair<algebra::Simplex, double>>
    buildDiscreteMorseFiltration(const std::vector<double> &scalar_field,
                                 const std::vector<std::vector<Index>> &connectivity);
    std::map<algebra::Simplex, algebra::Simplex>
    computeMorseMatching(const std::vector<std::pair<algebra::Simplex, double>> &filtration) const;
    std::vector<algebra::Simplex>
    getCriticalSimplices(const std::map<algebra::Simplex, algebra::Simplex> &matching) const;
    std::vector<std::pair<algebra::Simplex, algebra::Simplex>>
    getMorseBoundaries(const std::map<algebra::Simplex, algebra::Simplex> &matching) const;

private:
    bool acyclic_matching_ = true;
    Size optimization_level_ = 1;
    bool isValidMatching(const std::map<algebra::Simplex, algebra::Simplex> &matching) const;
    bool isAcyclic(const std::map<algebra::Simplex, algebra::Simplex> &matching) const;
    std::vector<algebra::Simplex>
    findCofaces(const algebra::Simplex &simplex,
                const std::vector<std::pair<algebra::Simplex, double>> &filtration) const;
};
std::vector<std::pair<algebra::Simplex, double>>
computeSublevelFiltration(const std::vector<double> &scalar_field, const std::vector<Size> &shape,
                          Size num_levels = 100);
std::vector<std::pair<algebra::Simplex, double>>
computeSuperlevelFiltration(const std::vector<double> &scalar_field, const std::vector<Size> &shape,
                            Size num_levels = 100);
errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
computeAdaptiveFiltration(const std::vector<double> &scalar_field, const std::vector<Size> &shape);
std::vector<Index> findCriticalPoints2d(const std::vector<double> &scalar_field, Size width,
                                        Size height);
std::vector<Index> findCriticalPoints3d(const std::vector<double> &scalar_field, Size width,
                                        Size height, Size depth);
std::vector<std::vector<double>> computeGradientField(const std::vector<double> &scalar_field,
                                                      const std::vector<Size> &shape);
std::vector<std::vector<Index>> computeWatershed2d(const std::vector<double> &scalar_field,
                                                   Size width, Size height);
} // namespace nerve::filtration
