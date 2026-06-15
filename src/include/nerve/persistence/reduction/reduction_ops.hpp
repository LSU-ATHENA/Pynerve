
#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/column_types.hpp"

#include <memory>
#include <vector>

namespace nerve::gpu
{
bool isAvailable();
}

namespace nerve::persistence
{

class Diagram;

// Type alias for Z2 column representation
using Z2Column = std::vector<Size>;

/**
 * @brief Matrix reduction algorithm for persistent homology computation
 */
class Reducer
{
public:
    /**
     * @brief Algorithm modes for different dataset sizes
     */
    enum class Mode
    {
        kFast,
        kComplete, // Full computation
        kHybrid    // Adaptive
    };

    /**
     * @brief Cohomology class for fast-style computation
     */
    struct CohomologyClass
    {
        int birth;
        int death;
        int dimension;
        double birthTime;
        double deathTime;

        CohomologyClass(int b, int d, int dim, double bt, double dt)
            : birth(b)
            , death(d)
            , dimension(dim)
            , birthTime(bt)
            , deathTime(dt)
        {}
    };

    explicit Reducer(const algebra::BoundaryMatrix &boundary_matrix);

    void compute();
    void computeWithCoefficients();

    const std::vector<Pair> &getPairs() const;
    const std::vector<Index> &getEssentials() const;
    const std::vector<Size> &getBetti() const;

    // Status getters
    Size getNumberOfOperations() const;
    double getComputationTime() const;
    bool hasPersistencePairs() const;

    // Algorithm mode selection
    Mode determineOptimalMode(Size dataset_size) const;

    // Apparent pair info (Chen & Kerber 2011)
    struct ApparentPairInfo
    {
        Size birth_index;
        Size death_index;
    };

    // Twist reduction and optimization methods
    std::vector<ApparentPairInfo> findApparentPairs(std::vector<bool> &cleared);
    void reduceTwist(std::vector<Index> &lowRowToCol, std::vector<bool> &cleared);

    // fast-style optimizations
    void cohomologyReduction();
    void acceleratedReduction();
    void cohomologyOptimization();

    // GPU configuration
    void enableGPU(bool use) { use_gpu_ = use; }
    bool gpuEnabled() const { return use_gpu_ && nerve::gpu::isAvailable(); }
    void setThreshold(Size threshold) { gpu_threshold_ = threshold; }
    Size getThreshold() const { return gpu_threshold_; }

    // Get boundary matrix reference (for GPU acceleration)
    const algebra::BoundaryMatrix *getMatrix() const { return matrix_; }

    // Internal helper methods
    void initializeReduction();
    void reduceMatrix();
    void reduceMatrixColumnByColumn();
    void computePersistencePairsFromPivots();
    void computeBettiNumbersFromPivots();
    void classifyEssentials();
    void clearLowestPivot(Size j);
    Index getPivot(Size col) const;
    Index findLowestPivot(Size j) const;
    Z2Column toZ2Rows(const Vector<std::pair<Size, double>> &column);
    void setPivot(Size j, Index pivot);
    void eliminatePivot(Size j, Size k);
    void hybridStandardReduction();
    void fastPhase1Reduction();
    void acceleratedPhase2Reduction();
    void matrixReductionDense();
    void twistedReduction();
    void parallelReduction();
    void AcceleratedReduction();
    void computePersistenceImpl();
    void CohomologyReduction();
    double getFiltrationValue(int column) const;
    int getSimplexDimension(int column) const;
    void CohomologyOptimization();

protected:
    // Allow derived classes to access/modify pivot columns
    std::vector<Index> &getPivotColumns() { return pivot_columns_; }
    const std::vector<Index> &getPivotColumns() const { return pivot_columns_; }

private:
    const algebra::BoundaryMatrix *matrix_;

    Size num_operations_;
    double computation_time_;
    std::vector<Pair> pairs_;
    std::vector<Index> essentials_;
    std::vector<Size> betti_;

    // fast-style data structures
    std::vector<CohomologyClass> cohomology_classes_;
    std::vector<int> cocycle_pivots_;
    std::vector<bool> column_processed_;
    std::vector<std::vector<int>> coboundary_matrix_;

    // Accelerated data structures
    std::vector<std::vector<std::pair<Size, double>>> reduced_matrix_;
    std::vector<Index> compact_pivots_;
    std::vector<Index> pivot_columns_;
    std::vector<int> simplex_dimensions_;
    std::vector<std::vector<std::pair<uint16_t, float>>> compact_matrix_;
    bool columns_loaded_ = false;

    // GPU acceleration configuration
    bool use_gpu_ = true;
    Size gpu_threshold_ = 10000;

    // fast-style methods
    void computeCoboundaryMatrix();
    void computeCoboundary();
    void cohomologyReductionStep(int column);
    void cohomologyStep(int column);
    int findCocyclePivot(int column) const;
    void addCocycle(int target, int source);
    void addCocycle(int target, int source, const std::vector<int> &vertices2, int vertex);
    void compressCocycles();

    // Accelerated methods
    Index findLowestPivotCacheAware(Size column) const;
    bool hasPivotInColumnAtomic(Size column, Index pivot) const;
    void addColumnSIMD(Size dest_col, Size src_col, double coefficient);

    // Clearing helper (Chen & Kerber 2011)
    void clearBirthColumn(Size pivotRow, std::vector<bool> &cleared);

    // Helper methods (see public section for implementations)
};

} // namespace nerve::persistence
