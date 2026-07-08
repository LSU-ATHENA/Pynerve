#include "nerve/persistence/reduction/reduction_ops.hpp"

#ifdef NERVE_HAS_CUDA
#include "nerve/gpu/persistence_reducer.hpp"
#endif

namespace nerve::persistence
{

double Reducer::getFiltrationValue(int column) const
{
    if (matrix_ == nullptr || column < 0)
    {
        return 0.0;
    }
    return matrix_->getFiltrationValue(static_cast<Size>(column));
}

int Reducer::getSimplexDimension(int column) const
{
    if (matrix_ == nullptr || column < 0)
    {
        return -1;
    }
    return matrix_->getColSimplexDimension(static_cast<Size>(column));
}

void Reducer::hybridStandardReduction()
{
    reduceMatrix();
}

Reducer::Mode Reducer::determineOptimalMode(Size dataset_size) const
{
    if (dataset_size <= 1000)
        return Mode::kFast;
    if (dataset_size <= 5000)
        return Mode::kHybrid;
    return Mode::kComplete;
}

void Reducer::fastPhase1Reduction()
{
    reduceMatrix();
}

void Reducer::cohomologyReduction()
{
#ifdef NERVE_HAS_CUDA
    if (matrix_ && matrix_->cols() >= 256)
    {
        std::vector<Index> pivots;
        std::vector<std::pair<Size, Size>> pairs;
        nerve::gpu::kernels::GpuPersistenceReducer gpu_reducer;
        auto result = gpu_reducer.computeCohomology(*matrix_, pivots, pairs);
        if (result.isSuccess())
        {
            pivot_columns_.swap(pivots);
            pairs_.clear();
            for (const auto &[birth_idx, death_idx] : pairs)
            {
                Pair p;
                p.birth_index = static_cast<Index>(birth_idx);
                p.death_index = static_cast<Index>(death_idx);
                p.birth = matrix_->getFiltrationValue(birth_idx);
                p.death = matrix_->getFiltrationValue(death_idx);
                p.dimension = matrix_->getColSimplexDimension(birth_idx);
                pairs_.push_back(p);
            }
            computeBettiNumbersFromPivots();
            classifyEssentials();
            return;
        }
    }
#endif
    reduceMatrix();
    computePersistencePairsFromPivots();
    computeBettiNumbersFromPivots();
    classifyEssentials();
}

void Reducer::acceleratedReduction()
{
#ifdef NERVE_HAS_CUDA
    if (matrix_ && matrix_->cols() >= 512)
    {
        std::vector<Index> pivots;
        std::vector<std::pair<Size, Size>> pairs;
        nerve::gpu::kernels::DynamicPartitionReducer gpu_reducer;
        auto result = gpu_reducer.compute(*matrix_, pivots, pairs);
        if (result.isSuccess())
        {
            pivot_columns_.swap(pivots);
            pairs_.clear();
            for (const auto &[birth_idx, death_idx] : pairs)
            {
                Pair p;
                p.birth_index = static_cast<Index>(birth_idx);
                p.death_index = static_cast<Index>(death_idx);
                p.birth = matrix_->getFiltrationValue(birth_idx);
                p.death = matrix_->getFiltrationValue(death_idx);
                p.dimension = matrix_->getColSimplexDimension(birth_idx);
                pairs_.push_back(p);
            }
            computeBettiNumbersFromPivots();
            classifyEssentials();
            return;
        }
    }
#endif
    reduceMatrix();
    computePersistencePairsFromPivots();
    computeBettiNumbersFromPivots();
    classifyEssentials();
}

void Reducer::cohomologyOptimization()
{
    cohomologyReduction();
}

void Reducer::acceleratedPhase2Reduction()
{
    reduceMatrix();
}

void Reducer::matrixReductionDense()
{
    reduceMatrix();
}

void Reducer::twistedReduction()
{
    reduceMatrix();
}

void Reducer::parallelReduction()
{
    reduceMatrix();
}

void Reducer::AcceleratedReduction()
{
    reduceMatrix();
}

void Reducer::computePersistenceImpl()
{
    compute();
}

void Reducer::CohomologyReduction()
{
    reduceMatrix();
}

Index Reducer::getPivot(Size j) const
{
    if (j < pivot_columns_.size())
    {
        return pivot_columns_[j];
    }
    return findLowestPivot(j);
}

Size Reducer::getNumberOfOperations() const
{
    return num_operations_;
}

double Reducer::getComputationTime() const
{
    return computation_time_;
}

bool Reducer::hasPersistencePairs() const
{
    return !pairs_.empty();
}

void Reducer::CohomologyOptimization()
{
    compute();
}

} // namespace nerve::persistence
