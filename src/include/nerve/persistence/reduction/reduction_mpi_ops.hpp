#pragma once

#include "nerve/config.hpp"

#if NERVE_HAS_MPI && __has_include(<mpi.h>)

#include "nerve/algebra/boundary.hpp"
#include "nerve/core_types.hpp"
#include "nerve/distributed/mpi_persistence.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

class MpiDistributedReducer
{
public:
    void compute(const algebra::BoundaryMatrix &matrix);

    const std::vector<Pair> &getPairs() const { return pairs_; }
    const std::vector<Index> &getEssentials() const { return essentials_; }
    const std::vector<Size> &getBetti() const { return betti_; }

private:
    void reduceLocalColumns();
    void exchangePivots();
    void resolveRemoteDependencies();
    void gatherResults();
    void computePersistencePairsFromPivots();
    void computeBettiNumbersFromPivots();

    const algebra::BoundaryMatrix *matrix_ = nullptr;
    distributed::MPICommunicator comm_;

    std::vector<Size> local_columns_;
    std::vector<Index> local_pivots_;

    // After resolve: for each column, the resolved lowest pivot row (-1 if none)
    std::vector<Index> global_pivots_;

    std::vector<Pair> pairs_;
    std::vector<Index> essentials_;
    std::vector<Size> betti_;
};

} // namespace nerve::persistence

#endif // NERVE_HAS_MPI
