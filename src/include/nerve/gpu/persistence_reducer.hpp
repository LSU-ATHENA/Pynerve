#pragma once
#ifdef NERVE_HAS_CUDA
#include "nerve/algebra/boundary.hpp"
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"

#include <utility>
#include <vector>

namespace nerve::gpu::kernels
{

class GpuPersistenceReducer
{
public:
    GpuPersistenceReducer();
    ~GpuPersistenceReducer();

    errors::ErrorResult<void> compute(const algebra::BoundaryMatrix &boundary_matrix,
                                      std::vector<Index> &out_pivots,
                                      std::vector<std::pair<Size, Size>> &out_pairs);

    errors::ErrorResult<void> computeCohomology(const algebra::BoundaryMatrix &boundary_matrix,
                                                std::vector<Index> &out_pivots,
                                                std::vector<std::pair<Size, Size>> &out_pairs);
};

class DynamicPartitionReducer
{
public:
    DynamicPartitionReducer();
    ~DynamicPartitionReducer();

    DynamicPartitionReducer(const DynamicPartitionReducer &) = delete;
    DynamicPartitionReducer &operator=(const DynamicPartitionReducer &) = delete;

    errors::ErrorResult<void> compute(const algebra::BoundaryMatrix &boundary_matrix,
                                      std::vector<Index> &out_pivots,
                                      std::vector<std::pair<Size, Size>> &out_pairs);
};

} // namespace nerve::gpu::kernels
#endif
