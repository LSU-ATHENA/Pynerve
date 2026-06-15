#pragma once
#include <cuda_runtime.h>

#include <vector>

namespace nerve::dmt::gpu
{

struct DiscreteGradientQuadrant
{
    int *d_gradient = nullptr;
    int *d_critical = nullptr;
    size_t grad_size = 0;
};

DiscreteGradientQuadrant computeGradientGPU(const int *d_cells, size_t n_cells, const int *d_pairs,
                                            size_t n_pairs, cudaStream_t stream = 0);

void freeGradientGPU(DiscreteGradientQuadrant &grad);

std::vector<int> extractCriticalCellsCPU(const std::vector<int> &gradient, size_t n_cells);

} // namespace nerve::dmt::gpu
