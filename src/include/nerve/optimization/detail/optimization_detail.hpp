#pragma once
#include "nerve/core_types.hpp"

#include <vector>

namespace nerve::optimization
{
class CompactSummary
{
public:
    CompactSummary(size_t max_pairs);
    void addPair(int dim, double birth, double death);
    double compressionRatio() const;
    size_t numPairs() const;
};

struct GPUPrimalConfig
{
    bool enable_gpu = false;
    size_t block_size = 256;
    bool use_tensor_cores = false;
    bool validate() const;
};

class StreamingPHOptimizer
{
public:
    struct WindowConfig
    {
        size_t window_size = 100;
        size_t stride = 10;
        size_t max_dim = 2;
        bool validate() const;
    };
    explicit StreamingPHOptimizer(const WindowConfig &config);
    WindowConfig getConfig() const;
};

namespace simd
{
void optimizerStep(const double *gradient, double *params, double learning_rate, size_t n);
}
} // namespace nerve::optimization
