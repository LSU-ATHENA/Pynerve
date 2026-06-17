#pragma once

#include "nerve/core_types.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"
#include "nerve/sheaf/sheaf_laplacian.hpp"

#include <memory>
#include <vector>

namespace nerve::sheaf
{

class MorphismOptimizer
{
public:
    MorphismOptimizer();
    void setMaxIterations(int iterations);
    int maxIterations() const;
};

} // namespace nerve::sheaf
