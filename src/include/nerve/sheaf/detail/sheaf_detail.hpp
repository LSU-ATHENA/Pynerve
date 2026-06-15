#pragma once
#include "nerve/core_types.hpp"
#include "nerve/sheaf/sheaf.hpp"

#include <memory>
#include <vector>

namespace nerve::sheaf
{
class SheafEngine
{
public:
    SheafEngine();
    bool initialize(int num_cells);
    bool isInitialized() const;
};

class MorphismOptimizer
{
public:
    MorphismOptimizer();
    void setMaxIterations(int iterations);
    int maxIterations() const;
};

class ParallelSheafBuilder
{
public:
    ParallelSheafBuilder(size_t num_threads);
    void build();
    bool isBuilt() const;
};

class SheafLaplacianFactory
{
public:
    struct Config
    {
        int max_dimension = 2;
        bool use_parallel = false;
        double threshold = 1e-10;
        bool validate() const;
    };
    static bool isValidConfig(const Config &config);
};

class SheafLaplacian
{
public:
    SheafLaplacian(int dimension);
    int dimension() const;
    std::vector<std::vector<double>> getMatrix() const;
};
} // namespace nerve::sheaf
