#pragma once
#ifdef NERVE_HAS_TORCH
#include "nerve/torch/ml_operations.hpp"
#include "nerve/torch/persistence_diagram.hpp"

#include <vector>

namespace nerve::torch
{
class AutogradTensor
{
public:
    AutogradTensor();
    explicit AutogradTensor(double value);
    AutogradTensor operator+(const AutogradTensor &other) const;
    AutogradTensor operator*(const AutogradTensor &other) const;
    void backward();
};

class DiagramOps
{
public:
    static std::vector<double> births(const std::vector<std::vector<double>> &diagram);
    static std::vector<double> deaths(const std::vector<std::vector<double>> &diagram);
    static std::vector<double> persistenceLengths(const std::vector<std::vector<double>> &diagram);
};

class PersistenceImageKernel
{
public:
    PersistenceImageKernel(size_t resolution_birth = 20, size_t resolution_death = 20,
                           double sigma = 0.5);
    std::vector<std::vector<double>> compute(const std::vector<std::vector<double>> &diagram);
};

class PersistenceStatistics
{
public:
    static double meanPersistence(const std::vector<std::vector<double>> &diagram);
    static double maxPersistence(const std::vector<std::vector<double>> &diagram);
    static std::vector<double>
    persistenceByDimension(const std::vector<std::vector<double>> &diagram, int dim);
};

class DiagramVectorizer
{
public:
    explicit DiagramVectorizer(size_t output_dim);
    std::vector<double> vectorize(const std::vector<std::vector<double>> &diagram);
};
} // namespace nerve::torch
#endif
