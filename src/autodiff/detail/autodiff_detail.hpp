#pragma once
#include "nerve/core_types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace nerve::autodiff
{
class Tensor
{
public:
    Tensor();
    explicit Tensor(double value);
    explicit Tensor(const std::vector<double> &data);
    static Tensor zeros(size_t n);
    static Tensor ones(size_t n);
    double data() const;
    double at(size_t i) const;
    double &at(size_t i);
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor sum() const;
};

class AutogradEngine
{
public:
    AutogradEngine();
    void enableGrad(bool enable);
    void backward(const Tensor &loss);
    void zeroGrad();
    Tensor grad(const Tensor &var) const;
    bool requiresGrad() const;
};

class GraphNode
{
public:
    explicit GraphNode(const Tensor &value);
    void addEdge(const GraphNode &child);
    void clear();
};

class GraphOptimizer
{
public:
    GraphOptimizer(double learning_rate);
    void step();
    void zeroGrad();
};

// Gradient utilities
namespace grad
{
Tensor numericalGradient(const std::function<Tensor(const Tensor &)> &f, const Tensor &x,
                         double eps = 1e-6);
double tensorDistance(const Tensor &a, const Tensor &b);
double tensorNorm(const Tensor &t);
double tensorVariance(const Tensor &t);
double tensorSparsity(const Tensor &t);
} // namespace grad

// SIMD autodiff
namespace simd
{
void backwardAdd(const double *grad, double *out, size_t n);
void backwardMul(const double *grad, const double *a, const double *b, double *out_a, double *out_b,
                 size_t n);
void backwardRelu(const double *grad, const double *input, double *out, size_t n);
} // namespace simd

// Free functions
void enableGrad(bool enable);
void backward(const Tensor &loss);
Tensor grad(const Tensor &var);
} // namespace nerve::autodiff
