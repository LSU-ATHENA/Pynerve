#pragma once
#include "nerve/core_types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace nerve::autodiff
{
class Tensor;
class ComputationalGraph;
class AutogradEngine;
class GraphNode;
class GraphOptimizer;
namespace grad
{
Tensor numericalGradient(const std::function<Tensor(const Tensor &)> &f, const Tensor &x,
                         double eps = 1e-6);
double tensorDistance(const Tensor &a, const Tensor &b);
double tensorNorm(const Tensor &t);
double tensorVariance(const Tensor &t);
double tensorSparsity(const Tensor &t);
} // namespace grad
class Tensor
{
public:
    Tensor();
    explicit Tensor(double value);
    explicit Tensor(const std::vector<double> &data);
    Tensor(const std::vector<double> &data, const std::vector<Size> &shape);
    static Tensor zeros(size_t n);
    static Tensor zeros(const std::vector<Size> &shape);
    static Tensor ones(size_t n);
    const std::vector<double> &data() const;
    double at(size_t i) const;
    double &at(size_t i);
    Size size() const;
    Size ndim() const;
    const std::vector<Size> &shape() const;
    double operator[](Size i) const;
    double &operator[](Size i);
    auto begin() const { return values_.begin(); }
    auto end() const { return values_.end(); }
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor sum() const;
    Tensor grad() const;
    void setGrad(const Tensor &gradient);
    void setRequiresGrad(bool requires);
    bool requiresGrad() const;
    void backward();
    void zeroGrad();

private:
    std::vector<double> values_;
    std::vector<Size> shape_;
    std::vector<double> gradient_;
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
Tensor computeGrad(const Tensor &var);
} // namespace nerve::autodiff
