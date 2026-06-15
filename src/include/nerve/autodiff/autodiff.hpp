
#pragma once

#include "nerve/core_types.hpp"

#include <concepts>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace nerve::autodiff
{

using Shape = std::vector<Size>;
class Tensor;

template <typename T>
concept ValidTensor = std::copyable<T> && requires(T t) {
    { t } -> std::convertible_to<Tensor>;
};

class Tensor
{
public:
    class ScalarProxy
    {
    public:
        ScalarProxy(Tensor *tensor, Size index);
        ScalarProxy &operator=(double value);
        ScalarProxy &operator+=(double value);
        ScalarProxy &operator-=(double value);
        ScalarProxy &operator*=(double value);
        ScalarProxy &operator/=(double value);
        [[nodiscard]] operator double() const;
        [[nodiscard]] double &operator[](Size inner_index);
        [[nodiscard]] const double &operator[](Size inner_index) const;

    private:
        Tensor *tensor_;
        Size index_;
    };

    class ConstScalarProxy
    {
    public:
        ConstScalarProxy(const Tensor *tensor, Size index);
        [[nodiscard]] operator double() const;
        [[nodiscard]] const double &operator[](Size inner_index) const;

    private:
        const Tensor *tensor_;
        Size index_;
    };

    Tensor();
    Tensor(const std::vector<double> &data, const Shape &shape);
    explicit Tensor(const std::vector<double> &data);
    explicit Tensor(double scalar);

    [[nodiscard]] static Tensor zeros(const Shape &shape);
    static Tensor ones(const Shape &shape);

    ScalarProxy operator[](Size index);
    ConstScalarProxy operator[](Size index) const;

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
    Tensor tanh() const;
    Tensor sum() const;
    Tensor max() const;
    Tensor min() const;

    std::vector<double> &data();
    const std::vector<double> &data() const;
    const Shape &shape() const;
    Size size() const;
    Size ndim() const;

    Tensor grad() const;
    void setGrad(const Tensor &gradient);
    void zeroGrad();
    void backward();
    void setRequiresGrad(bool requiresGrad);
    bool requiresGrad() const;

    Tensor toTensor() const;

private:
    std::vector<double> data_;
    Shape shape_;
    std::vector<double> gradient_;
    bool requires_grad_;

    static Size computeNumel(const Shape &shape);
    Size flatIndex(Size index) const;
    Size flatIndex(Size row, Size col) const;
    double &mutableAt(Size index);
    const double &at(Size index) const;
    double &mutableAt(Size row, Size col);
    const double &at(Size row, Size col) const;

    friend Tensor operator+(double scalar, const Tensor &tensor);
    friend Tensor operator-(double scalar, const Tensor &tensor);
    friend Tensor operator*(double scalar, const Tensor &tensor);
    friend Tensor operator/(double scalar, const Tensor &tensor);
};

Tensor operator+(double scalar, const Tensor &tensor);
Tensor operator-(double scalar, const Tensor &tensor);
Tensor operator*(double scalar, const Tensor &tensor);
Tensor operator/(double scalar, const Tensor &tensor);

class Variable
{
public:
    Variable();
    explicit Variable(const Tensor &data, bool requiresGrad = true);

    Tensor data() const;
    Tensor grad() const;
    bool requiresGrad() const;
    void setRequiresGrad(bool requiresGrad);

    Variable operator+(const Variable &other) const;
    Variable operator-(const Variable &other) const;
    Variable operator*(const Variable &other) const;
    Variable operator/(const Variable &other) const;

    Variable relu() const;
    Variable sigmoid() const;
    Variable tanh() const;
    Variable sum() const;
    Variable max() const;
    Variable min() const;

    void backward();
    void zeroGrad();

private:
    Tensor data_;
};

class ComputationalGraph
{
public:
    void addNode(const Tensor &tensor);
    void addEdge(const Tensor &from, const Tensor &to, const Tensor &grad);
    void clear();
    void backward();
    void zeroGrad();
    std::vector<Tensor> getParameters() const;
    std::vector<std::pair<Tensor, Tensor>> getEdges() const;
    std::vector<Tensor> getEdgeGradients() const;
    void optimize();

private:
    std::vector<Tensor> nodes_;
    std::vector<std::pair<Tensor, Tensor>> edges_;
    std::vector<Tensor> edge_gradients_;
};

class AutoDiffUtils
{
public:
    static bool checkGradient(const std::function<Tensor(const Tensor &)> &func,
                              const Tensor &input, double epsilon = 1e-6);
    static Tensor computeJacobian(const std::function<Tensor(const Tensor &)> &func,
                                  const Tensor &input);
    static Tensor computeHessian(const std::function<Tensor(const Tensor &)> &func,
                                 const Tensor &input);
    static Tensor numericalGradient(const std::function<double(const Tensor &)> &func,
                                    const Tensor &input, double epsilon = 1e-6);
    static void saveGraph(const ComputationalGraph &graph, const std::string &filename);
    static ComputationalGraph loadGraph(const std::string &filename);
    static void profileComputation(const std::function<void()> &computation,
                                   const std::string &name);
};

void enableGrad(const Tensor &tensor);
void backward(const Tensor &tensor);
Tensor grad(const Tensor &tensor);
double tensorNorm(const Tensor &tensor);
double tensorDistance(const Tensor &a, const Tensor &b);
double tensorVariance(const Tensor &tensor);
double tensorSparsity(const Tensor &tensor, double epsilon = 1e-12);

using AutoDiffTensor = Tensor;

} // namespace nerve::autodiff
