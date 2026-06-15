
#pragma once

#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace nerve::persistence::detail
{

inline double tensorSum(const autodiff::Tensor &tensor)
{
    const auto &data = tensor.data();
    return std::accumulate(data.begin(), data.end(), 0.0);
}

inline double tensorMeanAbs(const autodiff::Tensor &tensor)
{
    const auto &data = tensor.data();
    if (data.empty())
    {
        return 0.0;
    }
    double total = 0.0;
    for (double value : data)
    {
        total += std::abs(value);
    }
    return total / static_cast<double>(data.size());
}

inline Size tensorRows(const autodiff::Tensor &tensor)
{
    const auto &shape = tensor.shape();
    if (!shape.empty())
    {
        return shape[0];
    }
    return tensor.size();
}

inline Size tensorCols(const autodiff::Tensor &tensor)
{
    const auto &shape = tensor.shape();
    if (shape.size() >= 2)
    {
        return shape[1];
    }
    return shape.empty() ? 0 : 1;
}

inline autodiff::Tensor
finiteDifferenceGradient(const std::function<autodiff::Tensor(const autodiff::Tensor &)> &func,
                         const autodiff::Tensor &input, double epsilon)
{
    autodiff::Tensor gradients = autodiff::Tensor::zeros(input.shape());
    for (Size index = 0; index < input.data().size(); ++index)
    {
        autodiff::Tensor plus = input;
        autodiff::Tensor minus = input;
        plus.data()[index] += epsilon;
        minus.data()[index] -= epsilon;
        const double f_plus = tensorSum(func(plus));
        const double f_minus = tensorSum(func(minus));
        gradients.data()[index] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return gradients;
}

inline double evaluateTargetObjective(const autodiff::Tensor &persistence_output,
                                      const OptimizationTarget &target)
{
    const double aggregate = tensorSum(persistence_output);
    const double residual = aggregate - target.target_value;
    return target.weight * residual * residual;
}

} // namespace nerve::persistence::detail
