
#include "nerve/autodiff/autodiff.hpp"

#include <concepts>

namespace nerve::autodiff
{

template <typename T>
concept ValidRequiresGrad = std::convertible_to<T, bool>;

Variable::Variable()
    : data_()
{}

Variable::Variable(const Tensor &data, bool requiresGrad)
    : data_(data)
{
    data_.setRequiresGrad(requiresGrad);
}

[[nodiscard]] Tensor Variable::data() const
{
    return data_;
}

[[nodiscard]] Tensor Variable::grad() const
{
    return data_.grad();
}

[[nodiscard]] bool Variable::requiresGrad() const
{
    return data_.requiresGrad();
}

void Variable::setRequiresGrad(bool requiresGrad)
{
    data_.setRequiresGrad(requiresGrad);
}

[[nodiscard]] Variable Variable::operator+(const Variable &other) const
{
    return Variable(data_ + other.data_, requiresGrad() || other.requiresGrad());
}

[[nodiscard]] Variable Variable::operator-(const Variable &other) const
{
    return Variable(data_ - other.data_, requiresGrad() || other.requiresGrad());
}

[[nodiscard]] Variable Variable::operator*(const Variable &other) const
{
    return Variable(data_ * other.data_, requiresGrad() || other.requiresGrad());
}

[[nodiscard]] Variable Variable::operator/(const Variable &other) const
{
    return Variable(data_ / other.data_, requiresGrad() || other.requiresGrad());
}

[[nodiscard]] Variable Variable::relu() const
{
    return Variable(data_.relu(), requiresGrad());
}

[[nodiscard]] Variable Variable::sigmoid() const
{
    return Variable(data_.sigmoid(), requiresGrad());
}

[[nodiscard]] Variable Variable::tanh() const
{
    return Variable(data_.tanh(), requiresGrad());
}

Variable Variable::sum() const
{
    return Variable(data_.sum(), requiresGrad());
}

Variable Variable::max() const
{
    return Variable(data_.max(), requiresGrad());
}

Variable Variable::min() const
{
    return Variable(data_.min(), requiresGrad());
}

void Variable::backward()
{
    data_.backward();
}

void Variable::zeroGrad()
{
    data_.zeroGrad();
}

} // namespace nerve::autodiff
