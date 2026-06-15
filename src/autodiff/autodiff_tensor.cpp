
#include "nerve/autodiff/autodiff.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

namespace nerve::autodiff
{

namespace
{

constinit const double kEpsilon = 1e-12;

template <typename T>
concept ValidIndex = std::integral<T> && requires(T i) {
    { i } -> std::convertible_to<Size>;
};

} // namespace

Tensor::ScalarProxy::ScalarProxy(Tensor *tensor, Size index)
    : tensor_(tensor)
    , index_(index)
{}

Tensor::ScalarProxy &Tensor::ScalarProxy::operator=(double value)
{
    tensor_->mutableAt(index_) = value;
    return *this;
}

Tensor::ScalarProxy &Tensor::ScalarProxy::operator+=(double value)
{
    tensor_->mutableAt(index_) += value;
    return *this;
}

Tensor::ScalarProxy &Tensor::ScalarProxy::operator-=(double value)
{
    tensor_->mutableAt(index_) -= value;
    return *this;
}

Tensor::ScalarProxy &Tensor::ScalarProxy::operator*=(double value)
{
    tensor_->mutableAt(index_) *= value;
    return *this;
}

Tensor::ScalarProxy &Tensor::ScalarProxy::operator/=(double value)
{
    if (std::abs(value) <= kEpsilon)
    {
        throw std::invalid_argument("division by zero");
    }
    tensor_->mutableAt(index_) /= value;
    return *this;
}

Tensor::ScalarProxy::operator double() const
{
    return tensor_->at(index_);
}

[[nodiscard]] double &Tensor::ScalarProxy::operator[](Size inner_index)
{
    return tensor_->mutableAt(index_, inner_index);
}

const double &Tensor::ScalarProxy::operator[](Size inner_index) const
{
    return tensor_->at(index_, inner_index);
}

Tensor::ConstScalarProxy::ConstScalarProxy(const Tensor *tensor, Size index)
    : tensor_(tensor)
    , index_(index)
{}

Tensor::ConstScalarProxy::operator double() const
{
    return tensor_->at(index_);
}

const double &Tensor::ConstScalarProxy::operator[](Size inner_index) const
{
    return tensor_->at(index_, inner_index);
}

Tensor::Tensor()
    : data_()
    , shape_()
    , gradient_()
    , requires_grad_(false)
{}

Tensor::Tensor(const std::vector<double> &data, const Shape &shape)
    : data_(data)
    , shape_(shape)
    , gradient_(data.size(), 0.0)
    , requires_grad_(false)
{
    if (computeNumel(shape_) != data_.size())
    {
        throw std::invalid_argument("tensor shape does not match data size");
    }
}

Tensor::Tensor(const std::vector<double> &data)
    : Tensor(data, Shape{data.size()})
{}

Tensor::Tensor(double scalar)
    : Tensor(std::vector<double>{scalar}, Shape{1})
{}

Tensor Tensor::zeros(const Shape &shape)
{
    return Tensor(std::vector<double>(computeNumel(shape), 0.0), shape);
}

Tensor Tensor::ones(const Shape &shape)
{
    return Tensor(std::vector<double>(computeNumel(shape), 1.0), shape);
}

Tensor::ScalarProxy Tensor::operator[](Size index)
{
    return ScalarProxy(this, index);
}

Tensor::ConstScalarProxy Tensor::operator[](Size index) const
{
    return ConstScalarProxy(this, index);
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape_ == other.shape_)
    {
        std::vector<double> out(data_.size(), 0.0);
        for (Size i = 0; i < data_.size(); ++i)
        {
            out[i] = data_[i] + other.data_[i];
        }
        return Tensor(out, shape_);
    }
    if (other.size() == 1)
    {
        return (*this) + other.data_.front();
    }
    if (size() == 1)
    {
        return other + data_.front();
    }
    throw std::invalid_argument("tensor shape mismatch for addition");
}

Tensor Tensor::operator-(const Tensor &other) const
{
    if (shape_ == other.shape_)
    {
        std::vector<double> out(data_.size(), 0.0);
        for (Size i = 0; i < data_.size(); ++i)
        {
            out[i] = data_[i] - other.data_[i];
        }
        return Tensor(out, shape_);
    }
    if (other.size() == 1)
    {
        return (*this) - other.data_.front();
    }
    if (size() == 1)
    {
        return Tensor(std::vector<double>(other.size(), data_.front()), other.shape_) - other;
    }
    throw std::invalid_argument("tensor shape mismatch for subtraction");
}

Tensor Tensor::operator*(const Tensor &other) const
{
    if (shape_ == other.shape_)
    {
        std::vector<double> out(data_.size(), 0.0);
        for (Size i = 0; i < data_.size(); ++i)
        {
            out[i] = data_[i] * other.data_[i];
        }
        return Tensor(out, shape_);
    }
    if (other.size() == 1)
    {
        return (*this) * other.data_.front();
    }
    if (size() == 1)
    {
        return other * data_.front();
    }
    throw std::invalid_argument("tensor shape mismatch for multiplication");
}

Tensor Tensor::operator/(const Tensor &other) const
{
    if (shape_ == other.shape_)
    {
        std::vector<double> out(data_.size(), 0.0);
        for (Size i = 0; i < data_.size(); ++i)
        {
            if (std::abs(other.data_[i]) <= kEpsilon)
            {
                throw std::invalid_argument("division by zero");
            }
            out[i] = data_[i] / other.data_[i];
        }
        return Tensor(out, shape_);
    }
    if (other.size() == 1)
    {
        return (*this) / other.data_.front();
    }
    if (size() == 1)
    {
        std::vector<double> out(other.size(), 0.0);
        for (Size i = 0; i < other.size(); ++i)
        {
            if (std::abs(other.data_[i]) <= kEpsilon)
            {
                throw std::invalid_argument("division by zero");
            }
            out[i] = data_.front() / other.data_[i];
        }
        return Tensor(out, other.shape_);
    }
    throw std::invalid_argument("tensor shape mismatch for division");
}

Tensor Tensor::operator+(double scalar) const
{
    std::vector<double> out(data_);
    for (double &value : out)
    {
        value += scalar;
    }
    return Tensor(out, shape_);
}

Tensor Tensor::operator-(double scalar) const
{
    std::vector<double> out(data_);
    for (double &value : out)
    {
        value -= scalar;
    }
    return Tensor(out, shape_);
}

Tensor Tensor::operator*(double scalar) const
{
    std::vector<double> out(data_);
    for (double &value : out)
    {
        value *= scalar;
    }
    return Tensor(out, shape_);
}

Tensor Tensor::operator/(double scalar) const
{
    if (std::abs(scalar) <= kEpsilon)
    {
        throw std::invalid_argument("division by zero");
    }
    std::vector<double> out(data_);
    for (double &value : out)
    {
        value /= scalar;
    }
    return Tensor(out, shape_);
}

Tensor Tensor::relu() const
{
    std::vector<double> out(data_.size(), 0.0);
    for (Size i = 0; i < data_.size(); ++i)
    {
        out[i] = std::max(0.0, data_[i]);
    }
    return Tensor(out, shape_);
}

Tensor Tensor::sigmoid() const
{
    std::vector<double> out(data_.size(), 0.0);
    for (Size i = 0; i < data_.size(); ++i)
    {
        out[i] = 1.0 / (1.0 + std::exp(-data_[i]));
    }
    return Tensor(out, shape_);
}

Tensor Tensor::tanh() const
{
    std::vector<double> out(data_.size(), 0.0);
    for (Size i = 0; i < data_.size(); ++i)
    {
        out[i] = std::tanh(data_[i]);
    }
    return Tensor(out, shape_);
}

Tensor Tensor::sum() const
{
    return Tensor(std::accumulate(data_.begin(), data_.end(), 0.0));
}

Tensor Tensor::max() const
{
    if (data_.empty())
    {
        return Tensor(0.0);
    }
    return Tensor(*std::max_element(data_.begin(), data_.end()));
}

Tensor Tensor::min() const
{
    if (data_.empty())
    {
        return Tensor(0.0);
    }
    return Tensor(*std::min_element(data_.begin(), data_.end()));
}

std::vector<double> &Tensor::data()
{
    return data_;
}

const std::vector<double> &Tensor::data() const
{
    return data_;
}

const Shape &Tensor::shape() const
{
    return shape_;
}

Size Tensor::size() const
{
    return data_.size();
}

Size Tensor::ndim() const
{
    return shape_.size();
}

Tensor Tensor::grad() const
{
    return Tensor(gradient_, shape_);
}

void Tensor::setGrad(const Tensor &gradient)
{
    if (gradient.shape() != shape_)
    {
        throw std::invalid_argument("gradient shape must match tensor shape");
    }
    gradient_ = gradient.data();
}

void Tensor::zeroGrad()
{
    std::fill(gradient_.begin(), gradient_.end(), 0.0);
}

void Tensor::backward()
{
    if (!requires_grad_)
    {
        return;
    }
    gradient_.assign(data_.size(), 1.0);
}

void Tensor::setRequiresGrad(bool requiresGrad)
{
    requires_grad_ = requiresGrad;
}

bool Tensor::requiresGrad() const
{
    return requires_grad_;
}

Tensor Tensor::toTensor() const
{
    return *this;
}

Size Tensor::computeNumel(const Shape &shape)
{
    if (shape.empty())
    {
        return 0;
    }
    Size total = 1;
    for (const Size dim : shape)
    {
        if (dim != 0 && total > std::numeric_limits<Size>::max() / dim)
        {
            throw std::length_error("tensor shape element count overflows Size");
        }
        total *= dim;
    }
    return total;
}

Size Tensor::flatIndex(Size index) const
{
    if (index >= data_.size())
    {
        throw std::out_of_range("tensor index out of range");
    }
    return index;
}

Size Tensor::flatIndex(Size row, Size col) const
{
    if (shape_.size() < 2)
    {
        throw std::invalid_argument("tensor is not at least 2D");
    }
    if (row >= shape_[0] || col >= shape_[1])
    {
        throw std::out_of_range("tensor row/col index out of range");
    }
    return row * shape_[1] + col;
}

double &Tensor::mutableAt(Size index)
{
    return data_[flatIndex(index)];
}

const double &Tensor::at(Size index) const
{
    return data_[flatIndex(index)];
}

double &Tensor::mutableAt(Size row, Size col)
{
    return data_[flatIndex(row, col)];
}

const double &Tensor::at(Size row, Size col) const
{
    return data_[flatIndex(row, col)];
}

Tensor operator+(double scalar, const Tensor &tensor)
{
    return tensor + scalar;
}

Tensor operator-(double scalar, const Tensor &tensor)
{
    std::vector<double> out(tensor.data().size(), 0.0);
    for (Size i = 0; i < tensor.data().size(); ++i)
    {
        out[i] = scalar - tensor.data()[i];
    }
    return Tensor(out, tensor.shape());
}

Tensor operator*(double scalar, const Tensor &tensor)
{
    return tensor * scalar;
}

Tensor operator/(double scalar, const Tensor &tensor)
{
    std::vector<double> out(tensor.data().size(), 0.0);
    for (Size i = 0; i < tensor.data().size(); ++i)
    {
        if (std::abs(tensor.data()[i]) <= kEpsilon)
        {
            throw std::invalid_argument("division by zero");
        }
        out[i] = scalar / tensor.data()[i];
    }
    return Tensor(out, tensor.shape());
}

} // namespace nerve::autodiff
