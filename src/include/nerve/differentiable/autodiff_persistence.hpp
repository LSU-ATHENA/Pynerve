#pragma once

#include "nerve/algebra/simplex.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef TOPO_ENABLE_ADVANCED_DIFFERENTIABLE
namespace nerve::differentiable
{

template <std::floating_point T = double>
class AutodiffScalar
{
public:
    using ValueType = T;

    AutodiffScalar() = default;
    explicit AutodiffScalar(T value)
        : value_(value)
    {}
    AutodiffScalar(T value, T grad)
        : value_(value)
        , gradient_(grad)
        , requires_grad_(true)
    {}
    AutodiffScalar(T value, T grad, bool requires_grad)
        : value_(value)
        , gradient_(requires_grad ? grad : T(0))
        , requires_grad_(requires_grad)
    {}

    [[nodiscard]] const T &value() const noexcept { return value_; }
    [[nodiscard]] const T &grad() const noexcept { return gradient_; }
    [[nodiscard]] bool requiresGradient() const noexcept { return requires_grad_; }

    void enableGradient() noexcept { requires_grad_ = true; }
    void disableGradient() noexcept
    {
        requires_grad_ = false;
        gradient_ = T(0);
    }
    void setGradient(T grad) noexcept
    {
        gradient_ = grad;
        requires_grad_ = true;
    }
    void resetGradient() noexcept { gradient_ = T(0); }

    AutodiffScalar &operator=(T value) noexcept
    {
        value_ = value;
        return *this;
    }

    [[nodiscard]] AutodiffScalar operator+(const AutodiffScalar &other) const
    {
        const bool tracked = requires_grad_ || other.requires_grad_;
        return {value_ + other.value_, localGrad() + other.localGrad(), tracked};
    }
    [[nodiscard]] AutodiffScalar operator-(const AutodiffScalar &other) const
    {
        const bool tracked = requires_grad_ || other.requires_grad_;
        return {value_ - other.value_, localGrad() - other.localGrad(), tracked};
    }
    [[nodiscard]] AutodiffScalar operator*(const AutodiffScalar &other) const
    {
        const bool tracked = requires_grad_ || other.requires_grad_;
        return {value_ * other.value_, localGrad() * other.value_ + other.localGrad() * value_,
                tracked};
    }
    [[nodiscard]] AutodiffScalar operator/(const AutodiffScalar &other) const
    {
        if (std::abs(other.value_) <= std::numeric_limits<T>::epsilon())
        {
            throw std::domain_error("autodiff division by zero");
        }
        const bool tracked = requires_grad_ || other.requires_grad_;
        const T denom = other.value_ * other.value_;
        return {value_ / other.value_,
                (localGrad() * other.value_ - value_ * other.localGrad()) / denom, tracked};
    }

    [[nodiscard]] AutodiffScalar operator+(T other) const
    {
        return {value_ + other, localGrad(), requires_grad_};
    }
    [[nodiscard]] AutodiffScalar operator-(T other) const
    {
        return {value_ - other, localGrad(), requires_grad_};
    }
    [[nodiscard]] AutodiffScalar operator*(T other) const
    {
        return {value_ * other, localGrad() * other, requires_grad_};
    }
    [[nodiscard]] AutodiffScalar operator/(T other) const
    {
        if (std::abs(other) <= std::numeric_limits<T>::epsilon())
        {
            throw std::domain_error("autodiff division by zero");
        }
        return {value_ / other, localGrad() / other, requires_grad_};
    }

    [[nodiscard]] bool operator<(const AutodiffScalar &other) const noexcept
    {
        return value_ < other.value_;
    }
    [[nodiscard]] bool operator<=(const AutodiffScalar &other) const noexcept
    {
        return value_ <= other.value_;
    }
    [[nodiscard]] bool operator>(const AutodiffScalar &other) const noexcept
    {
        return value_ > other.value_;
    }
    [[nodiscard]] bool operator>=(const AutodiffScalar &other) const noexcept
    {
        return value_ >= other.value_;
    }
    [[nodiscard]] bool operator==(const AutodiffScalar &other) const noexcept
    {
        return value_ == other.value_;
    }
    [[nodiscard]] bool operator!=(const AutodiffScalar &other) const noexcept
    {
        return value_ != other.value_;
    }

    [[nodiscard]] static AutodiffScalar abs(const AutodiffScalar &x)
    {
        const T sign = x.value_ > T(0) ? T(1) : (x.value_ < T(0) ? T(-1) : T(0));
        return {std::abs(x.value_), sign * x.localGrad(), x.requires_grad_};
    }
    [[nodiscard]] static AutodiffScalar sqrt(const AutodiffScalar &x)
    {
        if (x.value_ < T(0))
        {
            throw std::domain_error("autodiff sqrt requires a non-negative value");
        }
        const T root = std::sqrt(x.value_);
        const T grad = root > T(0) ? x.localGrad() / (T(2) * root) : T(0);
        return {root, grad, x.requires_grad_};
    }
    [[nodiscard]] static AutodiffScalar exp(const AutodiffScalar &x)
    {
        const T value = std::exp(x.value_);
        return {value, value * x.localGrad(), x.requires_grad_};
    }
    [[nodiscard]] static AutodiffScalar log(const AutodiffScalar &x)
    {
        if (x.value_ <= T(0))
        {
            throw std::domain_error("autodiff log requires a positive value");
        }
        return {std::log(x.value_), x.localGrad() / x.value_, x.requires_grad_};
    }
    [[nodiscard]] static AutodiffScalar min(const AutodiffScalar &a, const AutodiffScalar &b)
    {
        if (a.value_ < b.value_)
        {
            return a;
        }
        if (b.value_ < a.value_)
        {
            return b;
        }
        const bool tracked = a.requires_grad_ || b.requires_grad_;
        return {a.value_, (a.localGrad() + b.localGrad()) / T(2), tracked};
    }
    [[nodiscard]] static AutodiffScalar max(const AutodiffScalar &a, const AutodiffScalar &b)
    {
        if (a.value_ > b.value_)
        {
            return a;
        }
        if (b.value_ > a.value_)
        {
            return b;
        }
        const bool tracked = a.requires_grad_ || b.requires_grad_;
        return {a.value_, (a.localGrad() + b.localGrad()) / T(2), tracked};
    }

private:
    [[nodiscard]] T localGrad() const noexcept { return requires_grad_ ? gradient_ : T(0); }

    T value_ = T(0);
    T gradient_ = T(0);
    bool requires_grad_ = false;
};

template <typename T = double>
struct AutodiffPersistencePair
{
    AutodiffScalar<T> birth;
    AutodiffScalar<T> death;
    AutodiffScalar<T> persistence;
    AutodiffScalar<T> dimension;

    AutodiffPersistencePair() = default;
    AutodiffPersistencePair(T b, T d, T p, T dim)
        : birth(b)
        , death(d)
        , persistence(p)
        , dimension(dim)
    {}

    [[nodiscard]] ::nerve::persistence::Pair toStandardPair() const
    {
        return ::nerve::persistence::Pair{static_cast<::nerve::Field>(birth.value()),
                                          static_cast<::nerve::Field>(death.value()),
                                          static_cast<::nerve::Dimension>(dimension.value())};
    }
};

template <typename T = double>
class AutodiffPersistenceDiagram
{
public:
    using PairType = AutodiffPersistencePair<T>;
    using ScalarType = AutodiffScalar<T>;

    void addPair(const PairType &pair) { pairs_.push_back(pair); }
    [[nodiscard]] const std::vector<PairType> &getPairs() const noexcept { return pairs_; }
    [[nodiscard]] std::vector<PairType> &getPairs() noexcept { return pairs_; }

    void enableGradients()
    {
        for (auto &pair : pairs_)
        {
            pair.birth.enableGradient();
            pair.death.enableGradient();
            pair.persistence.enableGradient();
            pair.dimension.enableGradient();
        }
    }
    void resetGradients()
    {
        for (auto &pair : pairs_)
        {
            pair.birth.resetGradient();
            pair.death.resetGradient();
            pair.persistence.resetGradient();
            pair.dimension.resetGradient();
        }
    }

    [[nodiscard]] ::nerve::persistence::Diagram toStandardDiagram() const
    {
        ::nerve::persistence::Diagram diagram;
        for (const auto &pair : pairs_)
        {
            diagram.addPair(pair.toStandardPair());
        }
        return diagram;
    }

private:
    std::vector<PairType> pairs_;
};

template <typename T = double>
class DifferentiablePersistence
{
public:
    using ScalarType = AutodiffScalar<T>;
    using DiagramType = AutodiffPersistenceDiagram<T>;
    using PairType = AutodiffPersistencePair<T>;

    struct ComputationConfig
    {
        bool enableGradients = true;
        bool track_intermediate_values = false;
        T gradient_tolerance = T(1e-8);
        size_t max_iterations = 1000;
        bool validateGradients = true;
    };

    explicit DifferentiablePersistence(const ComputationConfig &config = {});

    DiagramType compute(const std::vector<algebra::Simplex> &complex,
                        const core::DeterminismContract &contract = {});
    DiagramType
    computePersistenceWithGradients(const std::vector<algebra::Simplex> &complex,
                                    std::function<void(const DiagramType &)> gradient_callback = {},
                                    const core::DeterminismContract &contract = {});
    void computeGradients(const std::vector<algebra::Simplex> &complex,
                          const std::vector<ScalarType> &loss_gradients,
                          const core::DeterminismContract &contract = {});

    [[nodiscard]] bool validateGradients(const std::vector<algebra::Simplex> &complex,
                                         T finite_difference_tolerance = T(1e-6)) const;
    void setConfig(const ComputationConfig &config);
    [[nodiscard]] ComputationConfig getConfig() const;
    [[nodiscard]] const DiagramType &getLastDiagram() const;
    [[nodiscard]] const std::vector<ScalarType> &getIntermediateValues() const;

private:
    ComputationConfig config_;
    DiagramType last_diagram_;
    std::vector<ScalarType> intermediate_values_;

    [[nodiscard]] DiagramType
    computeForwardPass(const std::vector<algebra::Simplex> &complex) const;
    void computeBackwardPass(const std::vector<ScalarType> &loss_gradients);
    [[nodiscard]] std::vector<ScalarType>
    computeFiniteDifferenceGradients(const std::vector<algebra::Simplex> &complex,
                                     T epsilon = T(1e-8)) const;
    [[nodiscard]] bool isAutodiffSafeFunction(const std::function<void()> &func) const;
};

namespace autodiff_safe
{
template <std::invocable Func>
[[nodiscard]] bool isSafe(const Func &func)
{
    try
    {
        func();
        return true;
    }
    catch (...)
    {
        return false;
    }
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeAbs(const AutodiffScalar<T> &x)
{
    return AutodiffScalar<T>::abs(x);
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeSqrt(const AutodiffScalar<T> &x)
{
    return AutodiffScalar<T>::sqrt(x);
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeExp(const AutodiffScalar<T> &x)
{
    return AutodiffScalar<T>::exp(x);
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeLog(const AutodiffScalar<T> &x)
{
    return AutodiffScalar<T>::log(x);
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeMin(const AutodiffScalar<T> &a, const AutodiffScalar<T> &b)
{
    return AutodiffScalar<T>::min(a, b);
}

template <std::floating_point T>
[[nodiscard]] AutodiffScalar<T> safeMax(const AutodiffScalar<T> &a, const AutodiffScalar<T> &b)
{
    return AutodiffScalar<T>::max(a, b);
}
} // namespace autodiff_safe

namespace autodiff_unsafe
{
template <std::invocable Func>
[[nodiscard]] bool isUnsafe(const Func &func)
{
    return !autodiff_safe::isSafe(func);
}

template <std::floating_point T>
[[nodiscard]] T unsafeFloor(const AutodiffScalar<T> &x)
{
    return std::floor(x.value());
}

template <std::floating_point T>
[[nodiscard]] T unsafeCeil(const AutodiffScalar<T> &x)
{
    return std::ceil(x.value());
}

template <std::floating_point T>
[[nodiscard]] bool unsafeIsnan(const AutodiffScalar<T> &x)
{
    return std::isnan(x.value());
}

template <std::floating_point T>
[[nodiscard]] bool unsafeIsfinite(const AutodiffScalar<T> &x)
{
    return std::isfinite(x.value());
}
} // namespace autodiff_unsafe

} // namespace nerve::differentiable
#endif
