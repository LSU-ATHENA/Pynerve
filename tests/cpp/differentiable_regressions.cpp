#ifdef TOPO_ENABLE_ADVANCED_DIFFERENTIABLE
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/differentiable/autodiff_persistence.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

constexpr double kTol = 1e-10;

bool check_autodiff_scalar_construction()
{
    nerve::differentiable::AutodiffScalar<> s;
    if (std::abs(s.value()) > kTol)
    {
        std::cerr << "default value should be 0\n";
        return false;
    }
    if (s.requiresGradient())
    {
        std::cerr << "default should not require gradient\n";
        return false;
    }

    nerve::differentiable::AutodiffScalar<> s2(3.14);
    if (std::abs(s2.value() - 3.14) > kTol)
    {
        std::cerr << "value constructor failed\n";
        return false;
    }

    nerve::differentiable::AutodiffScalar<> s3(2.0, 1.0);
    if (std::abs(s3.value() - 2.0) > kTol || std::abs(s3.grad() - 1.0) > kTol)
    {
        std::cerr << "value+grad constructor failed\n";
        return false;
    }
    if (!s3.requiresGradient())
    {
        std::cerr << "should require grad when grad non-zero\n";
        return false;
    }

    return true;
}

bool check_autodiff_addition()
{
    nerve::differentiable::AutodiffScalar<> a(2.0, 1.0);
    nerve::differentiable::AutodiffScalar<> b(3.0, 0.5);

    auto c = a + b;
    if (std::abs(c.value() - 5.0) > kTol)
    {
        std::cerr << "addition value wrong: " << c.value() << "\n";
        return false;
    }

    auto d = a + 10.0;
    if (std::abs(d.value() - 12.0) > kTol)
    {
        std::cerr << "scalar addition wrong: " << d.value() << "\n";
        return false;
    }

    return true;
}

bool check_autodiff_multiplication()
{
    nerve::differentiable::AutodiffScalar<> a(3.0, 1.0);
    nerve::differentiable::AutodiffScalar<> b(2.0, 0.0);

    auto c = a * b;
    if (std::abs(c.value() - 6.0) > kTol)
    {
        std::cerr << "multiplication value wrong: " << c.value() << "\n";
        return false;
    }

    auto d = a * 4.0;
    if (std::abs(d.value() - 12.0) > kTol)
    {
        std::cerr << "scalar multiplication wrong: " << d.value() << "\n";
        return false;
    }

    return true;
}

bool check_autodiff_subtraction()
{
    nerve::differentiable::AutodiffScalar<> a(5.0, 1.0);
    nerve::differentiable::AutodiffScalar<> b(3.0, 0.5);

    auto c = a - b;
    if (std::abs(c.value() - 2.0) > kTol)
    {
        std::cerr << "subtraction value wrong: " << c.value() << "\n";
        return false;
    }

    return true;
}

bool check_autodiff_division()
{
    nerve::differentiable::AutodiffScalar<> a(6.0, 1.0);
    nerve::differentiable::AutodiffScalar<> b(2.0, 0.0);

    auto c = a / b;
    if (std::abs(c.value() - 3.0) > kTol)
    {
        std::cerr << "division value wrong: " << c.value() << "\n";
        return false;
    }

    auto d = a / 2.0;
    if (std::abs(d.value() - 3.0) > kTol)
    {
        std::cerr << "scalar division wrong: " << d.value() << "\n";
        return false;
    }

    return true;
}

bool check_autodiff_division_by_zero()
{
    nerve::differentiable::AutodiffScalar<> a(1.0);
    nerve::differentiable::AutodiffScalar<> b(0.0);

    bool threw = false;
    try
    {
        auto c = a / b;
        static_cast<void>(c);
    }
    catch (const std::domain_error &)
    {
        threw = true;
    }
    catch (...)
    {}
    if (!threw)
    {
        std::cerr << "division by zero should throw domain_error\n";
        return false;
    }

    return true;
}

bool check_autodiff_gradient_control()
{
    nerve::differentiable::AutodiffScalar<> s(1.0);
    if (s.requiresGradient())
    {
        std::cerr << "default should not require grad\n";
        return false;
    }

    s.enableGradient();
    if (!s.requiresGradient())
    {
        std::cerr << "enableGradient failed\n";
        return false;
    }

    s.setGradient(2.0);
    if (std::abs(s.grad() - 2.0) > kTol)
    {
        std::cerr << "setGradient failed\n";
        return false;
    }

    s.resetGradient();
    if (std::abs(s.grad()) > kTol)
    {
        std::cerr << "resetGradient should zero grad\n";
        return false;
    }

    s.disableGradient();
    if (s.requiresGradient())
    {
        std::cerr << "disableGradient failed\n";
        return false;
    }

    return true;
}

bool check_autodiff_comparison()
{
    nerve::differentiable::AutodiffScalar<> a(2.0);
    nerve::differentiable::AutodiffScalar<> b(3.0);

    if (!(a < b))
    {
        std::cerr << "2 < 3 should be true\n";
        return false;
    }
    if (a > b)
    {
        std::cerr << "2 > 3 should be false\n";
        return false;
    }

    nerve::differentiable::AutodiffScalar<> c(2.0);
    if (!(a == c))
    {
        std::cerr << "2 == 2 should be true\n";
        return false;
    }
    if (a == b)
    {
        std::cerr << "2 == 3 should be false\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_autodiff_scalar_construction())
    {
        std::cerr << "FAIL: autodiff scalar construction\n";
        return 1;
    }
    if (!check_autodiff_addition())
    {
        std::cerr << "FAIL: autodiff addition\n";
        return 1;
    }
    if (!check_autodiff_multiplication())
    {
        std::cerr << "FAIL: autodiff multiplication\n";
        return 1;
    }
    if (!check_autodiff_subtraction())
    {
        std::cerr << "FAIL: autodiff subtraction\n";
        return 1;
    }
    if (!check_autodiff_division())
    {
        std::cerr << "FAIL: autodiff division\n";
        return 1;
    }
    if (!check_autodiff_division_by_zero())
    {
        std::cerr << "FAIL: autodiff division by zero\n";
        return 1;
    }
    if (!check_autodiff_gradient_control())
    {
        std::cerr << "FAIL: autodiff gradient control\n";
        return 1;
    }
    if (!check_autodiff_comparison())
    {
        std::cerr << "FAIL: autodiff comparison\n";
        return 1;
    }
    return 0;
}
#endif
#ifndef TOPO_ENABLE_ADVANCED_DIFFERENTIABLE
int main()
{
    return 0;
}
#endif
