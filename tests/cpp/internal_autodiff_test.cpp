
#include "nerve/autodiff/detail/autodiff_detail.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

namespace
{


constexpr double TOL = 1e-12;

bool check_tensor_construction_default()
{
    nerve::autodiff::Tensor t;
    return t.size() == 0 && t.ndim() == 0;
}

bool check_tensor_construction_with_data()
{
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    nerve::autodiff::Tensor t(data, {2, 2});
    if (t.size() != 4)
    {
        return false;
    }
    if (t.ndim() != 2)
    {
        return false;
    }
    auto shape = t.shape();
    return shape.size() == 2 && shape[0] == 2 && shape[1] == 2;
}

bool check_tensor_construction_scalar()
{
    nerve::autodiff::Tensor t(3.14);
    if (t.size() != 1)
    {
        return false;
    }
    return std::abs(t.data()[0] - 3.14) < TOL;
}

bool check_tensor_construction_1d()
{
    std::vector<double> data = {1.0, 2.0, 3.0};
    nerve::autodiff::Tensor t(data);
    if (t.size() != 3)
    {
        return false;
    }
    auto shape = t.shape();
    return shape.size() == 1 && shape[0] == 3;
}

bool check_tensor_zeros_ones()
{
    auto z = nerve::autodiff::Tensor::zeros({2, 3});
    if (z.size() != 6)
    {
        return false;
    }
    for (double v : z.data())
    {
        if (std::abs(v) > TOL)
        {
            return false;
        }
    }

    auto o = nerve::autodiff::Tensor::ones({3});
    for (double v : o.data())
    {
        if (std::abs(v - 1.0) > TOL)
        {
            return false;
        }
    }
    return true;
}

bool check_tensor_indexing()
{
    nerve::autodiff::Tensor t({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    if (std::abs(static_cast<double>(t[0]) - 1.0) > TOL)
    {
        return false;
    }
    if (std::abs(static_cast<double>(t[5]) - 6.0) > TOL)
    {
        return false;
    }
    if (std::abs(t[1][2] - 6.0) > TOL)
    {
        return false;
    }
    return true;
}

bool check_tensor_mutable_indexing()
{
    nerve::autodiff::Tensor t({0.0, 0.0, 0.0}, {3});
    t[0] = 42.0;
    return std::abs(t.data()[0] - 42.0) < TOL;
}

bool check_tensor_addition()
{
    nerve::autodiff::Tensor a({1.0, 2.0}, {2});
    nerve::autodiff::Tensor b({3.0, 4.0}, {2});
    auto c = a + b;
    if (c.size() != 2)
    {
        return false;
    }
    return std::abs(c.data()[0] - 4.0) < TOL && std::abs(c.data()[1] - 6.0) < TOL;
}

bool check_tensor_subtraction()
{
    nerve::autodiff::Tensor a({5.0, 7.0}, {2});
    nerve::autodiff::Tensor b({2.0, 3.0}, {2});
    auto c = a - b;
    return std::abs(c.data()[0] - 3.0) < TOL && std::abs(c.data()[1] - 4.0) < TOL;
}

bool check_tensor_multiplication()
{
    nerve::autodiff::Tensor a({2.0, 3.0}, {2});
    nerve::autodiff::Tensor b({4.0, 5.0}, {2});
    auto c = a * b;
    return std::abs(c.data()[0] - 8.0) < TOL && std::abs(c.data()[1] - 15.0) < TOL;
}

bool check_tensor_division()
{
    nerve::autodiff::Tensor a({10.0, 20.0}, {2});
    nerve::autodiff::Tensor b({2.0, 4.0}, {2});
    auto c = a / b;
    return std::abs(c.data()[0] - 5.0) < TOL && std::abs(c.data()[1] - 5.0) < TOL;
}

bool check_tensor_scalar_arith()
{
    nerve::autodiff::Tensor a({1.0, 2.0, 3.0}, {3});
    auto p = a + 10.0;
    auto m = a - 1.0;
    auto mul = a * 2.0;
    auto d = a / 2.0;
    return std::abs(p.data()[0] - 11.0) < TOL && std::abs(m.data()[1] - 1.0) < TOL &&
           std::abs(mul.data()[2] - 6.0) < TOL && std::abs(d.data()[0] - 0.5) < TOL;
}

bool check_tensor_relu()
{
    nerve::autodiff::Tensor t({-2.0, -1.0, 0.0, 1.0, 2.0}, {5});
    auto r = t.relu();
    return std::abs(r.data()[0]) < TOL && std::abs(r.data()[1]) < TOL &&
           std::abs(r.data()[2]) < TOL && std::abs(r.data()[3] - 1.0) < TOL &&
           std::abs(r.data()[4] - 2.0) < TOL;
}

bool check_tensor_sigmoid()
{
    nerve::autodiff::Tensor t({0.0}, {1});
    auto s = t.sigmoid();
    return std::abs(s.data()[0] - 0.5) < 1e-6;
}

bool check_tensor_sum()
{
    nerve::autodiff::Tensor t({1.0, 2.0, 3.0, 4.0}, {4});
    auto s = t.sum();
    return std::abs(s.data()[0] - 10.0) < TOL;
}

bool check_tensor_backward()
{
    nerve::autodiff::Tensor t({2.0, 4.0}, {2});
    t.setRequiresGrad(true);
    t.backward();
    auto g = t.grad();
    if (g.size() != 2)
    {
        return false;
    }
    return std::abs(g.data()[0] - 1.0) < TOL && std::abs(g.data()[1] - 1.0) < TOL;
}

bool check_tensor_zero_grad()
{
    nerve::autodiff::Tensor t({1.0, 2.0}, {2});
    t.setRequiresGrad(true);
    t.backward();
    t.zeroGrad();
    auto g = t.grad();
    return std::abs(g.data()[0]) < TOL && std::abs(g.data()[1]) < TOL;
}

bool check_tensor_set_grad()
{
    nerve::autodiff::Tensor t({1.0, 2.0, 3.0}, {3});
    nerve::autodiff::Tensor g({0.5, 1.0, 1.5}, {3});
    t.setGrad(g);
    auto actual = t.grad();
    return std::abs(actual.data()[0] - 0.5) < TOL && std::abs(actual.data()[1] - 1.0) < TOL &&
           std::abs(actual.data()[2] - 1.5) < TOL;
}

bool check_tensor_requires_grad()
{
    nerve::autodiff::Tensor t({1.0}, {1});
    if (t.requiresGrad())
    {
        return false;
    }
    t.setRequiresGrad(true);
    return t.requiresGrad();
}

bool check_computational_graph_add_node()
{
    nerve::autodiff::ComputationalGraph graph;
    nerve::autodiff::Tensor t({1.0, 2.0}, {2});
    graph.addNode(t);
    auto params = graph.getParameters();
    return params.size() == 1;
}

bool check_computational_graph_add_edge()
{
    nerve::autodiff::ComputationalGraph graph;
    nerve::autodiff::Tensor x({2.0}, {1});
    nerve::autodiff::Tensor y({4.0}, {1});
    nerve::autodiff::Tensor local_grad({4.0}, {1});

    graph.addNode(x);
    graph.addNode(y);
    graph.addEdge(x, y, local_grad);

    auto edges = graph.getEdges();
    if (edges.size() != 1)
    {
        return false;
    }
    auto eg = graph.getEdgeGradients();
    return eg.size() == 1;
}

bool check_gradient_computation_y_equals_x_sq()
{
    nerve::autodiff::ComputationalGraph graph;

    nerve::autodiff::Tensor x({2.0}, {1});
    nerve::autodiff::Tensor y({4.0}, {1});
    nerve::autodiff::Tensor local_grad({4.0}, {1});

    graph.addNode(x);
    graph.addNode(y);
    graph.addEdge(x, y, local_grad);
    graph.backward();

    auto params = graph.getParameters();
    if (params.size() < 1)
    {
        return false;
    }

    auto grad_x = params[0].grad();
    if (grad_x.size() != 1)
    {
        return false;
    }

    return std::abs(grad_x.data()[0] - 4.0) < TOL;
}

bool check_gradient_computation_zero_grad()
{
    nerve::autodiff::ComputationalGraph graph;

    nerve::autodiff::Tensor x({3.0}, {1});
    nerve::autodiff::Tensor y({9.0}, {1});
    nerve::autodiff::Tensor local_grad({6.0}, {1});

    graph.addNode(x);
    graph.addNode(y);
    graph.addEdge(x, y, local_grad);
    graph.backward();

    graph.zeroGrad();

    auto params = graph.getParameters();
    if (params.empty())
    {
        return false;
    }
    auto g = params[0].grad();
    return std::abs(g.data()[0]) < TOL;
}

bool check_graph_clear()
{
    nerve::autodiff::ComputationalGraph graph;
    nerve::autodiff::Tensor t({1.0}, {1});
    graph.addNode(t);
    graph.clear();
    return graph.getParameters().empty();
}

bool check_graph_optimizer_optimize()
{
    nerve::autodiff::ComputationalGraph graph;

    nerve::autodiff::Tensor x({1.0}, {1});
    nerve::autodiff::Tensor y({2.0}, {1});
    nerve::autodiff::Tensor g({2.0}, {1});

    graph.addNode(x);
    graph.addNode(y);
    graph.addEdge(x, y, g);
    graph.optimize();

    return graph.getEdges().size() == 1;
}

bool check_autodiff_utils_numerical_gradient()
{
    auto func = [](const nerve::autodiff::Tensor &x) -> double {
        return x.data()[0] * x.data()[0];
    };

    nerve::autodiff::Tensor input({3.0}, {1});
    auto grad = nerve::autodiff::AutoDiffUtils::numericalGradient(func, input, 1e-6);
    if (grad.size() != 1)
    {
        return false;
    }
    return std::abs(grad.data()[0] - 6.0) < 0.01;
}

bool check_tensor_distance()
{
    nerve::autodiff::Tensor a({1.0, 0.0}, {2});
    nerve::autodiff::Tensor b({4.0, 0.0}, {2});
    double d = nerve::autodiff::tensorDistance(a, b);
    return std::abs(d - 3.0) < TOL;
}

bool check_tensor_norm()
{
    nerve::autodiff::Tensor t({3.0, 4.0}, {2});
    double n = nerve::autodiff::tensorNorm(t);
    return std::abs(n - 5.0) < TOL;
}

bool check_tensor_variance()
{
    nerve::autodiff::Tensor t({2.0, 4.0, 6.0}, {3});
    double v = nerve::autodiff::tensorVariance(t);
    return std::abs(v - 8.0 / 3.0) < TOL;
}

bool check_tensor_sparsity()
{
    nerve::autodiff::Tensor t({0.0, 1.0, 0.0, 2.0, 0.0}, {5});
    double s = nerve::autodiff::tensorSparsity(t, TOL);
    return std::abs(s - 0.6) < TOL;
}

bool check_enable_grad_and_backward()
{
    nerve::autodiff::Tensor t({5.0}, {1});
    nerve::autodiff::enableGrad(t);
    nerve::autodiff::backward(t);
    auto g = nerve::autodiff::grad(t);
    if (g.size() != 1)
    {
        return false;
    }
    return std::abs(g.data()[0] - 1.0) < TOL;
}

#ifdef NERVE_HAS_AVX512

bool check_simd_backward_add()
{
    double grad_a[4] = {1.0, 2.0, 3.0, 4.0};
    double grad_out[4] = {0.5, 1.0, 1.5, 2.0};
    nerve::autodiff::simdBackwardAdd(grad_a, grad_out, 4);
    return std::abs(grad_a[0] - 1.5) < TOL && std::abs(grad_a[1] - 3.0) < TOL &&
           std::abs(grad_a[2] - 4.5) < TOL && std::abs(grad_a[3] - 6.0) < TOL;
}

bool check_simd_backward_mul()
{
    double grad_a[4] = {1.0, 1.0, 1.0, 1.0};
    double grad_out[4] = {2.0, 3.0, 4.0, 5.0};
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    nerve::autodiff::simdBackwardMul(grad_a, grad_out, b, 4);
    return std::abs(grad_a[0] - 3.0) < TOL && std::abs(grad_a[1] - 7.0) < TOL &&
           std::abs(grad_a[2] - 13.0) < TOL && std::abs(grad_a[3] - 21.0) < TOL;
}

bool check_simd_backward_relu()
{
    double grad_a[4] = {0.0, 0.0, 0.0, 0.0};
    double grad_out[4] = {1.0, 2.0, 3.0, 4.0};
    double input[4] = {-1.0, 0.0, 1.0, 2.0};
    nerve::autodiff::simdBackwardRelu(grad_a, grad_out, input, 4);
    return std::abs(grad_a[0]) < TOL && std::abs(grad_a[1]) < TOL &&
           std::abs(grad_a[2] - 3.0) < TOL && std::abs(grad_a[3] - 4.0) < TOL;
}

bool check_simd_backward_add_small()
{
    double grad_a[2] = {1.0, 2.0};
    double grad_out[2] = {3.0, 4.0};
    nerve::autodiff::simdBackwardAdd(grad_a, grad_out, 2);
    return std::abs(grad_a[0] - 4.0) < TOL && std::abs(grad_a[1] - 6.0) < TOL;
}

#endif

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("tensor_construction_default", check_tensor_construction_default());
    run("tensor_construction_with_data", check_tensor_construction_with_data());
    run("tensor_construction_scalar", check_tensor_construction_scalar());
    run("tensor_construction_1d", check_tensor_construction_1d());
    run("tensor_zeros_ones", check_tensor_zeros_ones());
    run("tensor_indexing", check_tensor_indexing());
    run("tensor_mutable_indexing", check_tensor_mutable_indexing());
    run("tensor_addition", check_tensor_addition());
    run("tensor_subtraction", check_tensor_subtraction());
    run("tensor_multiplication", check_tensor_multiplication());
    run("tensor_division", check_tensor_division());
    run("tensor_scalar_arith", check_tensor_scalar_arith());
    run("tensor_relu", check_tensor_relu());
    run("tensor_sigmoid", check_tensor_sigmoid());
    run("tensor_sum", check_tensor_sum());
    run("tensor_backward", check_tensor_backward());
    run("tensor_zero_grad", check_tensor_zero_grad());
    run("tensor_set_grad", check_tensor_set_grad());
    run("tensor_requires_grad", check_tensor_requires_grad());
    run("computational_graph_add_node", check_computational_graph_add_node());
    run("computational_graph_add_edge", check_computational_graph_add_edge());
    run("gradient_y_equals_x_sq", check_gradient_computation_y_equals_x_sq());
    run("gradient_zero_grad", check_gradient_computation_zero_grad());
    run("graph_clear", check_graph_clear());
    run("graph_optimizer_optimize", check_graph_optimizer_optimize());
    run("autodiff_utils_numerical_gradient", check_autodiff_utils_numerical_gradient());
    run("tensor_distance", check_tensor_distance());
    run("tensor_norm", check_tensor_norm());
    run("tensor_variance", check_tensor_variance());
    run("tensor_sparsity", check_tensor_sparsity());
    run("enable_grad_and_backward", check_enable_grad_and_backward());

#ifdef NERVE_HAS_AVX512
    run("simd_backward_add", check_simd_backward_add());
    run("simd_backward_mul", check_simd_backward_mul());
    run("simd_backward_relu", check_simd_backward_relu());
    run("simd_backward_add_small", check_simd_backward_add_small());
#endif

    return failures > 0 ? 1 : 0;
}
