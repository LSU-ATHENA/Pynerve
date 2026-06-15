
#include "nerve/autodiff/autodiff.hpp"

#include <chrono>
#include <cmath>
#include <concepts>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nerve::autodiff
{
namespace
{

constexpr const char *kGraphFormat = "nerve_autodiff_graph_v1";

bool sameTensor(const Tensor &left, const Tensor &right)
{
    return left.shape() == right.shape() && left.data() == right.data();
}

Size findNodeIndex(const std::vector<Tensor> &nodes, const Tensor &tensor)
{
    for (Size i = 0; i < nodes.size(); ++i)
    {
        if (sameTensor(nodes[i], tensor))
        {
            return i;
        }
    }
    return nodes.size();
}

void writeTensor(std::ostream &out, const Tensor &tensor)
{
    out << "tensor " << tensor.data().size() << ' ' << tensor.shape().size();
    for (Size dim : tensor.shape())
    {
        out << ' ' << dim;
    }
    out << std::setprecision(17);
    for (double value : tensor.data())
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("cannot serialize non-finite autodiff tensor");
        }
        out << ' ' << value;
    }
    out << '\n';
}

Tensor readTensor(std::istream &in)
{
    std::string token;
    Size data_count = 0;
    Size shape_count = 0;
    if (!(in >> token >> data_count >> shape_count) || token != "tensor")
    {
        throw std::runtime_error("invalid tensor record in graph file");
    }
    Shape shape(shape_count);
    for (Size &dim : shape)
    {
        if (!(in >> dim))
        {
            throw std::runtime_error("invalid tensor shape in graph file");
        }
    }
    std::vector<double> data(data_count);
    for (double &value : data)
    {
        if (!(in >> value) || !std::isfinite(value))
        {
            throw std::runtime_error("invalid tensor value in graph file");
        }
    }
    return Tensor(data, shape);
}

} // namespace

template <typename T>
concept ValidGradientParams = std::floating_point<T> && requires(T e) {
    { e } -> std::convertible_to<double>;
    requires e > 0.0 && e < 1.0;
};

[[nodiscard]] bool AutoDiffUtils::checkGradient(const std::function<Tensor(const Tensor &)> &func,
                                                const Tensor &input, double epsilon)
{
    const Tensor analytical = computeJacobian(func, input);
    const Tensor numerical = numericalGradient(
        [func](const Tensor &x) {
            const Tensor out = func(x);
            return out.size() == 0 ? 0.0 : out.data().front();
        },
        input, epsilon);
    return tensorDistance(analytical, numerical) <= 1e-6;
}

[[nodiscard]] Tensor
AutoDiffUtils::computeJacobian(const std::function<Tensor(const Tensor &)> &func,
                               const Tensor &input)
{
    return numericalGradient(
        [func](const Tensor &x) {
            const Tensor out = func(x);
            return out.size() == 0 ? 0.0 : out.data().front();
        },
        input, 1e-6);
}

[[nodiscard]] Tensor
AutoDiffUtils::computeHessian(const std::function<Tensor(const Tensor &)> &func,
                              const Tensor &input)
{
    const Size n = input.size();
    std::vector<double> hessian(n * n, 0.0);
    const std::vector<double> base = input.data();

    for (Size i = 0; i < n; ++i)
    {
        std::vector<double> plus = base;
        std::vector<double> minus = base;
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        const Tensor grad_plus = computeJacobian(func, Tensor(plus, input.shape()));
        const Tensor grad_minus = computeJacobian(func, Tensor(minus, input.shape()));
        for (Size j = 0; j < n; ++j)
        {
            hessian[i * n + j] = (grad_plus.data()[j] - grad_minus.data()[j]) / (2e-6);
        }
    }
    return Tensor(hessian, Shape{n, n});
}

Tensor AutoDiffUtils::numericalGradient(const std::function<double(const Tensor &)> &func,
                                        const Tensor &input, double epsilon)
{
    if (epsilon <= 0.0)
    {
        throw std::invalid_argument("epsilon must be positive");
    }
    std::vector<double> gradValues(input.size(), 0.0);
    const std::vector<double> base = input.data();
    for (Size i = 0; i < base.size(); ++i)
    {
        std::vector<double> plus = base;
        std::vector<double> minus = base;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const double f_plus = func(Tensor(plus, input.shape()));
        const double f_minus = func(Tensor(minus, input.shape()));
        gradValues[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return Tensor(gradValues, input.shape());
}

void AutoDiffUtils::saveGraph(const ComputationalGraph &graph, const std::string &filename)
{
    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("failed to open graph file for writing");
    }
    const auto nodes = graph.getParameters();
    const auto edges = graph.getEdges();
    const auto edge_gradients = graph.getEdgeGradients();
    if (edges.size() != edge_gradients.size())
    {
        throw std::runtime_error("graph edge and gradient counts differ");
    }

    out << kGraphFormat << '\n';
    out << "nodes " << nodes.size() << '\n';
    for (const auto &node : nodes)
    {
        writeTensor(out, node);
    }
    out << "edges " << edges.size() << '\n';
    for (Size i = 0; i < edges.size(); ++i)
    {
        const Size from = findNodeIndex(nodes, edges[i].first);
        const Size to = findNodeIndex(nodes, edges[i].second);
        if (from == nodes.size() || to == nodes.size())
        {
            throw std::runtime_error("graph edge references a missing node");
        }
        out << "edge " << from << ' ' << to << '\n';
        writeTensor(out, edge_gradients[i]);
    }
}

ComputationalGraph AutoDiffUtils::loadGraph(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
    {
        throw std::runtime_error("failed to open graph file for reading");
    }
    std::string token;
    if (!(in >> token) || token != kGraphFormat)
    {
        throw std::runtime_error("unsupported autodiff graph file format");
    }

    Size node_count = 0;
    if (!(in >> token >> node_count) || token != "nodes")
    {
        throw std::runtime_error("graph file missing node section");
    }
    std::vector<Tensor> nodes;
    nodes.reserve(node_count);
    ComputationalGraph graph;
    for (Size i = 0; i < node_count; ++i)
    {
        nodes.push_back(readTensor(in));
        graph.addNode(nodes.back());
    }

    Size edge_count = 0;
    if (!(in >> token >> edge_count) || token != "edges")
    {
        throw std::runtime_error("graph file missing edge section");
    }
    for (Size i = 0; i < edge_count; ++i)
    {
        Size from = 0;
        Size to = 0;
        if (!(in >> token >> from >> to) || token != "edge" || from >= nodes.size() ||
            to >= nodes.size())
        {
            throw std::runtime_error("invalid edge record in graph file");
        }
        graph.addEdge(nodes[from], nodes[to], readTensor(in));
    }
    return graph;
}

void AutoDiffUtils::profileComputation(const std::function<void()> &computation,
                                       const std::string &name)
{
    const auto start = std::chrono::steady_clock::now();
    computation();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << name << " took " << elapsed << "ms\n";
}

void enableGrad(const Tensor &tensor)
{
    const_cast<Tensor &>(tensor).setRequiresGrad(true);
}

void backward(const Tensor &tensor)
{
    const_cast<Tensor &>(tensor).backward();
}

Tensor grad(const Tensor &tensor)
{
    return tensor.grad();
}

double tensorNorm(const Tensor &tensor)
{
    double sum_sq = 0.0;
    for (const double value : tensor.data())
    {
        sum_sq += value * value;
    }
    return std::sqrt(sum_sq);
}

double tensorDistance(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
    {
        throw std::invalid_argument("tensor shapes must match for distance");
    }
    double sum_sq = 0.0;
    for (Size i = 0; i < a.size(); ++i)
    {
        const double diff = a.data()[i] - b.data()[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

double tensorVariance(const Tensor &tensor)
{
    if (tensor.size() == 0)
    {
        return 0.0;
    }
    const double mean = std::accumulate(tensor.data().begin(), tensor.data().end(), 0.0) /
                        static_cast<double>(tensor.size());
    double sum_sq = 0.0;
    for (const double value : tensor.data())
    {
        const double diff = value - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(tensor.size());
}

double tensorSparsity(const Tensor &tensor, double epsilon)
{
    if (tensor.size() == 0)
    {
        return 1.0;
    }
    Size zeros = 0;
    for (const double value : tensor.data())
    {
        if (std::abs(value) <= epsilon)
        {
            ++zeros;
        }
    }
    return static_cast<double>(zeros) / static_cast<double>(tensor.size());
}

} // namespace nerve::autodiff
