// PyTorch Autograd Integration for Differentiable Persistence
// Implements torch::autograd::Function for gradient flow through topology

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace nerve::torch
{

// Helper: Union-Find for Persistence Computation

namespace detail
{

[[nodiscard]] at::Tensor validate_filtration_cpu(const at::Tensor &filtration, int64_t max_dim,
                                                 const char *name)
{
    TORCH_CHECK(max_dim == 0, name, " currently supports max_dim=0 only");
    TORCH_CHECK(filtration.defined(), name, " must be defined");
    TORCH_CHECK(filtration.dim() == 1, name, " must be 1D");
    TORCH_CHECK(filtration.size(0) > 0, name, " must be non-empty");
    TORCH_CHECK(filtration.is_floating_point(), name, " must be a floating-point tensor");
    at::Tensor filt_cpu = filtration.contiguous().cpu().to(at::kDouble);
    TORCH_CHECK(at::isfinite(filt_cpu).all().item<bool>(), name,
                " must contain only finite values");
    return filt_cpu;
}

struct UnionFind
{
    std::vector<int64_t> parent;
    std::vector<int64_t> rank;
    std::vector<double> birth_time;

    explicit UnionFind(int64_t n)
        : parent(n)
        , rank(n, 0)
        , birth_time(n)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int64_t find(int64_t x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unite(int64_t x, int64_t y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
            return;

        if (rank[x] < rank[y])
        {
            std::swap(x, y);
        }
        parent[y] = x;
        if (rank[x] == rank[y])
        {
            rank[x]++;
        }
    }
};

[[nodiscard]] std::vector<std::tuple<int64_t, int64_t, double>>
compute_0d_persistence(const at::Tensor &filtration_values,
                       const std::vector<std::pair<int64_t, int64_t>> &edges)
{
    const int64_t n = filtration_values.size(0);
    auto accessor = filtration_values.accessor<double, 1>();

    std::vector<std::tuple<double, int64_t, int64_t, int64_t>> sorted_edges;
    for (int64_t e = 0; e < static_cast<int64_t>(edges.size()); ++e)
    {
        auto [u, v] = edges[e];
        double death = std::max(accessor[u], accessor[v]);
        sorted_edges.emplace_back(death, e, u, v);
    }
    std::sort(sorted_edges.begin(), sorted_edges.end());

    UnionFind uf(n);
    for (int64_t i = 0; i < n; ++i)
    {
        uf.birth_time[i] = accessor[i];
    }

    std::vector<std::tuple<int64_t, int64_t, double>> persistence_pairs;

    for (const auto &[death_time, edge_idx, u, v] : sorted_edges)
    {
        int64_t cu = uf.find(u);
        int64_t cv = uf.find(v);

        if (cu != cv)
        {
            double birth_u = uf.birth_time[cu];
            double birth_v = uf.birth_time[cv];

            if (birth_u > birth_v)
            {
                persistence_pairs.emplace_back(cu, edge_idx, death_time);
                uf.birth_time[cv] = std::min(birth_u, birth_v);
            }
            else
            {
                persistence_pairs.emplace_back(cv, edge_idx, death_time);
                uf.birth_time[cu] = std::min(birth_u, birth_v);
            }

            uf.unite(cu, cv);
        }
    }

    for (int64_t i = 0; i < n; ++i)
    {
        if (uf.find(i) == i)
        {
            persistence_pairs.emplace_back(i, -1, std::numeric_limits<double>::infinity());
        }
    }

    return persistence_pairs;
}

} // namespace detail

// PH Compute Function (Non-differentiable)

at::Tensor ph_compute(const at::Tensor &filtration, int64_t max_dim)
{
    at::Tensor filt_cpu = detail::validate_filtration_cpu(filtration, max_dim, "filtration");
    const int64_t n = filt_cpu.size(0);

    std::vector<std::pair<int64_t, int64_t>> edges;
    edges.reserve(n * (n - 1) / 2);
    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            edges.emplace_back(i, j);
        }
    }

    auto pairs = detail::compute_0d_persistence(filt_cpu, edges);

    const auto num_pairs = static_cast<int64_t>(pairs.size());
    at::Tensor diagram =
        at::empty({num_pairs, 2}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
    auto out_accessor = diagram.accessor<double, 2>();
    auto filt_accessor = filt_cpu.accessor<double, 1>();

    for (int64_t pair_index = 0; pair_index < num_pairs; ++pair_index)
    {
        const auto [birth_vertex, death_edge, death_time] = pairs[static_cast<size_t>(pair_index)];
        out_accessor[pair_index][0] = filt_accessor[birth_vertex];
        out_accessor[pair_index][1] = death_time;
    }

    return diagram.to(filtration.device()).to(filtration.scalar_type());
}

// PH Grad Function (Differentiable)

class PHGradFunction : public ::torch::autograd::Function<PHGradFunction>
{
public:
    static at::Tensor forward(::torch::autograd::AutogradContext *ctx, at::Tensor filtration,
                              int64_t max_dim)
    {
        at::Tensor filt_cpu = detail::validate_filtration_cpu(filtration, max_dim, "filtration");
        const int64_t n = filt_cpu.size(0);

        std::vector<std::pair<int64_t, int64_t>> edges;
        edges.reserve(n * (n - 1) / 2);
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = i + 1; j < n; ++j)
            {
                edges.emplace_back(i, j);
            }
        }

        auto pairs = detail::compute_0d_persistence(filt_cpu, edges);

        const auto num_pairs = static_cast<int64_t>(pairs.size());
        at::Tensor diagram =
            at::empty({num_pairs, 2}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
        auto filt_accessor = filt_cpu.accessor<double, 1>();
        auto out_accessor = diagram.accessor<double, 2>();

        std::vector<int64_t> birth_indices;
        birth_indices.reserve(static_cast<size_t>(num_pairs));

        for (int64_t pair_index = 0; pair_index < num_pairs; ++pair_index)
        {
            const auto [birth_vertex, death_edge, death_time] =
                pairs[static_cast<size_t>(pair_index)];
            birth_indices.push_back(birth_vertex);
            out_accessor[pair_index][0] = filt_accessor[birth_vertex];
            out_accessor[pair_index][1] = death_time;
        }

        ctx->save_for_backward({filtration});
        ctx->saved_data["birth_indices"] =
            at::tensor(birth_indices, at::TensorOptions().dtype(at::kLong));

        return diagram.to(filtration.device()).to(filtration.scalar_type());
    }

    static ::torch::autograd::tensor_list backward(::torch::autograd::AutogradContext *ctx,
                                                   ::torch::autograd::tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto filtration = saved[0];

        auto birth_indices = ctx->saved_data["birth_indices"].toTensor();
        int64_t num_pairs = grad_output.size(0);

        at::Tensor grad_output_cpu = grad_output.contiguous().cpu().to(at::kDouble);
        at::Tensor grad_filtration =
            at::zeros({filtration.size(0)}, at::TensorOptions().dtype(at::kDouble));
        auto grad_accessor = grad_output_cpu.accessor<double, 2>();
        auto grad_filt_accessor = grad_filtration.accessor<double, 1>();
        auto birth_accessor = birth_indices.accessor<int64_t, 1>();

        for (int64_t i = 0; i < num_pairs; ++i)
        {
            int64_t birth_idx = birth_accessor[i];
            grad_filt_accessor[birth_idx] += grad_accessor[i][0];
        }

        return {grad_filtration.to(filtration.device()).to(filtration.scalar_type()), at::Tensor()};
    }
};

at::Tensor ph_grad(const at::Tensor &filtration, int64_t max_dim)
{
    return PHGradFunction::apply(filtration, max_dim);
}

// PH Smoothed Function (Differentiable with Gaussian smoothing)

class PHSmoothedFunction : public ::torch::autograd::Function<PHSmoothedFunction>
{
public:
    static at::Tensor forward(::torch::autograd::AutogradContext *ctx, at::Tensor filtration,
                              double sigma)
    {
        ctx->save_for_backward({filtration});
        ctx->saved_data["sigma"] = sigma;

        at::Tensor filt = filtration.contiguous().cpu().to(at::kDouble);
        const int64_t n = filt.size(0);

        at::Tensor smoothed = filt.clone();

        if (sigma > 0 && n > 1)
        {
            at::Tensor signal = filt.unsqueeze(0).unsqueeze(0);

            int64_t kernel_size = std::min(int64_t(2 * int64_t(3 * sigma) + 1), n);
            if (kernel_size % 2 == 0)
                kernel_size++;

            at::Tensor kernel = at::exp(-at::pow(at::arange(-(kernel_size / 2), kernel_size / 2 + 1,
                                                            at::TensorOptions().dtype(at::kDouble)),
                                                 2) /
                                        (2 * sigma * sigma));
            kernel = kernel / at::sum(kernel);
            kernel = kernel.unsqueeze(0).unsqueeze(0);

            int64_t padding = kernel_size / 2;
            smoothed = at::conv1d(signal, kernel, {}, 1, padding).squeeze();
        }

        auto diagram = PHGradFunction::apply(smoothed, int64_t{0});

        return diagram;
    }

    static ::torch::autograd::tensor_list backward(::torch::autograd::AutogradContext *ctx,
                                                   ::torch::autograd::tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto filtration = saved[0];

        double sigma = ctx->saved_data["sigma"].toDouble();

        const int64_t n = filtration.size(0);
        at::Tensor grad_filtration = at::zeros_like(filtration);

        if (sigma > 0 && n > 1)
        {
            int64_t kernel_size = std::min(int64_t(2 * int64_t(3 * sigma) + 1), n);
            if (kernel_size % 2 == 0)
                kernel_size++;

            at::Tensor kernel = at::exp(-at::pow(at::arange(-(kernel_size / 2), kernel_size / 2 + 1,
                                                            at::TensorOptions().dtype(at::kDouble)),
                                                 2) /
                                        (2 * sigma * sigma));
            kernel = kernel / at::sum(kernel);

            auto grad_diag = grad_output.select(1, 0);

            for (int64_t i = 0; i < n; ++i)
            {
                double grad_sum = 0;
                for (int64_t k = 0; k < kernel_size; ++k)
                {
                    int64_t j = i + k - kernel_size / 2;
                    if (j >= 0 && j < n)
                    {
                        grad_sum += grad_diag[j].item<double>() * kernel[k].item<double>();
                    }
                }
                grad_filtration[i] = grad_sum;
            }
        }
        else
        {
            grad_filtration = grad_output.select(1, 0);
        }

        return {grad_filtration, at::Tensor()};
    }
};

at::Tensor ph_smoothed(const at::Tensor &filtration, double sigma)
{
    return PHSmoothedFunction::apply(filtration, sigma);
}

// Persistence Loss Functions

/// Encourages certain topological features
[[nodiscard]] at::Tensor persistence_loss(const at::Tensor &diagram, double target_birth,
                                          double target_death)
{
    if (diagram.size(0) == 0)
    {
        return at::tensor(0.0, diagram.options());
    }

    at::Tensor births = diagram.select(1, 0);
    at::Tensor deaths = diagram.select(1, 1);

    // MSE loss to target persistence
    at::Tensor birth_loss = at::mean(at::pow(births - target_birth, 2));
    at::Tensor death_loss = at::mean(at::pow(deaths - target_death, 2));

    return birth_loss + death_loss;
}

/// Penalizes total persistence (sum of lifetimes)
[[nodiscard]] at::Tensor total_persistence(const at::Tensor &diagram)
{
    if (diagram.size(0) == 0)
    {
        return at::tensor(0.0, diagram.options());
    }

    at::Tensor lifetimes = diagram.select(1, 1) - diagram.select(1, 0);
    return at::sum(lifetimes);
}

/// Penalizes number of topological features
[[nodiscard]] at::Tensor feature_count_loss(const at::Tensor &diagram, int64_t target_count)
{
    int64_t count = diagram.size(0);
    at::Tensor diff = at::tensor(static_cast<double>(count - target_count), diagram.options());
    return at::pow(diff, 2);
}

} // namespace nerve::torch
