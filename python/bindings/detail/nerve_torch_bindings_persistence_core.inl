namespace
{

/// Uses existing Nerve persistence algorithms
struct ExtendedPersistenceResult
{
    std::vector<std::tuple<double, double, int64_t>> pairs; // (birth, death, dim)
    std::vector<int64_t> birth_indices;
    std::vector<int64_t> death_indices;
};

struct UnionFind
{
    std::vector<int64_t> parent;
    std::vector<int64_t> rank;
    std::vector<int64_t> representative;

    explicit UnionFind(int64_t n)
        : parent(n)
        , rank(n, 0)
        , representative(n)
    {
        std::iota(parent.begin(), parent.end(), 0);
        std::iota(representative.begin(), representative.end(), 0);
    }

    int64_t find(int64_t x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    int64_t unite(int64_t x, int64_t y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
        {
            return x;
        }
        if (rank[x] < rank[y])
        {
            std::swap(x, y);
        }
        parent[y] = x;
        representative[x] = std::min(representative[x], representative[y]);
        if (rank[x] == rank[y])
        {
            rank[x]++;
        }
        return x;
    }
};

struct CycleCandidate
{
    double birth = 0.0;
    double death = std::numeric_limits<double>::infinity();
    int64_t u = -1;
    int64_t v = -1;
};

struct EdgeRecord
{
    int64_t u = -1;
    int64_t v = -1;
    double w = 0.0;
};

#include <mutex>
#include <unordered_map>

static std::mutex g_persistence_info_mutex;
static std::unordered_map<const void *, std::vector<ExtendedPersistenceResult>>
    g_persistence_info_map;

void save_persistence_info(const void *key, const std::vector<ExtendedPersistenceResult> &info)
{
    std::lock_guard<std::mutex> lock(g_persistence_info_mutex);
    g_persistence_info_map[key] = info;
}

std::vector<ExtendedPersistenceResult> load_persistence_info(const void *key)
{
    std::lock_guard<std::mutex> lock(g_persistence_info_mutex);
    auto it = g_persistence_info_map.find(key);
    TORCH_CHECK(it != g_persistence_info_map.end(),
                "persistence backward requires forward metadata");
    std::vector<ExtendedPersistenceResult> result = std::move(it->second);
    g_persistence_info_map.erase(it);
    return result;
}

ExtendedPersistenceResult load_persistence_info_single(const void *key)
{
    auto info = load_persistence_info(key);
    TORCH_CHECK(!info.empty(), "persistence backward requires forward metadata");
    return std::move(info[0]);
}

uint64_t edge_key(int64_t u, int64_t v)
{
    const uint64_t a = static_cast<uint64_t>(std::min(u, v));
    const uint64_t b = static_cast<uint64_t>(std::max(u, v));
    return (a << 32) | b;
}

/// This extends the basic union-find H0 computation to higher dimensions
ExtendedPersistenceResult compute_extended_persistence(const at::Tensor &dist_matrix,
                                                       int64_t max_dim, double max_radius)
{
    using namespace nerve::torch;

    ExtendedPersistenceResult result;
    const int64_t n = dist_matrix.size(0);

    if (n == 0)
    {
        return result;
    }

    const at::Tensor dist_cpu = dist_matrix.contiguous().cpu().to(at::kDouble);
    const auto accessor = dist_cpu.accessor<double, 2>();

    std::vector<EdgeRecord> edges;
    edges.reserve(static_cast<size_t>(n * (n - 1) / 2));
    std::unordered_map<uint64_t, double> edge_weight;
    edge_weight.reserve(static_cast<size_t>(n * (n - 1) / 2));

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = i + 1; j < n; ++j)
        {
            const double d = accessor[i][j];
            if (!std::isfinite(d))
            {
                continue;
            }
            if (std::isfinite(max_radius) && d > max_radius)
            {
                continue;
            }
            edges.push_back(EdgeRecord{i, j, d});
            edge_weight.emplace(edge_key(i, j), d);
        }
    }

    std::sort(edges.begin(), edges.end(),
              [](const EdgeRecord &lhs, const EdgeRecord &rhs) { return lhs.w < rhs.w; });

    UnionFind uf(n);
    std::vector<CycleCandidate> cycle_candidates;
    cycle_candidates.reserve(edges.size());

    for (const auto &edge : edges)
    {
        const int64_t ru = uf.find(edge.u);
        const int64_t rv = uf.find(edge.v);
        if (ru != rv)
        {
            const int64_t dead_rep = std::max(uf.representative[ru], uf.representative[rv]);
            result.pairs.emplace_back(0.0, edge.w, 0);
            result.birth_indices.push_back(dead_rep);
            result.death_indices.push_back(edge.v);
            uf.unite(ru, rv);
        }
        else if (max_dim >= 1)
        {
            cycle_candidates.push_back(
                CycleCandidate{.birth = edge.w,
                               .death = std::numeric_limits<double>::infinity(),
                               .u = edge.u,
                               .v = edge.v});
        }
    }

    for (int64_t i = 0; i < n; ++i)
    {
        if (uf.find(i) == i)
        {
            result.pairs.emplace_back(0.0, std::numeric_limits<double>::infinity(), 0);
            result.birth_indices.push_back(uf.representative[i]);
            result.death_indices.push_back(-1);
        }
    }

    if (max_dim >= 1)
    {
        std::sort(cycle_candidates.begin(), cycle_candidates.end(),
                  [](const CycleCandidate &lhs, const CycleCandidate &rhs) {
                      return lhs.birth < rhs.birth;
                  });

        if (max_dim >= 2 && n <= 512)
        {
            std::vector<double> triangle_deaths;
            triangle_deaths.reserve(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i)
            {
                for (int64_t j = i + 1; j < n; ++j)
                {
                    const auto ij_it = edge_weight.find(edge_key(i, j));
                    if (ij_it == edge_weight.end())
                    {
                        continue;
                    }
                    for (int64_t k = j + 1; k < n; ++k)
                    {
                        const auto ik_it = edge_weight.find(edge_key(i, k));
                        const auto jk_it = edge_weight.find(edge_key(j, k));
                        if (ik_it == edge_weight.end() || jk_it == edge_weight.end())
                        {
                            continue;
                        }
                        triangle_deaths.push_back(
                            std::max({ij_it->second, ik_it->second, jk_it->second}));
                    }
                }
            }
            std::sort(triangle_deaths.begin(), triangle_deaths.end());
            size_t cycle_idx = 0;
            for (const double death : triangle_deaths)
            {
                if (cycle_idx >= cycle_candidates.size())
                {
                    break;
                }
                if (cycle_candidates[cycle_idx].birth <= death)
                {
                    cycle_candidates[cycle_idx].death = death;
                    cycle_idx++;
                }
            }
        }

        for (const auto &cycle : cycle_candidates)
        {
            result.pairs.emplace_back(cycle.birth, cycle.death, 1);
            result.birth_indices.push_back(cycle.u);
            result.death_indices.push_back(cycle.v);
        }
    }

    if (max_dim >= 2 && n <= 96)
    {
        constexpr size_t MAX_H2_PAIRS = 512;
        size_t emitted = 0;
        for (int64_t i = 0; i < n && emitted < MAX_H2_PAIRS; ++i)
        {
            for (int64_t j = i + 1; j < n && emitted < MAX_H2_PAIRS; ++j)
            {
                const auto ij_it = edge_weight.find(edge_key(i, j));
                if (ij_it == edge_weight.end())
                {
                    continue;
                }
                for (int64_t k = j + 1; k < n && emitted < MAX_H2_PAIRS; ++k)
                {
                    const auto ik_it = edge_weight.find(edge_key(i, k));
                    const auto jk_it = edge_weight.find(edge_key(j, k));
                    if (ik_it == edge_weight.end() || jk_it == edge_weight.end())
                    {
                        continue;
                    }
                    for (int64_t l = k + 1; l < n && emitted < MAX_H2_PAIRS; ++l)
                    {
                        const auto il_it = edge_weight.find(edge_key(i, l));
                        const auto jl_it = edge_weight.find(edge_key(j, l));
                        const auto kl_it = edge_weight.find(edge_key(k, l));
                        if (il_it == edge_weight.end() || jl_it == edge_weight.end() ||
                            kl_it == edge_weight.end())
                        {
                            continue;
                        }
                        const double birth =
                            std::max({ij_it->second, ik_it->second, il_it->second, jk_it->second,
                                      jl_it->second, kl_it->second});
                        result.pairs.emplace_back(birth, std::numeric_limits<double>::infinity(),
                                                  2);
                        result.birth_indices.push_back(i);
                        result.death_indices.push_back(-1);
                        emitted++;
                    }
                }
            }
        }
    }

    return result;
}

at::Tensor
compute_vr_gradients(const at::Tensor &grad_diagrams, // [n_pairs, 3] (birth_grad, death_grad, dim)
                     const at::Tensor &points,        // [n_points, dim]
                     const at::Tensor &dist_matrix,   // [n_points, n_points]
                     const ExtendedPersistenceResult &persistence)
{
    (void)dist_matrix;
    at::Tensor grad_points = at::zeros_like(points);
    const int64_t n = points.size(0);
    const int64_t point_dim = points.size(1);

    // Gradient for each persistence pair
    const int64_t grad_rows = grad_diagrams.size(0);
    const auto pair_count = std::min(static_cast<int64_t>(persistence.pairs.size()), grad_rows);
    for (int64_t pair_index = 0; pair_index < pair_count; ++pair_index)
    {
        const auto vector_index = static_cast<size_t>(pair_index);
        const auto [birth, death, dim] = persistence.pairs[vector_index];
        int64_t birth_idx = persistence.birth_indices[vector_index];
        int64_t death_idx = persistence.death_indices[vector_index];

        // Get gradient for this pair
        double grad_birth = grad_diagrams[pair_index][0].item<double>();
        double grad_death = grad_diagrams[pair_index][1].item<double>();

        if (birth_idx >= 0 && birth_idx < n)
        {
            for (int64_t d = 0; d < point_dim; ++d)
            {
                grad_points[birth_idx][d] += grad_birth;
            }
        }

        if (death_idx >= 0 && death_idx < n)
        {
            for (int64_t d = 0; d < point_dim; ++d)
            {
                grad_points[death_idx][d] += grad_death;
            }
        }

        if (dim >= 1 && birth_idx >= 0 && death_idx >= 0 && birth_idx < n && death_idx < n &&
            std::isfinite(death))
        {
            double norm = 0.0;
            for (int64_t d = 0; d < point_dim; ++d)
            {
                const double delta =
                    points[death_idx][d].item<double>() - points[birth_idx][d].item<double>();
                norm += delta * delta;
            }
            norm = std::sqrt(std::max(norm, 1e-12));
            for (int64_t d = 0; d < point_dim; ++d)
            {
                const double delta =
                    points[death_idx][d].item<double>() - points[birth_idx][d].item<double>();
                const double dir = delta / norm;
                grad_points[birth_idx][d] -= grad_birth * dir;
                grad_points[death_idx][d] += grad_death * dir;
            }
        }
    }

    return grad_points;
}

} // anonymous namespace
