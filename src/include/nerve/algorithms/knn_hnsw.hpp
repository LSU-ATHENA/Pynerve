#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef NERVE_USE_OPENMP
#include <omp.h>
#endif

namespace nerve::algorithms
{

template <typename T>
class HNSWIndex
{
public:
    struct Config
    {
        int M = 16;
        int ef_construction = 200;
        int ef_search = 100;
        size_t random_seed = 42;
        bool squared_distance = false;
    };

    explicit HNSWIndex(int dim, Config config = {})
        : dim_(dim)
        , config_(config)
        , rng_(config.random_seed)
    {
        if (dim <= 0)
            throw std::invalid_argument("dimension must be positive");
        if (config.M < 4)
            throw std::invalid_argument("M must be >= 4");
        if (config.ef_construction < config.M)
            throw std::invalid_argument("ef_construction must be >= M");
        if (config.ef_search < 1)
            throw std::invalid_argument("ef_search must be >= 1");
        entry_point_ = -1;
        max_layer_ = 0;
        ml_ = 1.0 / std::log(static_cast<double>(config.M));
    }

    void build(std::span<const T> points, size_t n_points)
    {
        if (n_points == 0)
            return;
        if (points.size() < n_points * static_cast<size_t>(dim_))
            throw std::invalid_argument("points span too small for n_points * dim");

        points_data_ = std::vector<T>(points.begin(), points.begin() + n_points * dim_);
        n_points_ = n_points;
        nodes_.clear();
        entry_point_ = -1;
        max_layer_ = 0;

        for (size_t i = 0; i < n_points; ++i)
        {
            insert(i);
        }
    }

    std::vector<std::pair<size_t, T>> search(std::span<const T> query, size_t k = 10) const
    {
        if (query.size() < static_cast<size_t>(dim_))
            throw std::invalid_argument("query span too small");
        if (entry_point_ < 0 || n_points_ == 0)
            return {};

        int ep = entry_point_;
        T dist = distance(query.data(), point_at(static_cast<size_t>(ep)));

        for (int lc = max_layer_; lc > 0; --lc)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                const auto &node = nodes_[static_cast<size_t>(ep)];
                size_t layer_idx = std::min(static_cast<size_t>(lc), node.neighbors.size() - 1);
                for (size_t nb : node.neighbors[layer_idx])
                {
                    T d = distance(query.data(), point_at(nb));
                    if (d < dist)
                    {
                        dist = d;
                        ep = static_cast<int>(nb);
                        changed = true;
                    }
                }
            }
        }

        return search_layer(query.data(), ep, 0, std::max(static_cast<int>(k), config_.ef_search));
    }

    std::vector<std::pair<size_t, T>> searchRadius(std::span<const T> query, T radius,
                                                   int ef_search = 0) const
    {
        if (query.size() < static_cast<size_t>(dim_))
            throw std::invalid_argument("query span too small");
        if (entry_point_ < 0 || n_points_ == 0)
            return {};

        int ef = (ef_search > 0) ? ef_search : config_.ef_search;

        int ep = entry_point_;
        T dist = distance(query.data(), point_at(static_cast<size_t>(ep)));

        for (int lc = max_layer_; lc > 0; --lc)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                const auto &node = nodes_[static_cast<size_t>(ep)];
                size_t layer_idx = std::min(static_cast<size_t>(lc), node.neighbors.size() - 1);
                for (size_t nb : node.neighbors[layer_idx])
                {
                    T d = distance(query.data(), point_at(nb));
                    if (d < dist)
                    {
                        dist = d;
                        ep = static_cast<int>(nb);
                        changed = true;
                    }
                }
            }
        }

        std::vector<bool> visited(n_points_, false);
        using Candidate = std::pair<size_t, T>;
        auto cmp_greater = [](const Candidate &a, const Candidate &b) {
            return a.second < b.second;
        };
        auto cmp_lesser = [](const Candidate &a, const Candidate &b) {
            return a.second > b.second;
        };

        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp_lesser)> candidates(
            cmp_lesser);
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp_greater)> results(
            cmp_greater);

        T d = distance(query.data(), point_at(static_cast<size_t>(ep)));
        if (d <= radius)
        {
            results.emplace(static_cast<size_t>(ep), d);
        }
        candidates.emplace(static_cast<size_t>(ep), d);
        visited[static_cast<size_t>(ep)] = true;

        while (!candidates.empty())
        {
            auto [idx, cand_dist] = candidates.top();
            candidates.pop();

            if (cand_dist > radius)
                break;

            const auto &node = nodes_[idx];
            size_t layer_idx = std::min(static_cast<size_t>(0), node.neighbors.size() - 1);

            for (size_t nb : node.neighbors[layer_idx])
            {
                if (visited[nb])
                    continue;
                visited[nb] = true;

                T nb_dist = distance(query.data(), point_at(nb));

                if (nb_dist <= radius)
                {
                    results.emplace(nb, nb_dist);
                }

                if (results.size() < static_cast<size_t>(ef))
                {
                    candidates.emplace(nb, nb_dist);
                }
            }
        }

        std::vector<std::pair<size_t, T>> sorted;
        sorted.reserve(results.size());
        while (!results.empty())
        {
            sorted.emplace_back(results.top());
            results.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    std::vector<std::vector<std::pair<size_t, T>>>
    batchSearch(const std::vector<std::span<const T>> &queries, size_t k = 10) const
    {
        std::vector<std::vector<std::pair<size_t, T>>> results(queries.size());

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < queries.size(); ++i)
        {
            results[i] = search(queries[i], k);
        }

        return results;
    }

    std::vector<std::vector<std::pair<size_t, T>>>
    batchSearch(std::span<const T> queries_flat, size_t n_queries, size_t k = 10) const
    {
        std::vector<std::vector<std::pair<size_t, T>>> results(n_queries);

#ifdef NERVE_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < n_queries; ++i)
        {
            std::span<const T> query(queries_flat.data() + i * dim_, dim_);
            results[i] = search(query, k);
        }

        return results;
    }

    size_t node_count() const { return n_points_; }
    int dimension() const { return dim_; }
    const Config &config() const { return config_; }

private:
    struct Node
    {
        std::vector<std::vector<size_t>> neighbors;
        int max_layer;
    };

    int dim_;
    Config config_;
    double ml_;
    mutable std::mt19937 rng_;
    int max_layer_;
    int entry_point_;
    size_t n_points_ = 0;
    std::vector<T> points_data_;
    std::vector<Node> nodes_;

    int random_layer()
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng_);
        if (r < std::numeric_limits<double>::min())
            r = std::numeric_limits<double>::min();
        double level = -std::log(r) * ml_;
        return std::max(0, std::min(static_cast<int>(level), max_layer_ + 1));
    }

    T distance(const T *a, const T *b) const
    {
        T sum = 0;
        for (int d = 0; d < dim_; ++d)
        {
            T diff = a[d] - b[d];
            sum += diff * diff;
        }
        if (config_.squared_distance)
            return sum;
        return std::sqrt(sum);
    }

    const T *point_at(size_t idx) const { return &points_data_[idx * static_cast<size_t>(dim_)]; }

    std::vector<std::pair<size_t, T>> search_layer(const T *query, int ep, int layer, int ef) const
    {
        std::vector<bool> visited(n_points_, false);

        using Candidate = std::pair<size_t, T>;
        auto cmp_greater = [](const Candidate &a, const Candidate &b) {
            return a.second < b.second;
        };
        auto cmp_lesser = [](const Candidate &a, const Candidate &b) {
            return a.second > b.second;
        };

        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp_lesser)> candidates(
            cmp_lesser);
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp_greater)> results(
            cmp_greater);

        T d = distance(query, point_at(static_cast<size_t>(ep)));
        candidates.emplace(static_cast<size_t>(ep), d);
        results.emplace(static_cast<size_t>(ep), d);
        visited[static_cast<size_t>(ep)] = true;

        while (!candidates.empty())
        {
            auto [idx, cand_dist] = candidates.top();
            candidates.pop();

            if (!results.empty() && results.size() >= static_cast<size_t>(ef) &&
                cand_dist > results.top().second)
            {
                break;
            }

            const auto &node = nodes_[idx];
            size_t layer_idx = std::min(static_cast<size_t>(layer), node.neighbors.size() - 1);

            for (size_t nb : node.neighbors[layer_idx])
            {
                if (visited[nb])
                    continue;
                visited[nb] = true;

                T nb_dist = distance(query, point_at(nb));
                T furthest = results.empty() ? std::numeric_limits<T>::max() : results.top().second;

                if (nb_dist < furthest || results.size() < static_cast<size_t>(ef))
                {
                    candidates.emplace(nb, nb_dist);
                    results.emplace(nb, nb_dist);

                    if (results.size() > static_cast<size_t>(ef))
                    {
                        results.pop();
                    }
                }
            }
        }

        std::vector<std::pair<size_t, T>> sorted;
        sorted.reserve(results.size());
        while (!results.empty())
        {
            sorted.emplace_back(results.top());
            results.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    std::vector<std::pair<size_t, T>>
    select_neighbors(const std::vector<std::pair<size_t, T>> &candidates, size_t M) const
    {
        if (candidates.size() <= M)
            return candidates;

        std::vector<std::pair<size_t, T>> selected;
        selected.reserve(M);
        std::vector<bool> taken(candidates.size(), false);

        for (size_t i = 0; i < candidates.size() && selected.size() < M; ++i)
        {
            bool found = false;
            for (size_t j = 0; j < i && !found; ++j)
            {
                if (!taken[j] && candidates[j].first == candidates[i].first)
                    found = true;
            }
            if (found)
                continue;
            taken[i] = true;
            selected.push_back(candidates[i]);
        }

        return selected;
    }

    void shrink_neighbors(size_t node_idx, int layer, size_t M_max)
    {
        auto &neighbors = nodes_[node_idx].neighbors[static_cast<size_t>(layer)];
        if (neighbors.size() <= M_max)
            return;

        std::vector<std::pair<size_t, T>> scored;
        scored.reserve(neighbors.size());
        const T *src = point_at(node_idx);
        for (size_t nb : neighbors)
        {
            scored.emplace_back(nb, distance(src, point_at(nb)));
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto &a, const auto &b) { return a.second < b.second; });

        neighbors.clear();
        for (size_t i = 0; i < M_max; ++i)
        {
            neighbors.push_back(scored[i].first);
        }
    }

    void insert(size_t idx)
    {
        int layer = random_layer();
        nodes_.emplace_back();
        auto &node = nodes_.back();
        node.max_layer = layer;
        node.neighbors.resize(static_cast<size_t>(layer) + 1);

        if (entry_point_ < 0)
        {
            entry_point_ = static_cast<int>(idx);
            max_layer_ = layer;
            return;
        }

        if (layer > max_layer_)
        {
            max_layer_ = layer;
        }

        int ep = entry_point_;
        const T *point = point_at(idx);

        for (int lc = max_layer_; lc > layer; --lc)
        {
            T d = distance(point, point_at(static_cast<size_t>(ep)));
            const auto &ep_node = nodes_[static_cast<size_t>(ep)];
            size_t layer_idx = std::min(static_cast<size_t>(lc), ep_node.neighbors.size() - 1);

            for (size_t nb : ep_node.neighbors[layer_idx])
            {
                T nd = distance(point, point_at(nb));
                if (nd < d)
                {
                    d = nd;
                    ep = static_cast<int>(nb);
                }
            }
        }

        for (int lc = std::min(layer, max_layer_); lc >= 0; --lc)
        {
            size_t M =
                (lc == 0) ? static_cast<size_t>(config_.M) * 2 : static_cast<size_t>(config_.M);
            auto nearest = search_layer(point, ep, lc, config_.ef_construction);

            auto selected = select_neighbors(nearest, M);

            for (const auto &[nb_idx, _] : selected)
            {
                node.neighbors[static_cast<size_t>(lc)].push_back(nb_idx);
                size_t nb_actual =
                    std::min(static_cast<size_t>(lc), nodes_[nb_idx].neighbors.size() - 1);
                nodes_[nb_idx].neighbors[nb_actual].push_back(idx);
                shrink_neighbors(nb_idx, static_cast<int>(nb_actual), M);
            }

            shrink_neighbors(idx, lc, M);

            if (!nearest.empty())
            {
                ep = static_cast<int>(nearest[0].first);
            }
        }

        if (layer > nodes_[static_cast<size_t>(entry_point_)].max_layer)
        {
            entry_point_ = static_cast<int>(idx);
        }
    }
};

} // namespace nerve::algorithms
