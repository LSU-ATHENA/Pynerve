
#pragma once
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <vector>

namespace nerve::metrics
{

using Diagram = std::vector<nerve::Pair>;
template <typename Target, typename Source>
nerve::error::Result<Target> forwardResultError(const nerve::error::Result<Source> &source)
{
    return nerve::error::Result<Target>::err(
        static_cast<nerve::error::TDAErrorCode>(source.error().value()),
        std::string(source.detail()), source.where());
}

struct PartitionedDiagram
{
    std::vector<const nerve::Pair *> finite;
    std::vector<const nerve::Pair *> essential;

    static PartitionedDiagram from(const Diagram &dgm, int dim = -1)
    {
        PartitionedDiagram out;
        for (const auto &p : dgm)
        {
            if (dim >= 0 && p.dimension != dim)
                continue;
            if (p.isInfinite())
                out.essential.push_back(&p);
            else
                out.finite.push_back(&p);
        }
        return out;
    }
};

class HopcroftKarp
{
public:
    explicit HopcroftKarp(int n_left, int n_right, std::vector<std::vector<int>> adj)
        : nL_(n_left)
        , nR_(n_right)
        , adj_(std::move(adj))
        , matchL_(n_left, -1)
        , matchR_(n_right, -1)
        , dist_(n_left, 0)
    {}

    int maxMatching()
    {
        int matching = 0;
        while (bfs())
        {
            for (int u = 0; u < nL_; ++u)
            {
                if (matchL_[u] == -1 && dfs(u))
                {
                    ++matching;
                }
            }
        }
        return matching;
    }

    bool perfectMatching() { return maxMatching() == nL_; }

private:
    static constexpr int INF = std::numeric_limits<int>::max();
    int nL_, nR_;
    std::vector<std::vector<int>> adj_;
    std::vector<int> matchL_, matchR_;
    std::vector<int> dist_;

    bool bfs()
    {
        std::queue<int> q;

        for (int u = 0; u < nL_; ++u)
        {
            if (matchL_[u] == -1)
            {
                dist_[u] = 0;
                q.push(u);
            }
            else
            {
                dist_[u] = -1;
            }
        }

        bool found_augmenting_path = false;

        while (!q.empty())
        {
            int u = q.front();
            q.pop();

            for (int v : adj_[u])
            {
                int w = matchR_[v];
                if (w == -1)
                {
                    found_augmenting_path = true;
                }
                else if (dist_[w] == -1)
                {
                    dist_[w] = dist_[u] + 1;
                    q.push(w);
                }
            }
        }

        return found_augmenting_path;
    }

    bool dfs(int u)
    {
        for (int v : adj_[u])
        {
            int w = matchR_[v];
            if (w == -1 || (dist_[w] == dist_[u] + 1 && dfs(w)))
            {
                matchL_[u] = v;
                matchR_[v] = u;
                return true;
            }
        }
        dist_[u] = -1;
        return false;
    }
};

class BottleneckDistance
{
public:
    struct Result
    {
        double distance;
        bool is_infinite; // true if essential pair counts differ
        bool was_exact;   // true if computed exactly (vs approximate)
    };

    static nerve::error::Result<double> compute(const Diagram &A, const Diagram &B,
                                                int dimension = -1)
    {
        auto va = validateDiagram(A);
        if (!va.isOk())
            return forwardResultError<double>(va);
        auto vb = validateDiagram(B);
        if (!vb.isOk())
            return forwardResultError<double>(vb);

        if (A.empty() && B.empty())
            return nerve::error::Result<double>::ok(0.0);

        auto pA = PartitionedDiagram::from(A, dimension);
        auto pB = PartitionedDiagram::from(B, dimension);

        if (pA.essential.size() != pB.essential.size())
            return nerve::error::Result<double>::ok(std::numeric_limits<double>::infinity());

        return computeFiniteBottleneck(pA.finite, pB.finite);
    }

    static nerve::error::Result<void> validateDiagram(const Diagram &dgm)
    {
        for (nerve::Size i = 0; i < dgm.size(); ++i)
        {
            const auto &p = dgm[i];
            if (!std::isfinite(p.birth))
                return nerve::error::Result<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                                       "birth must be finite at index " +
                                                           std::to_string(i));
            if (!std::isfinite(p.death) && !p.isInfinite())
                return nerve::error::Result<void>::err(
                    nerve::error::TDAErrorCode::InvalidInput,
                    "death must be finite or +infinity at index " + std::to_string(i));
            if (!p.isInfinite() && p.death < p.birth)
                return nerve::error::Result<void>::err(
                    nerve::error::TDAErrorCode::InvalidFiltrationValue,
                    "death < birth at index " + std::to_string(i));
        }
        return nerve::error::Result<void>::ok();
    }

private:
    static nerve::error::Result<double>
    computeFiniteBottleneck(const std::vector<const nerve::Pair *> &A,
                            const std::vector<const nerve::Pair *> &B)
    {
        if (A.empty() && B.empty())
        {
            return nerve::error::Result<double>::ok(0.0);
        }
        if (A.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            B.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            A.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - B.size())
        {
            return nerve::error::Result<double>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "bottleneck matching problem exceeds int limits");
        }
        const int nA = static_cast<int>(A.size());
        const int nB = static_cast<int>(B.size());

        std::vector<double> candidates;
        auto candidate_count = checkedCandidateCount(A.size(), B.size());
        if (candidate_count.isErr())
            return forwardResultError<double>(candidate_count);
        candidates.reserve(candidate_count.value());

        auto push_candidate = [&candidates](double value) -> nerve::error::Result<void> {
            if (!std::isfinite(value))
            {
                return nerve::error::Result<void>::err(
                    nerve::error::TDAErrorCode::InvalidFieldOperation,
                    "bottleneck candidate must be finite");
            }
            candidates.push_back(value);
            return nerve::error::Result<void>::ok();
        };

        for (int i = 0; i < nA; ++i)
        {
            auto diagonal = push_candidate(A[i]->lifetime() / 2.0);
            if (diagonal.isErr())
                return forwardResultError<double>(diagonal);
            for (int j = 0; j < nB; ++j)
            {
                double db = std::abs(A[i]->birth - B[j]->birth);
                double dd = std::abs(A[i]->death - B[j]->death);
                auto candidate = push_candidate(std::max(db, dd));
                if (candidate.isErr())
                    return forwardResultError<double>(candidate);
            }
        }
        for (int j = 0; j < nB; ++j)
        {
            auto diagonal = push_candidate(B[j]->lifetime() / 2.0);
            if (diagonal.isErr())
                return forwardResultError<double>(diagonal);
        }

        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        int lo = 0, hi = static_cast<int>(candidates.size()) - 1;
        double answer = candidates.back();

        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            double delta = candidates[mid];

            if (hasPerfectMatching(A, B, delta))
            {
                answer = delta;
                hi = mid - 1;
            }
            else
            {
                lo = mid + 1;
            }
        }

        return nerve::error::Result<double>::ok(answer);
    }

    static nerve::error::Result<std::size_t> checkedCandidateCount(std::size_t nA, std::size_t nB)
    {
        if (nB != 0 && nA > std::numeric_limits<std::size_t>::max() / nB)
        {
            return nerve::error::Result<std::size_t>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "bottleneck candidate count overflows size_t");
        }
        const std::size_t pair_count = nA * nB;
        if (pair_count > std::numeric_limits<std::size_t>::max() - nA)
        {
            return nerve::error::Result<std::size_t>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "bottleneck candidate count overflows size_t");
        }
        std::size_t count = pair_count + nA;
        if (count > std::numeric_limits<std::size_t>::max() - nB)
        {
            return nerve::error::Result<std::size_t>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "bottleneck candidate count overflows size_t");
        }
        count += nB;
        if (count > std::vector<double>().max_size())
        {
            return nerve::error::Result<std::size_t>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "bottleneck candidate count exceeds vector capacity");
        }
        return nerve::error::Result<std::size_t>::ok(std::move(count));
    }

    static bool hasPerfectMatching(const std::vector<const nerve::Pair *> &A,
                                   const std::vector<const nerve::Pair *> &B, double delta)
    {
        const int nA = static_cast<int>(A.size());
        const int nB = static_cast<int>(B.size());
        const int N = nA + nB;

        std::vector<std::vector<int>> adj(N);

        for (int i = 0; i < nA; ++i)
        {
            for (int j = 0; j < nB; ++j)
            {
                double db = std::abs(A[i]->birth - B[j]->birth);
                double dd = std::abs(A[i]->death - B[j]->death);
                if (std::max(db, dd) <= delta)
                {
                    adj[i].push_back(j);
                }
            }
            if (A[i]->lifetime() / 2.0 <= delta)
            {
                adj[i].push_back(nB + i); // A[i]'s diagonal slot
            }
        }

        for (int j = 0; j < nB; ++j)
        {
            if (B[j]->lifetime() / 2.0 <= delta)
            {
                adj[nA + j].push_back(j); // B[j]'s diagonal slot
            }
        }

        HopcroftKarp hk(N, N, std::move(adj));
        return hk.perfectMatching();
    }
};

class WassersteinDistance
{
public:
    static nerve::error::Result<double> compute(const Diagram &A, const Diagram &B, double p = 2.0,
                                                int dimension = -1)
    {
        auto va = BottleneckDistance::validateDiagram(A);
        if (!va.isOk())
            return forwardResultError<double>(va);
        auto vb = BottleneckDistance::validateDiagram(B);
        if (!vb.isOk())
            return forwardResultError<double>(vb);

        if (A.empty() && B.empty())
            return nerve::error::Result<double>::ok(0.0);

        auto pA = PartitionedDiagram::from(A, dimension);
        auto pB = PartitionedDiagram::from(B, dimension);

        if (pA.essential.size() != pB.essential.size())
            return nerve::error::Result<double>::ok(std::numeric_limits<double>::infinity());

        return computeWassersteinHungarian(pA.finite, pB.finite, p);
    }

private:
    static nerve::error::Result<double>
    computeWassersteinHungarian(const std::vector<const nerve::Pair *> &A,
                                const std::vector<const nerve::Pair *> &B, double p)
    {
        if (!(p > 0.0) || !std::isfinite(p))
        {
            return nerve::error::Result<double>::err(
                nerve::error::TDAErrorCode::InvalidInput,
                "Wasserstein order p must be finite and positive");
        }
        const std::size_t nA = A.size();
        const std::size_t nB = B.size();
        if (nA > std::numeric_limits<std::size_t>::max() - nB)
        {
            return nerve::error::Result<double>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "Wasserstein matching problem overflows size_t");
        }
        const std::size_t N = nA + nB; // augmented size
        if (N == 0)
        {
            return nerve::error::Result<double>::ok(0.0);
        }
        if (N > std::vector<std::vector<double>>().max_size() ||
            N > std::vector<double>().max_size())
        {
            return nerve::error::Result<double>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "Wasserstein cost matrix exceeds vector capacity");
        }

        const double inf = std::numeric_limits<double>::infinity();
        std::vector<std::vector<double>> cost(N, std::vector<double>(N, inf));

        for (std::size_t i = 0; i < nA; ++i)
        {
            for (std::size_t j = 0; j < nB; ++j)
            {
                double db = std::abs(A[i]->birth - B[j]->birth);
                double dd = std::abs(A[i]->death - B[j]->death);
                double dist = std::max(db, dd); // L-inf distance
                const double value = std::pow(dist, p);
                if (!std::isfinite(value))
                {
                    return nerve::error::Result<double>::err(
                        nerve::error::TDAErrorCode::InvalidFieldOperation,
                        "Wasserstein cost must be finite");
                }
                cost[i][j] = value;
            }
        }

        for (std::size_t i = 0; i < nA; ++i)
        {
            const double value = std::pow(A[i]->lifetime() / 2.0, p);
            if (!std::isfinite(value))
            {
                return nerve::error::Result<double>::err(
                    nerve::error::TDAErrorCode::InvalidFieldOperation,
                    "Wasserstein diagonal cost must be finite");
            }
            cost[i][nB + i] = value;
        }

        for (std::size_t j = 0; j < nB; ++j)
        {
            const double value = std::pow(B[j]->lifetime() / 2.0, p);
            if (!std::isfinite(value))
            {
                return nerve::error::Result<double>::err(
                    nerve::error::TDAErrorCode::InvalidFieldOperation,
                    "Wasserstein diagonal cost must be finite");
            }
            cost[nA + j][j] = value;
        }
        for (std::size_t j = 0; j < nB; ++j)
        {
            for (std::size_t i = 0; i < nA; ++i)
            {
                cost[nA + j][nB + i] = 0.0;
            }
        }

        auto assignment = hungarian(cost, N);
        if (!assignment.isOk())
            return forwardResultError<double>(assignment);

        double total = 0.0;
        const auto &asgn = assignment.value();
        for (std::size_t i = 0; i < N; ++i)
        {
            const double selected_cost = cost[i][asgn[i]];
            if (!std::isfinite(selected_cost))
            {
                return nerve::error::Result<double>::err(
                    nerve::error::TDAErrorCode::ConvergenceFailure,
                    "Wasserstein assignment contains infeasible edge");
            }
            total += selected_cost;
            if (!std::isfinite(total))
            {
                return nerve::error::Result<double>::err(
                    nerve::error::TDAErrorCode::InvalidFieldOperation,
                    "Wasserstein total cost must be finite");
            }
        }

        const double distance = std::pow(total, 1.0 / p);
        if (!std::isfinite(distance))
        {
            return nerve::error::Result<double>::err(
                nerve::error::TDAErrorCode::InvalidFieldOperation,
                "Wasserstein distance must be finite");
        }
        return nerve::error::Result<double>::ok(distance);
    }

    static nerve::error::Result<std::vector<std::size_t>>
    hungarian(const std::vector<std::vector<double>> &cost, std::size_t n)
    {
        std::vector<std::size_t> assignment(n);
        std::vector<double> u(n + 1), v(n + 1);
        std::vector<std::size_t> p(n + 1), way(n + 1);

        for (std::size_t i = 1; i <= n; ++i)
        {
            p[0] = i;
            std::size_t j0 = 0;
            std::vector<double> minVal(n + 1, std::numeric_limits<double>::infinity());
            std::vector<bool> used(n + 1, false);

            do
            {
                used[j0] = true;
                std::size_t i0 = p[j0];
                double delta = std::numeric_limits<double>::infinity();
                std::size_t j1 = 0;

                for (std::size_t j = 1; j <= n; ++j)
                {
                    if (!used[j])
                    {
                        double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minVal[j])
                        {
                            minVal[j] = cur;
                            way[j] = j0;
                        }
                        if (minVal[j] < delta)
                        {
                            delta = minVal[j];
                            j1 = j;
                        }
                    }
                }
                if (!std::isfinite(delta) || j1 == 0)
                {
                    return nerve::error::Result<std::vector<std::size_t>>::err(
                        nerve::error::TDAErrorCode::ConvergenceFailure,
                        "Hungarian solver failed to find augmenting path");
                }

                for (std::size_t j = 0; j <= n; ++j)
                {
                    if (used[j])
                    {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minVal[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            do
            {
                std::size_t j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        for (std::size_t j = 1; j <= n; ++j)
        {
            if (p[j] != 0)
            {
                assignment[p[j] - 1] = j - 1;
            }
        }

        return nerve::error::Result<std::vector<std::size_t>>::ok(std::move(assignment));
    }
};

} // namespace nerve::metrics
