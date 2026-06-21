#include "nerve/core.hpp"
#include "nerve/persistence/vr/detail/vr_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{
namespace
{

static thread_local std::vector<std::vector<int64_t>> t_binom;
static thread_local int t_binom_n = -1;

void ensure_binom(int n)
{
    if (n <= t_binom_n && !t_binom.empty())
        return;
    t_binom_n = n;
    t_binom.resize(n + 1);
    for (int i = 0; i <= n; ++i)
    {
        t_binom[i].resize(i + 1);
        t_binom[i][0] = t_binom[i][i] = 1;
        for (int j = 1; j < i; ++j)
            t_binom[i][j] = t_binom[i - 1][j - 1] + t_binom[i - 1][j];
    }
}

int64_t C(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    return t_binom[static_cast<size_t>(n)][static_cast<size_t>(k)];
}

int64_t binom_idx_2(int a, int b)
{
    return C(a, 1) + C(b, 2);
}
int64_t binom_idx_3(int a, int b, int c)
{
    return C(a, 1) + C(b, 2) + C(c, 3);
}

} // anonymous namespace

static std::vector<Pair> computeVrPersistenceImplicit(const core::BufferView<const double> &points,
                                                      Size point_dim, const VRConfig &config)
{
    const int n = static_cast<int>(points.size() / point_dim);
    const int p_dim = static_cast<int>(point_dim);
    const int max_d = static_cast<int>(config.max_dim);
    const double thr = config.max_radius;
    const double *pts = points.data();

    ensure_binom(n);
    std::vector<Pair> pairs;

    // Precompute edge distances
    int64_t n_edges = static_cast<int64_t>(n) * (n - 1) / 2;
    std::vector<double> edge_d(n_edges + 1, -1.0);
    auto ekey = [&](int i, int j) {
        if (i > j)
            std::swap(i, j);
        return static_cast<int64_t>(i) * (2LL * n - i - 1) / 2 + (j - i - 1);
    };
    auto ed = [&](int i, int j) -> double {
        int64_t k = ekey(i, j);
        double &c = edge_d[k];
        if (c >= 0)
            return c;
        const double *pi = pts + static_cast<std::ptrdiff_t>(i) * p_dim;
        const double *pj = pts + static_cast<std::ptrdiff_t>(j) * p_dim;
        double s = 0.0;
        for (int d = 0; d < p_dim; ++d)
        {
            double df = pi[d] - pj[d];
            s += df * df;
        }
        return c = std::sqrt(s);
    };

    // H0: union-find
    {
        std::vector<int> par(n), dead(n, false);
        std::iota(par.begin(), par.end(), 0);
        std::vector<double> birth(n, 0.0);
        std::vector<std::tuple<double, int, int>> edges;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
            {
                double d = ed(i, j);
                if (d <= thr)
                    edges.emplace_back(d, i, j);
            }
        std::sort(edges.begin(), edges.end());
        auto fnd = [&](int x) -> int {
            while (par[x] != x)
            {
                par[x] = par[par[x]];
                x = par[x];
            }
            return x;
        };
        for (const auto &[w, u, v] : edges)
        {
            int ru = fnd(u), rv = fnd(v);
            if (ru != rv)
            {
                if (ru < rv)
                    std::swap(ru, rv);
                if (birth[ru] < w)
                    pairs.push_back({birth[ru], w, 0});
                par[ru] = rv;
                dead[ru] = true;
            }
        }
        for (int i = 0; i < n; ++i)
            if (!dead[i] && par[i] == i)
                pairs.push_back({0.0, std::numeric_limits<double>::infinity(), 0});
    }
    if (max_d < 1)
        return pairs;

    // DIM 1+: cohomology (coboundary) reduction
    // For each dimension d: columns = d-simplices, rows = (d+1)-simplices (cofacets).
    // Coboundary of sigma: all (d+1)-simplices tau = sigma  U  {v} for v > max_vertex(sigma).
    // Computed LAZILY: only enumerated when needed.

    struct Simp
    {
        double diam;
        int64_t bidx;
        std::vector<int> verts;
        int pos; // sorted order position
    };

    // Track zero-reduced columns from previous dimension (births in dim d-1)
    // that will be resolved when they appear as cofacets (rows) in dim d.
    // bidx -> birth diam
    std::vector<bool> prev_zero;
    std::vector<double> prev_birth;

    for (int d = 1; d <= max_d; ++d)
    {
        int col_vn = d + 1; // column simplex vertex count
        int row_vn = d + 2; // cofacet (row) simplex vertex count

        // Enumerate columns (d-simplices)
        std::vector<Simp> cols;
        int64_t col_max_idx = C(n, col_vn);
        cols.reserve(std::min(col_max_idx, (int64_t)1 << 27));

        if (col_vn == 2)
        {
            for (int a = 0; a < n; ++a)
                for (int b = a + 1; b < n; ++b)
                {
                    double d = ed(a, b);
                    if (d <= thr)
                        cols.push_back({d, binom_idx_2(a, b), {a, b}, 0});
                }
        }
        else
        {
            for (int a = 0; a < n - 2; ++a)
                for (int b = a + 1; b < n - 1; ++b)
                {
                    double dab = ed(a, b);
                    if (dab > thr)
                        continue;
                    for (int c = b + 1; c < n; ++c)
                    {
                        double dmax = std::max({dab, ed(a, c), ed(b, c)});
                        if (dmax <= thr)
                            cols.push_back({dmax, binom_idx_3(a, b, c), {a, b, c}, 0});
                    }
                }
        }
        if (cols.empty())
        {
            prev_zero.clear();
            continue;
        }

        // Sort columns by diameter
        std::vector<int> cord(cols.size());
        std::iota(cord.begin(), cord.end(), 0);
        std::sort(cord.begin(), cord.end(), [&](int x, int y) {
            if (std::abs(cols[x].diam - cols[y].diam) > 1e-12)
                return cols[x].diam < cols[y].diam;
            return x < y;
        });
        for (size_t i = 0; i < cord.size(); ++i)
            cols[cord[i]].pos = static_cast<int>(i);

        // Build row position lookup for cofacets
        // For each cofacet (d+1)-simplex, pre-compute its position.
        // Use direct array: bidx -> position
        std::vector<Simp> rows;
        int64_t row_max_idx = C(n, row_vn);
        rows.reserve(std::min(row_max_idx, (int64_t)1 << 27));
        std::vector<int> rpos(row_max_idx + 1, -1);

        {
            std::vector<int> stk(row_vn);
            std::function<void(int, int)> erow = [&](int depth, int start) {
                if (depth == row_vn)
                {
                    double dmax = 0.0;
                    for (int i = 0; i < row_vn; ++i)
                        for (int j = i + 1; j < row_vn; ++j)
                            dmax = std::max(dmax, ed(stk[i], stk[j]));
                    if (dmax <= thr)
                    {
                        int64_t bidx = 0;
                        for (int i = 0; i < row_vn; ++i)
                            if (stk[i] >= i)
                                bidx += C(stk[i], i + 1);
                        rows.push_back({dmax, bidx, stk, 0});
                    }
                    return;
                }
                int end = n - (row_vn - depth);
                for (int i = start; i <= end; ++i)
                {
                    stk[depth] = i;
                    bool under = true;
                    for (int p = 0; p < depth && under; ++p)
                        if (ed(stk[p], i) > thr)
                            under = false;
                    if (under)
                        erow(depth + 1, i + 1);
                }
            };
            if (row_vn <= n)
                erow(0, 0);
        }

        // Sort rows by diameter
        std::vector<int> rord(rows.size());
        std::iota(rord.begin(), rord.end(), 0);
        std::sort(rord.begin(), rord.end(), [&](int x, int y) {
            if (std::abs(rows[x].diam - rows[y].diam) > 1e-12)
                return rows[x].diam < rows[y].diam;
            return x < y;
        });
        for (size_t i = 0; i < rord.size(); ++i)
            rows[rord[i]].pos = static_cast<int>(i);

        // Fill direct lookup: cofacet bidx -> sorted position
        for (size_t i = 0; i < rows.size(); ++i)
            rpos[rows[i].bidx] = rows[rord[i]].pos;

        // Cohomology reduction
        int nc = static_cast<int>(cols.size());
        std::vector<int> low(nc, -1);                // column -> cofacet (row) pos
        std::vector<int> row_owner(rows.size(), -1); // cofacet -> column that owns it
        std::vector<std::vector<int>> red(nc);       // reduced columns for XOR

        // Process columns OLDEST FIRST (reverse order: rightmost = youngest)
        for (int pi = nc - 1; pi >= 0; --pi)
        {
            int ci = cord[pi];
            const auto &verts = cols[ci].verts;
            int vn = static_cast<int>(verts.size());
            int max_v = verts.back();

            // Lazily enumerate coboundary: all cofacets sigma  U  {v} for v > max_v
            std::vector<int> coboundary;
            for (int v = max_v + 1; v < n; ++v)
            {
                // Check if all edges from v to existing vertices are within threshold
                double dmax = cols[ci].diam;
                for (int i = 0; i < vn && dmax <= thr; ++i)
                    dmax = std::max(dmax, ed(verts[i], v));
                if (dmax > thr)
                    continue;

                // Compute cofacet binomial index
                int64_t co_bidx = 0;
                int k = 0;
                for (int i = 0; i < vn; ++i)
                {
                    if (verts[i] >= k)
                        co_bidx += C(verts[i], k + 1);
                    ++k;
                }
                if (v >= k)
                    co_bidx += C(v, k + 1);
                // Lookup position
                int pos = rpos[co_bidx];
                if (pos >= 0)
                    coboundary.push_back(pos);
            }

            if (coboundary.empty())
                continue;

            // Sort by youngest first (largest position = smallest diam = youngest... wait)
            // In cohomology: process columns OLDEST first (reverse order).
            // The youngest cofacet has the largest position (most to the right in filtration).
            // So sort descending to get youngest first.
            std::sort(coboundary.begin(), coboundary.end(), std::greater<int>());

            std::vector<int> working = coboundary;
            int col_pos = cols[ci].pos;

            while (!working.empty())
            {
                int pivot = working[0];
                int killer = row_owner[pivot];
                if (killer < 0)
                {
                    low[col_pos] = pivot;
                    row_owner[pivot] = col_pos;
                    break;
                }
                const auto &other = red[killer];
                std::vector<int> merged;
                merged.reserve(working.size() + other.size());
                size_t wa = 0, wb = 0;
                while (wa < working.size() && wb < other.size())
                {
                    if (working[wa] > other[wb])
                        merged.push_back(working[wa++]);
                    else if (other[wb] > working[wa])
                        merged.push_back(other[wb++]);
                    else
                    {
                        ++wa;
                        ++wb;
                    }
                }
                while (wa < working.size())
                    merged.push_back(working[wa++]);
                while (wb < other.size())
                    merged.push_back(other[wb++]);
                working = std::move(merged);
            }
            red[col_pos] = std::move(working);
        }

        // Extract pairs
        // Cohomology: column sigma (d-simplex) with pivot tau ((d+1)-simplex) -> (sigma, tau) dim d
        for (int ci = 0; ci < nc; ++ci)
        {
            int ri = low[ci];
            if (ri < 0)
                continue;
            double birth = cols[ci].diam;
            double death = rows[ri].diam;
            if (death > birth + 1e-12)
                pairs.push_back({birth, death, d});
        }

        // Essential classes in dimension d: zero-reduced columns
        for (int ci = 0; ci < nc; ++ci)
        {
            if (low[ci] < 0)
                pairs.push_back({cols[ci].diam, std::numeric_limits<double>::infinity(), d});
        }

        // Track zero-reduced columns for cross-dimension
        if (d < max_d)
        {
            int64_t nzc = C(n, col_vn);
            prev_zero.assign(nzc + 1, false);
            prev_birth.assign(nzc + 1, 0.0);
            for (int ci = 0; ci < nc; ++ci)
            {
                if (low[ci] < 0)
                {
                    prev_zero[cols[ci].bidx] = true;
                    prev_birth[cols[ci].bidx] = cols[ci].diam;
                }
            }
        }
    }

    // Filter zero-persistence
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [](const Pair &p) {
                                   return p.death > 0 && std::abs(p.death - p.birth) < 1e-10;
                               }),
                pairs.end());
    return pairs;
}

} // namespace nerve::persistence
