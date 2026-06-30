#include "nerve/persistence/utils/exact_engine_fast.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{
namespace
{

static thread_local std::vector<std::vector<int64_t>> t_binom;

void ensure_binom(int n)
{
    if (!t_binom.empty() && (int)t_binom.size() > n)
        return;
    t_binom.assign(n + 1, {});
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
    return t_binom[n][k];
}

int64_t bidx_enc(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    return C(a, 1) + C(b, 2);
}
int64_t bidx_enc(int a, int b, int c)
{
    int x[3] = {a, b, c};
    std::sort(x, x + 3);
    return C(x[0], 1) + C(x[1], 2) + C(x[2], 3);
}

// Decode vertex list from binomial index. v[dim] filled in place (must be pre-allocated).
void bidx_decode(int64_t bidx, int dim, int n, int *v)
{
    int64_t rem = bidx;
    for (int kk = dim; kk >= 1; --kk)
    {
        int lo = 0, hi = n - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (C(mid, kk) <= rem)
                lo = mid;
            else
                hi = mid - 1;
        }
        v[kk - 1] = lo;
        rem -= C(lo, kk);
    }
}

inline uint64_t pack_key(double diam, int64_t bidx)
{
    int64_t di = (int64_t)(diam * 1e6 + 0.5);
    uint64_t hi = (uint64_t)(INT32_MAX - (int32_t)di) << 32;
    uint64_t lo = (uint64_t)bidx & 0xFFFFFFFFU;
    return hi | lo;
}

inline double unpack_diam(uint64_t pk)
{
    return (double)(INT32_MAX - (int32_t)(pk >> 32)) * 1e-6;
}
inline int64_t unpack_bidx(uint64_t pk)
{
    return (int64_t)(pk & 0xFFFFFFFFULL);
}

} // namespace

ExactPersistenceResult
computeExactCohomologyZ2Fast(int n, int max_dim, double thr,
                             const std::vector<std::vector<int>> &neighbors,
                             const std::unordered_map<std::uint64_t, double> &edge_w)
{
    ensure_binom(n);

    int ne = (int)((int64_t)n * (n - 1) / 2);
    std::vector<double> ed_dist(ne + 1, -1.0);
    auto wt = [&](int a, int b) -> double {
        if (a > b)
            std::swap(a, b);
        int k = (int)((int64_t)a * (2LL * n - a - 1) / 2 + (b - a - 1));
        double &c = ed_dist[k];
        if (c >= 0)
            return c;
        std::uint64_t key = (std::uint64_t)(std::uint32_t)a << 32 | (std::uint32_t)b;
        auto it = edge_w.find(key);
        return c = (it != edge_w.end()) ? it->second : std::numeric_limits<double>::infinity();
    };

    // Edge collapse is currently disabled. The ec_neighbors vector is reserved
    // for future use when edge-collapse optimization is re-enabled.
    const auto &nb = neighbors;

    // Pre-enumerate vertices and edges only
    std::vector<double> diam;
    std::vector<int> dims;
    std::vector<int64_t> bidx_S;

    for (int a = 0; a < n; ++a)
    {
        dims.push_back(0);
        diam.push_back(0.0);
        bidx_S.push_back(C(a, 1));
    }

    for (int a = 0; a < n; ++a)
    {
        for (int b : nb[a])
        {
            if (b <= a)
                continue;
            double d = wt(a, b);
            if (d <= thr)
            {
                dims.push_back(1);
                diam.push_back(d);
                bidx_S.push_back(bidx_enc(a, b));
            }
        }
    }

    int S = (int)dims.size();

    // Sort by filtration: (diam asc, dim asc, bidx asc)
    std::vector<int> ord(S);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int x, int y) {
        if (std::abs(diam[x] - diam[y]) > 1e-12)
            return diam[x] < diam[y];
        if (dims[x] != dims[y])
            return dims[x] < dims[y];
        return bidx_S[x] < bidx_S[y];
    });
    std::vector<int> pos(S);
    for (int i = 0; i < S; ++i)
        pos[ord[i]] = i;

    // Position lookups
    std::vector<int> b2pv0(n, -1), b2pv1(C(n, 2), -1);
    for (int i = 0; i < S; ++i)
    {
        int64_t b = bidx_S[i];
        int d = dims[i];
        if (d == 0 && b < (int64_t)b2pv0.size())
            b2pv0[b] = pos[i];
        else if (d == 1 && b < (int64_t)b2pv1.size())
            b2pv1[b] = pos[i];
    }
    auto bpos_edge = [&](int64_t b) -> int { return b < (int64_t)b2pv1.size() ? b2pv1[b] : -1; };

    // Vertex data for simplex decoding
    std::vector<int> verts_d;
    std::vector<int> vs_off;
    for (int a = 0; a < n; ++a)
    {
        vs_off.push_back((int)verts_d.size());
        verts_d.push_back(a);
    }
    for (int a = 0; a < n; ++a)
    {
        for (int b : nb[a])
        {
            if (b <= a)
                continue;
            if (wt(a, b) <= thr)
            {
                vs_off.push_back((int)verts_d.size());
                int x = std::min(a, b), y = std::max(a, b);
                verts_d.push_back(x);
                verts_d.push_back(y);
            }
        }
    }
    auto getv = [&](int oi) -> const int * { return verts_d.data() + vs_off[oi]; };

    // Union-find dim-0
    std::vector<Pair> pairs;
    pairs.reserve(S / 4 + 10);
    std::unordered_map<int, int> owner;

    {
        std::vector<std::tuple<double, int64_t, int, int>> ufe;
        for (int a = 0; a < n; ++a)
            for (int b : nb[a])
            {
                if (b <= a)
                    continue;
                double d = wt(a, b);
                if (d <= thr)
                    ufe.emplace_back(d, (int64_t)0, a, b);
            }
        for (auto &t : ufe)
        {
            auto &[w, bid, uu, vv] = t;
            if (uu > vv)
                std::swap(uu, vv);
            bid = bidx_enc(uu, vv);
        }
        std::sort(ufe.begin(), ufe.end(), [](auto &x, auto &y) {
            if (std::abs(std::get<0>(x) - std::get<0>(y)) > 1e-12)
                return std::get<0>(x) < std::get<0>(y);
            return std::get<1>(x) > std::get<1>(y);
        });

        std::vector<int> par(n), dead(n, false);
        std::iota(par.begin(), par.end(), 0);
        std::vector<double> birth(n, 0.0);
        auto fnd = [&](int x) -> int {
            while (par[x] != x)
            {
                par[x] = par[par[x]];
                x = par[x];
            }
            return x;
        };
        for (auto &[w, bid, uu, vv] : ufe)
        {
            int ru = fnd(uu), rv = fnd(vv);
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

        // Mark MST edges as cleared
        std::iota(par.begin(), par.end(), 0);
        for (auto &[w, bid, uu, vv] : ufe)
        {
            int ru = fnd(uu), rv = fnd(vv);
            if (ru != rv)
            {
                if (ru < rv)
                    std::swap(ru, rv);
                par[ru] = rv;
                int64_t eb = bidx_enc(uu, vv);
                int ep = bpos_edge(eb);
                if (ep >= 0)
                    owner[ep] = -2;
            }
        }
    }

    // Triangle position table
    std::unordered_map<int64_t, int> b2pm2; // triangle bidx -> position
    std::unordered_map<int64_t, double> tri_filtration;
    std::unordered_map<int64_t, double> tet_filtration;

    // Dim-1 pair tracking
    std::vector<bool> is_apparent_pair(S, false);

    // Column reduction state
    std::vector<int> low(S, -1);
    std::vector<std::vector<uint64_t>> red(S);
    std::vector<bool> was_processed(S, false);
    std::unordered_map<int, std::vector<uint64_t>> red_extra;
    std::unordered_map<int, int> owner_dim2;
    std::vector<uint64_t> merge_buf1, merge_buf2;

    // Insert entry e into sorted-descending vector wr (Z2: cancel duplicates)
    auto xor_insert = [](std::vector<uint64_t> &wr, uint64_t e) {
        auto it = std::lower_bound(wr.begin(), wr.end(), e, std::greater<uint64_t>());
        if (it != wr.end() && *it == e)
            wr.erase(it);
        else
            wr.insert(it, e);
    };

    // XOR merge: wr = wr XOR o (symmetric difference of sorted descending arrays)
    auto xor_merge = [&](std::vector<uint64_t> &wr, const std::vector<uint64_t> &o) {
        merge_buf1.clear();
        merge_buf1.reserve(wr.size() + o.size());
        size_t wa = 0, wb = 0;
        while (wa < wr.size() && wb < o.size())
        {
            if (wr[wa] == o[wb])
            {
                ++wa;
                ++wb;
                continue;
            }
            if (wr[wa] > o[wb])
                merge_buf1.push_back(wr[wa++]);
            else
                merge_buf1.push_back(o[wb++]);
        }
        while (wa < wr.size())
            merge_buf1.push_back(wr[wa++]);
        while (wb < o.size())
            merge_buf1.push_back(o[wb++]);
        wr.swap(merge_buf1);
    };

    int max_simplex_dim = std::min(max_dim + 2, n - 1);

    auto enum_tri_cofs_into = [&](int edge_oi, std::vector<uint64_t> &out) {
        out.clear();
        const int *vs = getv(edge_oi);
        int a = vs[0], b = vs[1];
        const auto &na = nb[a];
        const auto &nb2 = nb[b];
        double edge_diam = diam[edge_oi];
        size_t pa = 0, pb = 0;
        while (pa < na.size() && pb < nb2.size())
        {
            if (na[pa] == nb2[pb])
            {
                int c = na[pa];
                double td = edge_diam;
                double dac = wt(a, c);
                if (dac > thr)
                {
                    pa++;
                    pb++;
                    continue;
                }
                if (dac > td)
                    td = dac;
                double dbc = wt(b, c);
                if (dbc > thr)
                {
                    pa++;
                    pb++;
                    continue;
                }
                if (dbc > td)
                    td = dbc;
                xor_insert(out, pack_key(td, bidx_enc(a, b, c)));
                pa++;
                pb++;
            }
            else if (na[pa] < nb2[pb])
                pa++;
            else
                pb++;
        }
    };

    // Dim-1 reduction (edge -> triangle)
    if (max_dim >= 1 && max_simplex_dim >= 2)
    {
        // Collect columns with positions for sorting
        std::vector<std::pair<int, int>> col_order; // (cp, oi)

        for (int oi = 0; oi < S; ++oi)
        {
            if (dims[oi] != 1)
                continue;
            int cp = pos[oi];

            if (owner.count(cp))
            {
                // MST edge: only store triangle filtrations (fast path)
                const int *vs = getv(oi);
                int a = vs[0], b = vs[1];
                const auto &na2 = nb[a];
                const auto &nb2_2 = nb[b];
                size_t pa = 0, pb = 0;
                double edge_diam = diam[oi];
                while (pa < na2.size() && pb < nb2_2.size())
                {
                    if (na2[pa] == nb2_2[pb])
                    {
                        int c = na2[pa];
                        double td = edge_diam;
                        double dac = wt(a, c);
                        if (dac > thr)
                        {
                            pa++;
                            pb++;
                            continue;
                        }
                        if (dac > td)
                            td = dac;
                        double dbc = wt(b, c);
                        if (dbc > thr)
                        {
                            pa++;
                            pb++;
                            continue;
                        }
                        if (dbc > td)
                            td = dbc;
                        tri_filtration[bidx_enc(a, b, c)] = td;
                        pa++;
                        pb++;
                    }
                    else if (na2[pa] < nb2_2[pb])
                        pa++;
                    else
                        pb++;
                }
                continue;
            }

            // Non-MST edge: full enumeration with pack_keys for reduction.
            enum_tri_cofs_into(oi, red[cp]);

            if (red[cp].empty())
            {
                pairs.push_back({diam[oi], std::numeric_limits<double>::infinity(), 1});
                continue;
            }

            // Apparent pair check (red[cp] is already sorted descending)
            bool apparent = false;
            {
                double od = diam[oi];
                int64_t zpc_bidx = -1;
                double zpc_diam = 0;
                for (int pi = 0; pi < (int)red[cp].size(); ++pi)
                {
                    double kd = unpack_diam(red[cp][pi]);
                    if (std::abs(kd - od) < 1e-12)
                    {
                        zpc_diam = kd;
                        zpc_bidx = unpack_bidx(red[cp][pi]);
                        break;
                    }
                }
                if (zpc_bidx >= 0)
                {
                    int zv[3];
                    bidx_decode(zpc_bidx, 3, n, zv);
                    apparent = true;
                    for (int fk = 2; fk >= 0 && apparent; --fk)
                    {
                        int fv[2];
                        int fi = 0;
                        for (int jj = 0; jj < 3; ++jj)
                            if (jj != fk)
                                fv[fi++] = zv[jj];
                        if (fv[0] > fv[1])
                            std::swap(fv[0], fv[1]);
                        int64_t fb = bidx_enc(fv[0], fv[1]);
                        int fp = bpos_edge(fb);
                        if (fp < 0)
                        {
                            apparent = false;
                            break;
                        }
                        if (std::abs(diam[ord[fp]] - zpc_diam) < 1e-12)
                        {
                            if (fp != cp)
                                apparent = false;
                            break;
                        }
                    }
                }
            }
            if (apparent)
            {
                is_apparent_pair[oi] = true;
                pairs.push_back({diam[oi], diam[oi], 1});
                red[cp].clear();
                continue;
            }

            was_processed[cp] = true;
            col_order.emplace_back(cp, oi);
        }

        // Sort by position descending (youngest first) for reduction order
        std::sort(col_order.begin(), col_order.end(),
                  [](auto &a, auto &b) { return a.first > b.first; });

        // Reduce columns in order
        for (auto &[cp, oi] : col_order)
        {
            if (red[cp].empty())
                continue;
            auto &wr = red[cp];

            while (!wr.empty())
            {
                uint64_t pk = wr[0];
                int64_t b = unpack_bidx(pk);

                int pivot_pos = -1;
                auto it = b2pm2.find(b);
                if (it != b2pm2.end())
                    pivot_pos = it->second;
                else
                {
                    pivot_pos = S + (int)b2pm2.size();
                    b2pm2[b] = pivot_pos;
                }

                auto oit = owner.find(pivot_pos);
                int k = (oit != owner.end()) ? oit->second : -1;
                if (k < 0)
                {
                    low[cp] = pivot_pos;
                    owner[pivot_pos] = cp;
                    break;
                }
                if (k >= 0 && k < S)
                    xor_merge(wr, red[k]);
                else if (k >= S)
                {
                    auto rit = red_extra.find(k);
                    if (rit != red_extra.end())
                        xor_merge(wr, rit->second);
                    else
                        break;
                }
            }
        }

        // Extract dim-1 pairs
        for (auto &[cp, oi] : col_order)
        {
            if (low[cp] < 0)
            {
                pairs.push_back({diam[ord[cp]], std::numeric_limits<double>::infinity(), 1});
                continue;
            }
            int oi2 = ord[cp];
            double bv = diam[oi2];
            if (!red[cp].empty())
            {
                double death_diam = unpack_diam(red[cp][0]);
                pairs.push_back({bv, death_diam, 1});
            }
        }
    }

    int raw_edge_count = ne;
    (void)raw_edge_count;

    if (max_dim >= 2 && max_simplex_dim >= 3)
    {
        constexpr double kInf = std::numeric_limits<double>::infinity();
        struct ColMeta
        {
            int64_t bidx;
            double diam;
            int sid;
            bool pivoted;
            bool handled;
        };
        std::vector<ColMeta> prev_cols;
        int next_sid = S;
        std::unordered_set<int64_t> seen_tris;
        seen_tris.reserve((int)tri_filtration.size() + (int)((int64_t)n * (n - 1)));

        std::unordered_set<int64_t> killed_tris;
        for (const auto &[bidx, pos_val] : b2pm2)
            if (owner.count(pos_val) && owner.at(pos_val) >= 0)
                killed_tris.insert(bidx);
        for (int oi = 0; oi < S; ++oi)
        {
            if (dims[oi] != 1)
                continue;
            int cp = pos[oi];
            if (owner.count(cp))
                continue;
            const auto &wr = red[cp];
            if (!wr.empty())
                continue;
            const int *vs = getv(oi);
            int a = vs[0], b = vs[1];
            const auto &na2 = nb[a], &nb2_2 = nb[b];
            size_t pa = 0, pb = 0;
            while (pa < na2.size() && pb < nb2_2.size())
            {
                if (na2[pa] == nb2_2[pb])
                {
                    int c = na2[pa];
                    if (wt(a, c) <= thr && wt(b, c) <= thr)
                        killed_tris.insert(bidx_enc(a, b, c));
                    pa++;
                    pb++;
                }
                else if (na2[pa] < nb2_2[pb])
                    pa++;
                else
                    pb++;
            }
        }

        for (auto &[bidx, filt] : tri_filtration)
        {
            if (killed_tris.count(bidx))
                continue;
            b2pm2[bidx] = next_sid;
            prev_cols.push_back({bidx, filt, next_sid++, false, false});
            seen_tris.insert(bidx);
        }

        for (int oi = 0; oi < S; ++oi)
        {
            if (dims[oi] != 1)
                continue;
            int cp = pos[oi];
            const auto &wr = red[cp];
            for (uint64_t pk : wr)
            {
                int64_t tb = unpack_bidx(pk);
                if (!seen_tris.insert(tb).second || killed_tris.count(tb))
                    continue;
                double td = unpack_diam(pk);
                b2pm2[tb] = next_sid;
                prev_cols.push_back({tb, td, next_sid++, false, false});
            }
            if (wr.empty() && owner.count(cp) == 0)
            {
                const int *vs = getv(oi);
                int a = vs[0], b = vs[1];
                const auto &na2 = nb[a], &nb2_2 = nb[b];
                size_t pa = 0, pb = 0;
                double edge_diam = diam[oi];
                while (pa < na2.size() && pb < nb2_2.size())
                {
                    if (na2[pa] == nb2_2[pb])
                    {
                        int c = na2[pa];
                        double td = edge_diam;
                        double dac = wt(a, c);
                        if (dac > thr)
                        {
                            pa++;
                            pb++;
                            continue;
                        }
                        if (dac > td)
                            td = dac;
                        double dbc = wt(b, c);
                        if (dbc > thr)
                        {
                            pa++;
                            pb++;
                            continue;
                        }
                        if (dbc > td)
                            td = dbc;
                        int64_t tbidx = bidx_enc(a, b, c);
                        if (!killed_tris.count(tbidx) && seen_tris.insert(tbidx).second)
                        {
                            tri_filtration[tbidx] = td;
                            b2pm2[tbidx] = next_sid;
                            prev_cols.push_back({tbidx, td, next_sid++, false, false});
                        }
                        pa++;
                        pb++;
                    }
                    else if (na2[pa] < nb2_2[pb])
                        pa++;
                    else
                        pb++;
                }
            }
        }

        for (int d = 2; d <= max_dim && !prev_cols.empty(); ++d)
        {
            int col_vn = d + 1, row_vn = d + 2;
            if (row_vn > n)
                break;
            std::vector<int> cord(prev_cols.size());
            std::iota(cord.begin(), cord.end(), 0);
            std::sort(cord.begin(), cord.end(), [&](int x, int y) {
                if (std::abs(prev_cols[x].diam - prev_cols[y].diam) > 1e-12)
                    return prev_cols[x].diam < prev_cols[y].diam;
                return prev_cols[x].bidx < prev_cols[y].bidx;
            });

            std::unordered_map<int64_t, int> row_pos;
            std::unordered_map<int64_t, double> row_diam;
            std::unordered_map<int, int> col_owner;
            std::vector<int> cv(col_vn);
            static thread_local std::vector<uint64_t> cof_buf;

            // Pre-allocate row_pos to avoid rehashing
            row_pos.reserve(C(n, row_vn)); // max possible unique cofacets

            for (int pi = (int)cord.size() - 1; pi >= 0; --pi)
            {
                auto &col = prev_cols[cord[pi]];
                bidx_decode(col.bidx, col_vn, n, cv.data());
                cof_buf.clear();

                if (col_vn == 3)
                {
                    int a = cv[0], b = cv[1], c = cv[2];
                    int64_t idx = bidx_enc(a, b, c), idx_below = idx, idx_above = 0;
                    int j = n - 1, kk = 3;
                    while (j >= 0)
                    {
                        while (kk > 0 && C(j, kk) <= idx_below)
                        {
                            idx_below -= C(j, kk);
                            idx_above += C(j, kk + 1);
                            --j;
                            --kk;
                        }
                        if (j < 0)
                            break;
                        double nd = col.diam;
                        for (int vi = 0; vi < 3; ++vi)
                        {
                            double wv = wt(cv[vi], j);
                            if (wv > thr)
                            {
                                nd = -1;
                                break;
                            }
                            if (wv > nd)
                                nd = wv;
                        }
                        if (nd >= 0)
                        {
                            int vs2[4] = {a, b, c, j};
                            std::sort(vs2, vs2 + 4);
                            int64_t cf = 0;
                            for (int ki = 0; ki < 4; ++ki)
                                if (vs2[ki] >= ki)
                                    cf += C(vs2[ki], ki + 1);
                            xor_insert(cof_buf, pack_key(nd, cf));
                        }
                        --j;
                    }
                }
                else
                {
                    int best_i = 0;
                    for (int vi = 1; vi < col_vn; ++vi)
                        if (nb[cv[vi]].size() < nb[cv[best_i]].size())
                            best_i = vi;
                    for (int v : nb[cv[best_i]])
                    {
                        bool in_sim = false;
                        for (int vi = 0; vi < col_vn; ++vi)
                            if (cv[vi] == v)
                            {
                                in_sim = true;
                                break;
                            }
                        if (in_sim)
                            continue;
                        bool ok = true;
                        for (int vi = 0; vi < col_vn && ok; ++vi)
                            if (vi != best_i &&
                                !std::binary_search(nb[cv[vi]].begin(), nb[cv[vi]].end(), v))
                                ok = false;
                        if (!ok)
                            continue;
                        double nd = col.diam;
                        for (int vi = 0; vi < col_vn; ++vi)
                        {
                            double wv = wt(cv[vi], v);
                            if (wv > thr)
                            {
                                nd = -1;
                                break;
                            }
                            if (wv > nd)
                                nd = wv;
                        }
                        if (nd >= 0)
                        {
                            std::vector<int> vs(cv.begin(), cv.begin() + col_vn);
                            vs.push_back(v);
                            std::sort(vs.begin(), vs.end());
                            int64_t cf = 0;
                            for (int ki = 0; ki < row_vn; ++ki)
                                if (vs[ki] >= ki)
                                    cf += C(vs[ki], ki + 1);
                            xor_insert(cof_buf, pack_key(nd, cf));
                        }
                    }
                }

                if (cof_buf.empty())
                {
                    col.handled = true;
                    pairs.push_back({col.diam, kInf, d});
                    continue;
                }
                for (uint64_t pk : cof_buf)
                {
                    int64_t rb = unpack_bidx(pk);
                    row_diam[rb] = unpack_diam(pk);
                    row_pos.try_emplace(rb, S + (int)row_pos.size());
                }
                auto &wr = cof_buf;
                while (!wr.empty())
                {
                    uint64_t pk = wr[0];
                    int64_t b = unpack_bidx(pk);
                    auto rp = row_pos.find(b);
                    if (rp == row_pos.end())
                        break;
                    auto oit = col_owner.find(rp->second);
                    if (oit == col_owner.end())
                    {
                        col.pivoted = true;
                        col_owner[rp->second] = col.sid;
                        break;
                    }
                    int kid = oit->second;
                    if (kid >= 0 && kid < S)
                        xor_merge(wr, red[kid]);
                    else
                    {
                        auto rit = red_extra.find(kid);
                        if (rit != red_extra.end())
                            xor_merge(wr, rit->second);
                        else
                            break;
                    }
                }
                if (col.sid < S)
                    red[col.sid] = std::move(wr);
                else
                    red_extra[col.sid] = std::move(wr);
            }

            for (auto &col : prev_cols)
            {
                if (col.handled)
                    continue;
                if (col.pivoted)
                {
                    int sid = col.sid;
                    auto &stored = (sid < S) ? red[sid] : red_extra[sid];
                    if (!stored.empty())
                    {
                        int64_t rb = unpack_bidx(stored[0]);
                        auto rit = row_diam.find(rb);
                        if (rit != row_diam.end() && rit->second > col.diam + 1e-12)
                            pairs.push_back({col.diam, rit->second, d});
                    }
                }
                else
                    pairs.push_back({col.diam, kInf, d});
            }
            if (d >= max_dim)
                break;
            std::vector<ColMeta> next_cols;
            std::unordered_set<int64_t> next_seen;
            for (auto &col : prev_cols)
            {
                if (!col.pivoted || col.handled)
                    continue;
                int sid = col.sid;
                auto &stored = (sid < S) ? red[sid] : red_extra[sid];
                for (uint64_t pk : stored)
                {
                    int64_t cb = unpack_bidx(pk);
                    auto rit = row_diam.find(cb);
                    if (rit != row_diam.end() && next_seen.insert(cb).second)
                        next_cols.push_back({cb, rit->second, next_sid++, false, false});
                }
            }
            prev_cols = std::move(next_cols);
        }
    }

    {
        size_t w = 0;
        for (size_t r = 0; r < pairs.size(); ++r)
        {
            const auto &p = pairs[r];
            double diff = p.death - p.birth;
            if (!(p.death > 0 && diff > -1e-5 && diff < 1e-5))
                pairs[w++] = pairs[r];
        }
        pairs.resize(w);
    }

    std::vector<Size> betti(std::max(0, max_dim + 1), 0);
    for (const auto &p : pairs)
        if (std::isinf(p.death) && p.dimension < (int)betti.size())
            betti[p.dimension]++;
    return {pairs, betti, 0};
}

} // namespace nerve::persistence
