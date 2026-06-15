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
int64_t bidx_enc(int a, int b, int c, int d)
{
    int x[4] = {a, b, c, d};
    std::sort(x, x + 4);
    return C(x[0], 1) + C(x[1], 2) + C(x[2], 3) + C(x[3], 4);
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

    auto enum_tetrahedron_cofacets = [&](const int *tri_v, double tri_diam,
                                         std::vector<uint64_t> &out, bool sparse) {
        out.clear();
        int a = tri_v[0], b = tri_v[1], c = tri_v[2];
        if (sparse && n <= 2000)
        {
            static thread_local std::vector<int> common;
            size_t pa = 0, pb = 0, pc = 0;
            const auto &na = nb[a];
            const auto &nb2 = nb[b];
            const auto &nc2 = nb[c];
            common.clear();
            while (pa < na.size() && pb < nb2.size() && pc < nc2.size())
            {
                if (na[pa] == nb2[pb] && na[pa] == nc2[pc])
                {
                    if (na[pa] > c)
                        common.push_back(na[pa]);
                    pa++;
                    pb++;
                    pc++;
                }
                else if (na[pa] <= nb2[pb] && na[pa] <= nc2[pc])
                    pa++;
                else if (nb2[pb] <= na[pa] && nb2[pb] <= nc2[pc])
                    pb++;
                else
                    pc++;
            }
            for (int d : common)
            {
                double nd = tri_diam;
                double wd = wt(a, d);
                if (wd > thr)
                    continue;
                if (wd > nd)
                    nd = wd;
                wd = wt(b, d);
                if (wd > thr)
                    continue;
                if (wd > nd)
                    nd = wd;
                wd = wt(c, d);
                if (wd > thr)
                    continue;
                if (wd > nd)
                    nd = wd;
                xor_insert(out, pack_key(nd, bidx_enc(a, b, c, d)));
            }
        }
        else
        {
            int64_t idx = bidx_enc(a, b, c);
            int64_t idx_below = idx, idx_above = 0;
            int j = n - 1, k = 3;
            while (j >= 0)
            {
                while (k > 0 && C(j, k) <= idx_below)
                {
                    idx_below -= C(j, k);
                    idx_above += C(j, k + 1);
                    --j;
                    --k;
                }
                if (j < 0)
                    break;
                int64_t cf_idx = idx_above + C(j, k + 1) + idx_below;
                double cf_diam = tri_diam;
                for (int vi = 0; vi < 3; ++vi)
                {
                    double nd = wt(tri_v[vi], j);
                    if (nd > thr)
                    {
                        cf_diam = -1.0;
                        break;
                    }
                    if (nd > cf_diam)
                        cf_diam = nd;
                }
                if (cf_diam >= 0)
                    xor_insert(out, pack_key(cf_diam, cf_idx));
                --j;
            }
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

            // Non-MST edge: full enumeration with pack_keys for reduction
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
                int64_t tb = unpack_bidx(red[cp][0]);
                auto tit = tri_filtration.find(tb);
                if (tit != tri_filtration.end())
                {
                    pairs.push_back({bv, tit->second, 1});
                }
            }
        }
    }

    int raw_edge_count = ne;

    // Dim-2 reduction (triangle -> tetrahedron, LAZY)
    if (max_dim >= 2 && max_simplex_dim >= 3)
    {
        // Collect triangles from tri_filtration
        std::unordered_set<int64_t> cleared_tris;
        for (auto &[bidx, tpos] : b2pm2)
        {
            if (owner.count(tpos))
                cleared_tris.insert(bidx);
        }

        struct TriColumn
        {
            int64_t bidx;
            double diam;
            int tpos;
            bool pivoted;
            bool handled;
        };
        std::vector<TriColumn> dim2_cols;

        for (auto &[bidx, filt] : tri_filtration)
        {
            int tpos = -1;
            auto tit = b2pm2.find(bidx);
            if (tit != b2pm2.end())
                tpos = tit->second;
            else
            {
                tpos = S + (int)b2pm2.size();
                b2pm2[bidx] = tpos;
            }
            dim2_cols.push_back({bidx, filt, tpos, false, false});
        }
        std::sort(dim2_cols.begin(), dim2_cols.end(),
                  [](auto &a, auto &b) { return a.diam > b.diam; });

        std::unordered_map<int64_t, int> b2pm3;
        std::vector<uint64_t> cof_buf;
        int tv[3];

        for (auto &tc : dim2_cols)
        {
            bidx_decode(tc.bidx, 3, n, tv);

            enum_tetrahedron_cofacets(tv, tc.diam, cof_buf, raw_edge_count < n * n / 4);

            if (cof_buf.empty())
            {
                tc.handled = true;
                pairs.push_back({tc.diam, std::numeric_limits<double>::infinity(), 2});
                continue;
            }

            // Apparent pair check
            bool apparent = false;
            {
                double od = tc.diam;
                int64_t zpc_bidx = -1;
                double zpc_diam = 0;
                for (int pi = 0; pi < (int)cof_buf.size(); ++pi)
                {
                    double kd = unpack_diam(cof_buf[pi]);
                    if (std::abs(kd - od) < 1e-12)
                    {
                        zpc_diam = kd;
                        zpc_bidx = unpack_bidx(cof_buf[pi]);
                        break;
                    }
                }
                if (zpc_bidx >= 0)
                {
                    int zv[4];
                    bidx_decode(zpc_bidx, 4, n, zv);
                    apparent = true;
                    for (int fk = 3; fk >= 0 && apparent; --fk)
                    {
                        int fv[3];
                        int fi = 0;
                        for (int jj = 0; jj < 4; ++jj)
                            if (jj != fk)
                                fv[fi++] = zv[jj];
                        std::sort(fv, fv + 3);
                        int64_t fb = bidx_enc(fv[0], fv[1], fv[2]);
                        auto tit2 = tri_filtration.find(fb);
                        if (tit2 == tri_filtration.end())
                        {
                            apparent = false;
                            break;
                        }
                        if (std::abs(tit2->second - zpc_diam) < 1e-12)
                        {
                            if (fb != tc.bidx)
                                apparent = false;
                            break;
                        }
                    }
                }
            }
            if (apparent)
            {
                tc.handled = true;
                pairs.push_back({tc.diam, tc.diam, 2});
                continue;
            }

            // Assign dynamic tetrahedron positions
            for (uint64_t pk : cof_buf)
            {
                int64_t b = unpack_bidx(pk);
                double dm = unpack_diam(pk);
                tet_filtration[b] = dm;
                if (b2pm3.find(b) == b2pm3.end())
                    b2pm3.try_emplace(b, S + (int)b2pm3.size());
            }

            // Reduction
            auto &wr = cof_buf;
            while (!wr.empty())
            {
                uint64_t pk = wr[0];
                int64_t b = unpack_bidx(pk);
                auto it3 = b2pm3.find(b);
                if (it3 == b2pm3.end())
                    break;
                int pivot_pos = it3->second;

                auto oit = owner_dim2.find(pivot_pos);
                if (oit == owner_dim2.end())
                {
                    tc.pivoted = true;
                    owner_dim2[pivot_pos] = tc.tpos;
                    break;
                }
                int k = oit->second;
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
                else
                    break;
            }
            if (tc.tpos < S)
                red[tc.tpos] = std::move(wr);
            else
                red_extra[tc.tpos] = std::move(wr);
        }

        // Extract dim-2 pairs
        for (auto &tc : dim2_cols)
        {
            if (tc.handled)
                continue;
            if (tc.pivoted)
            {
                auto &stored = (tc.tpos < S) ? red[tc.tpos] : red_extra[tc.tpos];
                if (!stored.empty())
                {
                    int64_t tb = unpack_bidx(stored[0]);
                    auto tit = tet_filtration.find(tb);
                    if (tit != tet_filtration.end())
                    {
                        pairs.push_back({tc.diam, tit->second, 2});
                    }
                }
            }
            else
            {
                pairs.push_back({tc.diam, std::numeric_limits<double>::infinity(), 2});
            }
        }
    }

    // Zero-persistence filter
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [](const Pair &p) {
                                   return p.death > 0 && std::abs(p.death - p.birth) < 1e-10;
                               }),
                pairs.end());

    // Build dim_to_cols for each dimension
    std::vector<Size> betti(max_dim + 1, 0);
    for (const auto &p : pairs)
        if (std::isinf(p.death) && p.dimension < (int)betti.size())
            betti[p.dimension]++;

    return {pairs, betti, 0};
}

} // namespace nerve::persistence
