#include <limits>
#include <vector>

namespace nerve::metrics
{

double hungarianTotalCost(const std::vector<std::vector<double>> &cost)
{
    const size_t n = cost.size();
    if (n == 0)
    {
        return 0.0;
    }

    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<size_t> p(n + 1, 0);
    std::vector<size_t> way(n + 1, 0);

    for (size_t i = 1; i <= n; ++i)
    {
        p[0] = i;
        size_t j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);

        do
        {
            used[j0] = true;
            const size_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            size_t j1 = 0;

            for (size_t j = 1; j <= n; ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (size_t j = 0; j <= n; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<size_t> assignment(n, 0);
    for (size_t j = 1; j <= n; ++j)
    {
        if (p[j] != 0)
        {
            assignment[p[j] - 1] = j - 1;
        }
    }

    double total = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        total += cost[i][assignment[i]];
    }
    return total;
}

} // namespace nerve::metrics
