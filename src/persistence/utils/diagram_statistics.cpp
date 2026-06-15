
#include "nerve/persistence/utils/diagram_statistics.hpp"

#include <algorithm>
#include <cmath>
namespace nerve::persistence
{

std::vector<Size> bettiNumbersFromPairs(const std::vector<Pair> &pairs)
{
    Size max_dim = 0;
    for (const auto &pair : pairs)
    {
        if (pair.dimension >= 0)
        {
            max_dim = std::max(max_dim, static_cast<Size>(pair.dimension));
        }
    }
    std::vector<Size> betti(max_dim + 1, 0);
    for (const auto &pair : pairs)
    {
        if (pair.dimension >= 0 && pair.isInfinite())
        {
            const Size dim = static_cast<Size>(pair.dimension);
            if (dim < betti.size())
            {
                betti[dim]++;
            }
        }
    }
    return betti;
}

double shannonEntropyNormalized(const std::vector<double> &weights)
{
    double total = 0.0;
    for (double w : weights)
    {
        if (w > 0.0 && std::isfinite(w))
        {
            total += w;
        }
    }
    if (total <= 0.0)
    {
        return 0.0;
    }
    double entropy = 0.0;
    for (double w : weights)
    {
        if (w > 0.0 && std::isfinite(w))
        {
            const double p = w / total;
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

} // namespace nerve::persistence
