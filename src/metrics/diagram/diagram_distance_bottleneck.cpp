#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace nerve::metrics
{

constexpr double kDistanceEpsilon = 1e-12;

struct DiagramPoint
{
    double birth = 0.0;
    double death = 0.0;
};

inline double linfDistance(const DiagramPoint &lhs, const DiagramPoint &rhs)
{
    return std::max(std::abs(lhs.birth - rhs.birth), std::abs(lhs.death - rhs.death));
}

inline double diagonalDistance(const DiagramPoint &point)
{
    return std::abs(point.death - point.birth) * 0.5;
}

inline bool hasSupportedMatchingSize(size_t n1, size_t n2)
{
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
    return n1 <= max_index && n2 <= max_index && n1 <= max_index - n2;
}

bool checkedCandidateCount(size_t n1, size_t n2, size_t &count)
{
    constexpr size_t max_size = std::numeric_limits<size_t>::max();
    count = 1;
    if (n1 > max_size - count)
    {
        return false;
    }
    count += n1;
    if (n2 > max_size - count)
    {
        return false;
    }
    count += n2;
    if (n1 != 0 && n2 > (max_size - count) / n1)
    {
        return false;
    }
    count += n1 * n2;
    return true;
}

bool tryAugment(size_t left, const std::vector<std::vector<size_t>> &adjacency,
                std::vector<bool> &seenRight, std::vector<int> &rightMatch)
{
    for (size_t right : adjacency[left])
    {
        if (seenRight[right])
        {
            continue;
        }
        seenRight[right] = true;
        if (rightMatch[right] < 0 ||
            tryAugment(static_cast<size_t>(rightMatch[right]), adjacency, seenRight, rightMatch))
        {
            rightMatch[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

bool hasPerfectMatching(const std::vector<DiagramPoint> &points1,
                        const std::vector<DiagramPoint> &points2, double threshold)
{
    const size_t n1 = points1.size();
    const size_t n2 = points2.size();
    const size_t n = n1 + n2;
    if (n == 0)
    {
        return true;
    }

    std::vector<std::vector<size_t>> adjacency(n);

    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            if (linfDistance(points1[i], points2[j]) <= threshold + kDistanceEpsilon)
            {
                adjacency[i].push_back(j);
            }
        }
        if (diagonalDistance(points1[i]) <= threshold + kDistanceEpsilon)
        {
            adjacency[i].push_back(n2 + i);
        }
    }

    for (size_t j = 0; j < n2; ++j)
    {
        if (diagonalDistance(points2[j]) <= threshold + kDistanceEpsilon)
        {
            adjacency[n1 + j].push_back(j);
        }
    }

    std::vector<int> rightMatch(n, -1);
    size_t matched = 0;
    for (size_t left = 0; left < n; ++left)
    {
        std::vector<bool> seenRight(n, false);
        if (tryAugment(left, adjacency, seenRight, rightMatch))
        {
            ++matched;
        }
    }
    return matched == n;
}

double computeBottleneck(const std::vector<DiagramPoint> &points1,
                         const std::vector<DiagramPoint> &points2)
{
    if (points1.empty() && points2.empty())
    {
        return 0.0;
    }
    if (!hasSupportedMatchingSize(points1.size(), points2.size()))
    {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> candidates;
    size_t candidate_count = 0;
    if (!checkedCandidateCount(points1.size(), points2.size(), candidate_count))
    {
        return std::numeric_limits<double>::infinity();
    }
    candidates.reserve(candidate_count);
    candidates.push_back(0.0);

    for (const auto &point : points1)
    {
        candidates.push_back(diagonalDistance(point));
    }
    for (const auto &point : points2)
    {
        candidates.push_back(diagonalDistance(point));
    }
    for (const auto &lhs : points1)
    {
        for (const auto &rhs : points2)
        {
            candidates.push_back(linfDistance(lhs, rhs));
        }
    }

    std::ranges::sort(candidates);
    const auto dedup = std::ranges::unique(
        candidates, [](double a, double b) { return std::abs(a - b) <= kDistanceEpsilon; });
    candidates.erase(dedup.begin(), dedup.end());

    if (candidates.empty())
    {
        return 0.0;
    }

    size_t lo = 0;
    size_t hi = candidates.size() - 1;
    double best = candidates[hi];
    while (lo <= hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        const double threshold = candidates[mid];
        if (hasPerfectMatching(points1, points2, threshold))
        {
            best = threshold;
            if (mid == 0)
            {
                break;
            }
            hi = mid - 1;
        }
        else
        {
            lo = mid + 1;
        }
    }

    return best;
}

} // namespace nerve::metrics
