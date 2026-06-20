#pragma once

#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::graphs::detail
{

using Interval = std::pair<double, double>;
using SectionMap = std::map<Index, std::vector<double>>;

struct ComponentForest
{
    std::vector<Size> parent;
    std::vector<double> birth;

    Size add(double birth_time)
    {
        const Size id = parent.size();
        parent.push_back(id);
        birth.push_back(birth_time);
        return id;
    }

    Size find(Size id)
    {
        while (parent[id] != id)
        {
            parent[id] = parent[parent[id]];
            id = parent[id];
        }
        return id;
    }

    void unite(Size left, Size right, double death_time, std::vector<Interval> &intervals)
    {
        Size root_left = find(left);
        Size root_right = find(right);
        assert(root_left < birth.size());
        assert(root_right < birth.size());
        if (root_left == root_right)
        {
            return;
        }

        Size survivor = root_left;
        Size dying = root_right;
        if (birth[root_right] < birth[root_left] ||
            (birth[root_right] == birth[root_left] && root_right < root_left))
        {
            survivor = root_right;
            dying = root_left;
        }

        intervals.emplace_back(birth[dying], death_time);
        parent[dying] = survivor;
        birth[survivor] = std::min(birth[survivor], birth[dying]);
    }
};

inline Size sectionWidth(const SectionMap &sections)
{
    Size width = 0;
    for (const auto &entry : sections)
    {
        width = std::max(width, entry.second.size());
    }
    return std::max<Size>(width, 1);
}

inline std::vector<double> paddedSection(const SectionMap &sections, Index vertex, Size width)
{
    std::vector<double> out(width, 0.0);
    const auto it = sections.find(vertex);
    if (it == sections.end())
    {
        return out;
    }
    const Size copy_width = std::min(width, it->second.size());
    std::copy_n(it->second.begin(), copy_width, out.begin());
    return out;
}

inline std::vector<double> sectionDifference(const SectionMap &sections, Index from, Index to,
                                             Size width)
{
    auto left = paddedSection(sections, from, width);
    auto right = paddedSection(sections, to, width);
    for (Size i = 0; i < width; ++i)
    {
        right[i] -= left[i];
    }
    return right;
}

inline void validateVertex(const Graph *graph, Index vertex)
{
    if (vertex < 0 || (graph != nullptr && vertex >= static_cast<Index>(graph->numVertices())))
    {
        throw std::out_of_range("Vertex index out of range");
    }
}

inline void validateFiniteValues(const std::vector<double> &values, const char *field_name)
{
    if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument(std::string(field_name) + " values must be finite");
    }
}

} // namespace nerve::graphs::detail
