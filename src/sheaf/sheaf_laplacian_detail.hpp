#pragma once

#include "nerve/sheaf/sheaf_laplacian.hpp"
#include "nerve/types.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
namespace nerve::sheaf::detail
{

inline std::vector<uint32_t>
sortedNodeIds(const std::unordered_map<uint32_t, SheafLaplacianRuntime::SheafNode> &nodes)
{
    std::vector<uint32_t> ids;
    ids.reserve(nodes.size());
    for (const auto &[node_id, _] : nodes)
    {
        ids.push_back(node_id);
    }
    std::ranges::sort(ids);
    return ids;
}

inline std::unordered_map<uint32_t, Size> buildNodeIndexMap(const std::vector<uint32_t> &node_ids)
{
    std::unordered_map<uint32_t, Size> index;
    index.reserve(node_ids.size());
    for (Size i = 0; i < node_ids.size(); ++i)
    {
        index[node_ids[i]] = i;
    }
    return index;
}

inline bool containsName(const std::vector<std::string> &names, const std::string &name)
{
    return std::ranges::find(names, name) != names.end();
}

inline bool isGeneratedAttributeNames(const std::vector<std::string> &names)
{
    if (names.empty())
    {
        return false;
    }
    for (Size i = 0; i < names.size(); ++i)
    {
        if (names[i] != "attr_" + std::to_string(i))
        {
            return false;
        }
    }
    return true;
}

inline bool isFiniteVector(const std::vector<double> &values)
{
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

inline Size attributeIndex(const SheafLaplacianRuntime::SheafNode &node, const std::string &name)
{
    const auto it = std::ranges::find(node.attribute_names, name);
    if (it == node.attribute_names.end())
    {
        return node.attributes.size();
    }
    const auto index = static_cast<Size>(std::distance(node.attribute_names.begin(), it));
    return index < node.attributes.size() ? index : node.attributes.size();
}

inline bool getNodeAttributeValue(const SheafLaplacianRuntime::SheafNode &node,
                                  const std::string &name, double *value)
{
    const Size index = attributeIndex(node, name);
    if (index == node.attributes.size())
    {
        return false;
    }
    *value = node.attributes[index];
    return std::isfinite(*value);
}

inline void setNodeAttributeValue(SheafLaplacianRuntime::SheafNode &node, const std::string &name,
                                  double value)
{
    const Size index = attributeIndex(node, name);
    if (index == node.attributes.size())
    {
        node.attribute_names.push_back(name);
        node.attributes.push_back(value);
        return;
    }
    node.attributes[index] = value;
}

inline void eraseNodeAttribute(SheafLaplacianRuntime::SheafNode &node, const std::string &name)
{
    const Size index = attributeIndex(node, name);
    if (index == node.attributes.size())
    {
        return;
    }
    node.attribute_names.erase(node.attribute_names.begin() + static_cast<std::ptrdiff_t>(index));
    node.attributes.erase(node.attributes.begin() + static_cast<std::ptrdiff_t>(index));
}

inline std::vector<std::string>
collectAttributeNames(const std::vector<std::string> &configured,
                      const std::unordered_map<uint32_t, SheafLaplacianRuntime::SheafNode> &nodes)
{
    std::vector<std::string> names = configured;
    for (uint32_t node_id : sortedNodeIds(nodes))
    {
        for (const auto &name : nodes.at(node_id).attribute_names)
        {
            if (!containsName(names, name))
            {
                names.push_back(name);
            }
        }
    }
    return names;
}

inline void normalizeValues(std::vector<double> &values, double tolerance)
{
    if (values.empty())
    {
        return;
    }
    const auto [min_it, max_it] = std::ranges::minmax_element(values);
    const double range = *max_it - *min_it;
    if (range <= tolerance)
    {
        std::ranges::fill(values, 0.0);
        return;
    }
    for (double &value : values)
    {
        value = (value - *min_it) / range;
    }
}

} // namespace nerve::sheaf::detail
#endif
