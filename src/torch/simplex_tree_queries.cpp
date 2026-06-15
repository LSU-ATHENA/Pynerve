#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace nerve::torch
{

std::vector<std::vector<int64_t>> SimplexTree::get_cofaces(const std::vector<int64_t> &vertices,
                                                           int64_t max_dim) const
{
    const int64_t node = find(vertices);
    if (node < 0)
    {
        return {};
    }

    const int64_t base_dim = static_cast<int64_t>(vertices.size()) - 1;
    const int64_t max_output_dim = (max_dim < 0) ? max_dimension_ : max_dim;
    if (max_output_dim < base_dim)
    {
        return {};
    }

    std::vector<std::vector<int64_t>> cofaces;
    std::vector<std::pair<int64_t, std::vector<int64_t>>> stack;
    stack.emplace_back(node, vertices);

    while (!stack.empty())
    {
        auto [current, current_vertices] = std::move(stack.back());
        stack.pop_back();

        const int64_t current_dim = static_cast<int64_t>(current_vertices.size()) - 1;
        if (!current_vertices.empty() && current_dim <= max_output_dim)
        {
            cofaces.push_back(current_vertices);
        }
        if (current_dim >= max_output_dim)
        {
            continue;
        }

        for (int64_t child = first_child_[current].item<int64_t>(); child != -1;
             child = next_sibling_[child].item<int64_t>())
        {
            const int64_t child_vertex = vertex_indices_[child].item<int64_t>();
            if (child_vertex < 0)
            {
                continue;
            }
            auto next_vertices = current_vertices;
            next_vertices.push_back(child_vertex);
            stack.emplace_back(child, std::move(next_vertices));
        }
    }

    std::sort(cofaces.begin(), cofaces.end());
    return cofaces;
}

std::vector<std::vector<int64_t>> SimplexTree::get_faces(const std::vector<int64_t> &vertices) const
{
    if (vertices.empty())
    {
        return {};
    }

    std::vector<std::vector<int64_t>> faces;
    std::vector<int64_t> current;
    current.reserve(vertices.size());

    std::function<void(size_t)> enumerate = [&](size_t idx) {
        if (idx == vertices.size())
        {
            if (!current.empty() && contains(current))
            {
                faces.push_back(current);
            }
            return;
        }

        current.push_back(vertices[idx]);
        enumerate(idx + 1);
        current.pop_back();

        enumerate(idx + 1);
    };
    enumerate(0);

    std::sort(faces.begin(), faces.end(),
              [](const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs) {
                  if (lhs.size() != rhs.size())
                  {
                      return lhs.size() < rhs.size();
                  }
                  return lhs < rhs;
              });
    return faces;
}

std::vector<std::vector<int64_t>> SimplexTree::get_simplices_by_dimension(int64_t dim) const
{
    if (dim < 0)
    {
        return {};
    }

    std::vector<std::vector<int64_t>> simplices;
    std::vector<std::pair<int64_t, std::vector<int64_t>>> stack;
    stack.reserve(static_cast<size_t>(std::max<int64_t>(1, num_simplices())));

    for (int64_t child = first_child_[root_idx_].item<int64_t>(); child != -1;
         child = next_sibling_[child].item<int64_t>())
    {
        const int64_t vertex = vertex_indices_[child].item<int64_t>();
        if (vertex < 0)
        {
            continue;
        }
        stack.emplace_back(child, std::vector<int64_t>{vertex});
    }

    while (!stack.empty())
    {
        auto [current, current_vertices] = std::move(stack.back());
        stack.pop_back();

        const int64_t current_dim = static_cast<int64_t>(current_vertices.size()) - 1;
        if (current_dim == dim)
        {
            simplices.push_back(std::move(current_vertices));
            continue;
        }
        if (current_dim > dim)
        {
            continue;
        }

        for (int64_t child = first_child_[current].item<int64_t>(); child != -1;
             child = next_sibling_[child].item<int64_t>())
        {
            const int64_t child_vertex = vertex_indices_[child].item<int64_t>();
            if (child_vertex < 0)
            {
                continue;
            }
            auto next_vertices = current_vertices;
            next_vertices.push_back(child_vertex);
            stack.emplace_back(child, std::move(next_vertices));
        }
    }

    std::sort(simplices.begin(), simplices.end());
    return simplices;
}

void SimplexTree::remove(const std::vector<int64_t> &vertices)
{
    if (vertices.empty())
    {
        return;
    }

    const int64_t node = find(vertices);
    if (node <= root_idx_)
    {
        return;
    }

    const int64_t parent = parent_pointers_[node].item<int64_t>();
    if (parent < 0)
    {
        return;
    }

    int64_t first = first_child_[parent].item<int64_t>();
    if (first == node)
    {
        first_child_[parent] = next_sibling_[node];
    }
    else
    {
        int64_t prev = first;
        while (prev != -1)
        {
            const int64_t sibling = next_sibling_[prev].item<int64_t>();
            if (sibling == node)
            {
                next_sibling_[prev] = next_sibling_[node];
                break;
            }
            prev = sibling;
        }
    }

    std::vector<int64_t> to_detach;
    to_detach.push_back(node);
    for (size_t i = 0; i < to_detach.size(); ++i)
    {
        const int64_t current = to_detach[i];
        for (int64_t child = first_child_[current].item<int64_t>(); child != -1;
             child = next_sibling_[child].item<int64_t>())
        {
            to_detach.push_back(child);
        }
    }

    for (const int64_t idx : to_detach)
    {
        parent_pointers_[idx] = -1;
        first_child_[idx] = -1;
        next_sibling_[idx] = -1;
        vertex_indices_[idx] = -1;
        filtration_values_[idx] = std::numeric_limits<double>::infinity();
    }

    int64_t updated_max_dim = 0;
    std::vector<std::pair<int64_t, int64_t>> traversal;
    traversal.emplace_back(root_idx_, -1);
    while (!traversal.empty())
    {
        const auto [current, parent_dim] = traversal.back();
        traversal.pop_back();
        for (int64_t child = first_child_[current].item<int64_t>(); child != -1;
             child = next_sibling_[child].item<int64_t>())
        {
            const int64_t vertex = vertex_indices_[child].item<int64_t>();
            if (vertex < 0)
            {
                continue;
            }
            const int64_t child_dim = parent_dim + 1;
            updated_max_dim = std::max(updated_max_dim, child_dim);
            traversal.emplace_back(child, child_dim);
        }
    }
    max_dimension_ = updated_max_dim;
}

} // namespace nerve::torch
