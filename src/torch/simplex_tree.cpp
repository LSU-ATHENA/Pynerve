#include "nerve/torch/boundary_matrix.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

namespace nerve::torch
{

namespace
{

void validate_filtration_value(double filtration, const char *name)
{
    TORCH_CHECK(std::isfinite(filtration), name, " must be finite");
}

void validate_max_radius(double max_radius)
{
    TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius),
                "max_radius must be positive and not NaN");
}

void validate_max_dim(int64_t max_dim, int64_t max_supported)
{
    TORCH_CHECK(max_dim >= 0 && max_dim <= max_supported, "max_dim must be between 0 and ",
                max_supported);
}

void validate_simplex_vertices(const std::vector<int64_t> &vertices)
{
    int64_t previous = -1;
    for (int64_t vertex : vertices)
    {
        TORCH_CHECK(vertex >= 0, "simplex vertices must be non-negative");
        TORCH_CHECK(vertex > previous, "simplex vertices must be strictly increasing");
        previous = vertex;
    }
}

at::Tensor validate_point_tensor_cpu(const at::Tensor &points, const char *name)
{
    TORCH_CHECK(points.defined(), name, " must be defined");
    TORCH_CHECK(points.dim() == 2, name, " must be a 2D tensor [N, D]");
    TORCH_CHECK(points.size(0) > 0, name, " must contain at least one point");
    TORCH_CHECK(points.size(1) > 0, name, " must contain at least one coordinate dimension");
    TORCH_CHECK(points.is_floating_point(), name, " must be a floating-point tensor");
    TORCH_CHECK(at::isfinite(points).all().item<bool>(), name,
                " must contain only finite coordinates");
    return points.contiguous().cpu().to(at::kDouble);
}

} // namespace

SimplexTree::SimplexTree()
{
    // Initialize with capacity for root
    vertex_indices_ = at::full({capacity_}, -1, at::TensorOptions().dtype(at::kLong));
    parent_pointers_ = at::full({capacity_}, -1, at::TensorOptions().dtype(at::kLong));
    first_child_ = at::full({capacity_}, -1, at::TensorOptions().dtype(at::kLong));
    next_sibling_ = at::full({capacity_}, -1, at::TensorOptions().dtype(at::kLong));
    filtration_values_ = at::zeros({capacity_}, at::TensorOptions().dtype(at::kDouble));

    // Root node (index 0) - represents empty simplex
    vertex_indices_[0] = -1;
    parent_pointers_[0] = -1;
    filtration_values_[0] = 0.0;
}

SimplexTree::SimplexTree(const at::Tensor &points, double max_radius)
    : SimplexTree()
{
    TORCH_CHECK(max_radius >= 0.0 && !std::isnan(max_radius),
                "max_radius must be non-negative and not NaN");
    if (max_radius > 0.0)
    {
        TORCH_CHECK(points.defined() && points.dim() == 2, "points must be a 2D tensor [N, D]");
        build_vr(points, max_radius, std::min<int64_t>(points.size(1) + 1, 3));
    }
}

void SimplexTree::ensure_capacity(int64_t min_capacity)
{
    if (min_capacity <= capacity_)
        return;

    while (capacity_ < min_capacity)
    {
        capacity_ *= GROWTH_FACTOR;
    }

    vertex_indices_ =
        at::cat({vertex_indices_,
                 at::full({capacity_ - vertex_indices_.size(0)}, -1,
                          at::TensorOptions().dtype(at::kLong).device(vertex_indices_.device()))});
    parent_pointers_ =
        at::cat({parent_pointers_,
                 at::full({capacity_ - parent_pointers_.size(0)}, -1,
                          at::TensorOptions().dtype(at::kLong).device(parent_pointers_.device()))});
    first_child_ =
        at::cat({first_child_,
                 at::full({capacity_ - first_child_.size(0)}, -1,
                          at::TensorOptions().dtype(at::kLong).device(first_child_.device()))});
    next_sibling_ =
        at::cat({next_sibling_,
                 at::full({capacity_ - next_sibling_.size(0)}, -1,
                          at::TensorOptions().dtype(at::kLong).device(next_sibling_.device()))});
    filtration_values_ = at::cat(
        {filtration_values_,
         at::zeros({capacity_ - filtration_values_.size(0)},
                   at::TensorOptions().dtype(at::kDouble).device(filtration_values_.device()))});
}

int64_t SimplexTree::allocate_node(int64_t vertex, int64_t parent, double filtration)
{
    ensure_capacity(next_free_idx_ + 1);

    int64_t idx = next_free_idx_++;
    vertex_indices_[idx] = vertex;
    parent_pointers_[idx] = parent;
    filtration_values_[idx] = filtration;
    first_child_[idx] = -1;
    next_sibling_[idx] = -1;

    return idx;
}

void SimplexTree::insert(const std::vector<int64_t> &vertices, double filtration)
{
    if (vertices.empty())
        return;
    validate_simplex_vertices(vertices);
    validate_filtration_value(filtration, "filtration");

    int64_t current = root_idx_;

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        int64_t v = vertices[i];
        int64_t child = first_child_[current].item<int64_t>();

        // Find child with this vertex
        int64_t found = -1;
        while (child != -1)
        {
            if (vertex_indices_[child].item<int64_t>() == v)
            {
                found = child;
                break;
            }
            child = next_sibling_[child].item<int64_t>();
        }

        if (found == -1)
        {
            // Create new child
            found = allocate_node(v, current, filtration);

            // Insert as first child
            int64_t old_first = first_child_[current].item<int64_t>();
            first_child_[current] = found;
            next_sibling_[found] = old_first;
        }

        current = found;

        // Update dimension
        int64_t dim = static_cast<int64_t>(i);
        if (dim > max_dimension_)
        {
            max_dimension_ = dim;
        }
    }

    // Update filtration at leaf
    filtration_values_[current] = filtration;
}

void SimplexTree::insert_batch(const at::Tensor &simplices, const at::Tensor &filtration_values)
{
    TORCH_CHECK(simplices.defined(), "simplices must be defined");
    TORCH_CHECK(filtration_values.defined(), "filtration_values must be defined");
    TORCH_CHECK(simplices.dim() == 2, "simplices must be a 2D tensor");
    TORCH_CHECK(simplices.scalar_type() == at::kLong, "simplices must use torch.long dtype");
    TORCH_CHECK(filtration_values.dim() == 1, "filtration_values must be a 1D tensor");
    TORCH_CHECK(filtration_values.is_floating_point(),
                "filtration_values must be a floating-point tensor");
    TORCH_CHECK(filtration_values.size(0) == simplices.size(0),
                "filtration_values length must match simplices rows");
    // Convert to CPU for processing
    auto simp_cpu = simplices.contiguous().cpu();
    auto filt_cpu = filtration_values.contiguous().cpu().to(at::kDouble);
    TORCH_CHECK(at::isfinite(filt_cpu).all().item<bool>(),
                "filtration_values must contain only finite values");

    auto simp_accessor = simp_cpu.accessor<int64_t, 2>();
    auto filt_accessor = filt_cpu.accessor<double, 1>();

    int64_t num_simplices = simp_cpu.size(0);

    for (int64_t i = 0; i < num_simplices; ++i)
    {
        std::vector<int64_t> vertices;
        for (int64_t j = 0; j < simp_cpu.size(1); ++j)
        {
            int64_t v = simp_accessor[i][j];
            TORCH_CHECK(v >= -1, "simplices may only use -1 as padding");
            if (v >= 0)
                vertices.push_back(v);
        }
        validate_simplex_vertices(vertices);
        insert(vertices, filt_accessor[i]);
    }
}

bool SimplexTree::contains(const std::vector<int64_t> &vertices) const
{
    return find(vertices) != -1;
}

int64_t SimplexTree::find(const std::vector<int64_t> &vertices) const
{
    if (vertices.empty())
        return root_idx_;

    int64_t current = root_idx_;

    for (int64_t v : vertices)
    {
        int64_t child = first_child_[current].item<int64_t>();

        int64_t found = -1;
        while (child != -1)
        {
            if (vertex_indices_[child].item<int64_t>() == v)
            {
                found = child;
                break;
            }
            child = next_sibling_[child].item<int64_t>();
        }

        if (found == -1)
            return -1;
        current = found;
    }

    return current;
}

std::vector<int64_t> SimplexTree::get_vertices(int64_t node_idx) const
{
    TORCH_CHECK(node_idx >= 0 && node_idx < next_free_idx_, "node_idx out of bounds");
    std::vector<int64_t> vertices;

    int64_t current = node_idx;
    while (current != root_idx_ && current != -1)
    {
        vertices.push_back(vertex_indices_[current].item<int64_t>());
        current = parent_pointers_[current].item<int64_t>();
    }

    std::reverse(vertices.begin(), vertices.end());
    return vertices;
}

void SimplexTree::set_filtration(const std::vector<int64_t> &vertices, double value)
{
    if (!vertices.empty())
    {
        validate_simplex_vertices(vertices);
    }
    validate_filtration_value(value, "filtration");
    int64_t node = find(vertices);
    if (node != -1)
    {
        filtration_values_[node] = value;
    }
}

double SimplexTree::get_filtration(const std::vector<int64_t> &vertices) const
{
    int64_t node = find(vertices);
    if (node != -1)
    {
        return filtration_values_[node].item<double>();
    }
    return std::numeric_limits<double>::infinity();
}

void SimplexTree::to(at::Device device)
{
    vertex_indices_ = vertex_indices_.to(device);
    parent_pointers_ = parent_pointers_.to(device);
    first_child_ = first_child_.to(device);
    next_sibling_ = next_sibling_.to(device);
    filtration_values_ = filtration_values_.to(device);
}

at::Tensor SimplexTree::to_boundary_matrix(int64_t dim) const
{
    TORCH_CHECK(dim >= 0, "dimension must be non-negative");
    BoundaryMatrix boundary(*this, dim, BoundaryMatrix::Format::CSC);
    return boundary.to_dense();
}

std::pair<at::Tensor, at::Tensor> SimplexTree::get_sorted_simplices() const
{
    // Return simplices and their filtration values sorted by filtration
    auto values = filtration_values_.slice(0, 0, next_free_idx_);
    auto sorted = at::sort(values);

    return {std::get<0>(sorted), std::get<1>(sorted)};
}

at::Tensor SimplexTree::to_tensor() const
{
    const int64_t row_count = std::max<int64_t>(0, num_simplices());
    at::Tensor simplices =
        at::full({row_count, max_dimension_ + 1}, -1, at::TensorOptions().dtype(at::kLong));

    auto simplices_acc = simplices.accessor<int64_t, 2>();
    int64_t row = 0;
    for (int64_t dim = 0; dim <= max_dimension_; ++dim)
    {
        auto dim_simplices = get_simplices_by_dimension(dim);
        for (const auto &simplex : dim_simplices)
        {
            if (row >= row_count)
            {
                break;
            }
            for (size_t i = 0; i < simplex.size() && i < static_cast<size_t>(max_dimension_ + 1);
                 ++i)
            {
                simplices_acc[row][static_cast<int64_t>(i)] = simplex[i];
            }
            ++row;
        }
    }

    if (row < row_count)
    {
        return simplices.narrow(0, 0, row).clone();
    }
    return simplices;
}

} // namespace nerve::torch
