#include "nerve/torch/filtration.hpp"
#include "nerve/torch/simplex_tree.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve::torch
{

Filtration::Filtration()
    : num_simplices_(0)
    , max_dimension_(0)
    , max_vertices_per_simplex_(0)
    , capacity_(1024)
{
    simplices_ = at::full({capacity_, 1}, -1, at::TensorOptions().dtype(at::kLong));
    values_ = at::zeros({capacity_}, at::TensorOptions().dtype(at::kDouble));
    dimensions_ = at::zeros({capacity_}, at::TensorOptions().dtype(at::kLong));
    sorted_order_ = at::arange(capacity_, at::TensorOptions().dtype(at::kLong));
}

Filtration::Filtration(const at::Tensor &simplices, const at::Tensor &values,
                       const at::Tensor &dimensions)
    : simplices_(simplices)
    , values_(values)
    , dimensions_(dimensions)
    , sorted_order_(at::arange(simplices.size(0), at::TensorOptions().dtype(at::kLong)))
    , is_sorted_(false)
{
    num_simplices_ = simplices.size(0);
    max_vertices_per_simplex_ = simplices.size(1);

    if (dimensions_.numel() > 0)
    {
        max_dimension_ = at::max(dimensions_).item<int64_t>();
    }

    capacity_ = num_simplices_;
}

Filtration::Filtration(const SimplexTree &tree)
    : Filtration()
{
    const int64_t max_dim = tree.max_dimension();
    for (int64_t dim = 0; dim <= max_dim; ++dim)
    {
        auto simplices = tree.get_simplices_by_dimension(dim);
        for (const auto &simplex : simplices)
        {
            append(simplex, tree.get_filtration(simplex));
        }
    }
    sort_by_filtration();
}

void Filtration::ensure_capacity(int64_t min_capacity)
{
    if (min_capacity <= capacity_)
        return;

    while (capacity_ < min_capacity)
    {
        capacity_ *= 2;
    }

    // Resize tensors
    simplices_ = at::cat(
        {simplices_, at::full({capacity_ - simplices_.size(0), max_vertices_per_simplex_}, -1,
                              at::TensorOptions().dtype(at::kLong).device(simplices_.device()))},
        0);
    values_ = at::cat(
        {values_, at::zeros({capacity_ - values_.size(0)},
                            at::TensorOptions().dtype(at::kDouble).device(values_.device()))});
    dimensions_ =
        at::cat({dimensions_,
                 at::zeros({capacity_ - dimensions_.size(0)},
                           at::TensorOptions().dtype(at::kLong).device(dimensions_.device()))});
    sorted_order_ =
        at::arange(capacity_, at::TensorOptions().dtype(at::kLong).device(sorted_order_.device()));
}

void Filtration::update_max_dimension()
{
    if (dimensions_.numel() > 0)
    {
        auto used_dims = dimensions_.slice(0, 0, num_simplices_);
        max_dimension_ = at::max(used_dims).item<int64_t>();
    }
}

void Filtration::append(const std::vector<int64_t> &vertices, double value)
{
    if (vertices.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    {
        throw std::length_error("simplex vertex count exceeds int64_t range");
    }
    const int64_t vertex_count = static_cast<int64_t>(vertices.size());
    int64_t dim = vertex_count - 1;

    // Ensure proper simplex width
    if (vertex_count > max_vertices_per_simplex_)
    {
        // Resize simplices tensor to accommodate wider simplices
        int64_t new_width = vertex_count;
        at::Tensor new_simplices =
            at::full({capacity_, new_width}, -1,
                     at::TensorOptions().dtype(at::kLong).device(simplices_.device()));

        if (max_vertices_per_simplex_ > 0)
        {
            new_simplices.slice(1, 0, max_vertices_per_simplex_).copy_(simplices_);
        }
        simplices_ = new_simplices;
        max_vertices_per_simplex_ = new_width;
    }

    ensure_capacity(num_simplices_ + 1);

    // Fill vertex indices
    for (int64_t i = 0; i < vertex_count; ++i)
    {
        simplices_[num_simplices_][i] = vertices[static_cast<size_t>(i)];
    }

    values_[num_simplices_] = value;
    dimensions_[num_simplices_] = dim;

    num_simplices_++;

    if (dim > max_dimension_)
    {
        max_dimension_ = dim;
    }

    is_sorted_ = false;
}

void Filtration::append_batch(const at::Tensor &vertices_list, const at::Tensor &values)
{
    auto v_cpu = vertices_list.cpu();
    auto vals_cpu = values.cpu();

    auto v_accessor = v_cpu.accessor<int64_t, 2>();
    auto vals_accessor = vals_cpu.accessor<double, 1>();

    int64_t num_new = v_cpu.size(0);

    for (int64_t i = 0; i < num_new; ++i)
    {
        std::vector<int64_t> vertices;
        for (int64_t j = 0; j < v_cpu.size(1); ++j)
        {
            int64_t v = v_accessor[i][j];
            if (v >= 0)
                vertices.push_back(v);
        }
        append(vertices, vals_accessor[i]);
    }
}

void Filtration::append_tensor(const at::Tensor &simplex, double value)
{
    auto cpu_simplex = simplex.cpu();
    std::vector<int64_t> vertices;

    for (int64_t i = 0; i < cpu_simplex.numel(); ++i)
    {
        int64_t v = cpu_simplex[i].item<int64_t>();
        if (v >= 0)
            vertices.push_back(v);
    }

    append(vertices, value);
}

void Filtration::sort_by_filtration()
{
    if (is_sorted_)
        return;

    auto used_values = values_.slice(0, 0, num_simplices_);
    auto sorted = at::sort(used_values);

    sorted_order_ = std::get<1>(sorted);
    is_sorted_ = true;
}

void Filtration::sort_by_dimension_and_filtration()
{
    // Sort first by dimension, then by filtration within each dimension
    auto used_dims = dimensions_.slice(0, 0, num_simplices_);
    auto used_vals = values_.slice(0, 0, num_simplices_);

    // Create combined sort key: (dimension, value)
    auto keys = at::stack({used_dims, used_vals}, 1);
    auto sorted = at::argsort(keys.select(1, 0) * 1e10 + keys.select(1, 1));

    sorted_order_ = sorted;
    is_sorted_ = true;
}

at::Tensor Filtration::get_sorted_order() const
{
    return sorted_order_.slice(0, 0, num_simplices_);
}

at::Tensor Filtration::get_simplices_in_dimension(int64_t dim) const
{
    auto mask = dimensions_.slice(0, 0, num_simplices_) == dim;
    auto indices = at::nonzero(mask).squeeze();

    if (indices.numel() == 0)
    {
        return at::empty({0, max_vertices_per_simplex_}, at::TensorOptions().dtype(at::kLong));
    }

    return simplices_.index_select(0, indices).slice(1, 0, dim + 1);
}

at::Tensor Filtration::get_values_in_dimension(int64_t dim) const
{
    auto mask = dimensions_.slice(0, 0, num_simplices_) == dim;
    auto indices = at::nonzero(mask).squeeze();

    if (indices.numel() == 0)
    {
        return at::empty({0}, at::TensorOptions().dtype(at::kDouble));
    }

    return values_.index_select(0, indices);
}

at::Tensor Filtration::get_simplex_counts() const
{
    at::Tensor counts = at::zeros({max_dimension_ + 1}, at::TensorOptions().dtype(at::kLong));

    for (int64_t i = 0; i < num_simplices_; ++i)
    {
        int64_t dim = dimensions_[i].item<int64_t>();
        counts[dim] += 1;
    }

    return counts;
}

double Filtration::get_value(int64_t idx) const
{
    TORCH_CHECK(idx >= 0 && idx < num_simplices_, "Index out of bounds");
    return values_[idx].item<double>();
}

at::Tensor Filtration::get_simplex(int64_t idx) const
{
    TORCH_CHECK(idx >= 0 && idx < num_simplices_, "Index out of bounds");
    int64_t dim = dimensions_[idx].item<int64_t>();
    return simplices_[idx].slice(0, 0, dim + 1);
}

int64_t Filtration::get_dimension(int64_t idx) const
{
    TORCH_CHECK(idx >= 0 && idx < num_simplices_, "Index out of bounds");
    return dimensions_[idx].item<int64_t>();
}

Filtration Filtration::get_sublevel_set(double max_value) const
{
    auto mask = values_.slice(0, 0, num_simplices_) <= max_value;
    auto indices = at::nonzero(mask).squeeze();

    if (indices.numel() == 0)
    {
        return Filtration();
    }

    Filtration result;
    result.simplices_ = simplices_.index_select(0, indices);
    result.values_ = values_.index_select(0, indices);
    result.dimensions_ = dimensions_.index_select(0, indices);
    result.num_simplices_ = indices.numel();
    result.max_dimension_ = at::max(result.dimensions_).item<int64_t>();
    result.max_vertices_per_simplex_ = max_vertices_per_simplex_;
    result.capacity_ = result.num_simplices_;

    return result;
}

Filtration Filtration::get_dimension_filtration(int64_t dim) const
{
    auto mask = dimensions_.slice(0, 0, num_simplices_) == dim;
    auto indices = at::nonzero(mask).squeeze();

    if (indices.numel() == 0)
    {
        return Filtration();
    }

    Filtration result;
    result.simplices_ = simplices_.index_select(0, indices);
    result.values_ = values_.index_select(0, indices);
    result.dimensions_ = dimensions_.index_select(0, indices);
    result.num_simplices_ = indices.numel();
    result.max_dimension_ = dim;
    result.max_vertices_per_simplex_ = max_vertices_per_simplex_;
    result.capacity_ = result.num_simplices_;

    return result;
}

void Filtration::to(at::Device device)
{
    simplices_ = simplices_.to(device);
    values_ = values_.to(device);
    dimensions_ = dimensions_.to(device);
    sorted_order_ = sorted_order_.to(device);
}

std::vector<Filtration> Filtration::batch_split(int64_t batch_size) const
{
    std::vector<Filtration> batches;
    int64_t num_batches = (num_simplices_ + batch_size - 1) / batch_size;

    for (int64_t i = 0; i < num_batches; ++i)
    {
        int64_t start = i * batch_size;
        int64_t end = std::min(start + batch_size, num_simplices_);

        Filtration batch;
        batch.simplices_ = simplices_.slice(0, start, end);
        batch.values_ = values_.slice(0, start, end);
        batch.dimensions_ = dimensions_.slice(0, start, end);
        batch.num_simplices_ = end - start;
        batch.max_dimension_ = max_dimension_;
        batch.max_vertices_per_simplex_ = max_vertices_per_simplex_;
        batch.capacity_ = batch.num_simplices_;

        batches.push_back(batch);
    }

    return batches;
}

Filtration Filtration::batch_concat(const std::vector<Filtration> &filtrations)
{
    if (filtrations.empty())
    {
        return Filtration();
    }

    std::vector<at::Tensor> simplices_list;
    std::vector<at::Tensor> values_list;
    std::vector<at::Tensor> dimensions_list;

    for (const auto &f : filtrations)
    {
        simplices_list.push_back(f.simplices().slice(0, 0, f.num_simplices_));
        values_list.push_back(f.values().slice(0, 0, f.num_simplices_));
        dimensions_list.push_back(f.dimensions().slice(0, 0, f.num_simplices_));
    }

    Filtration result;
    result.simplices_ = at::cat(simplices_list, 0);
    result.values_ = at::cat(values_list, 0);
    result.dimensions_ = at::cat(dimensions_list, 0);
    result.num_simplices_ = result.simplices_.size(0);
    result.max_vertices_per_simplex_ = result.simplices_.size(1);
    result.update_max_dimension();
    result.capacity_ = result.num_simplices_;

    return result;
}

at::Tensor Filtration::to_batched_tensor() const
{
    return simplices_.slice(0, 0, num_simplices_);
}

double Filtration::total_filtration_range() const
{
    if (num_simplices_ == 0)
        return 0.0;
    auto used_vals = values_.slice(0, 0, num_simplices_);
    double min_val = at::min(used_vals).item<double>();
    double max_val = at::max(used_vals).item<double>();
    return max_val - min_val;
}

double Filtration::mean_filtration() const
{
    if (num_simplices_ == 0)
        return 0.0;
    return at::mean(values_.slice(0, 0, num_simplices_)).item<double>();
}

at::Tensor Filtration::histogram(int64_t num_bins) const
{
    auto used_vals = values_.slice(0, 0, num_simplices_);
    double min_val = at::min(used_vals).item<double>();
    double max_val = at::max(used_vals).item<double>();

    return at::histc(used_vals, num_bins, min_val, max_val);
}

#include "detail/filtration_factory_ops.inl"

} // namespace nerve::torch
