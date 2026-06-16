#pragma once

#if __has_include(<torch/torch.h>)
#include <torch/torch.h>
#endif

#include <cstdint>
#include <vector>

namespace nerve::torch
{

// Forward declaration
class SimplexTree;

/**
 * @brief Filtered simplicial complex using ATen tensor storage.
 *
 * Stores simplices and their filtration values as ATen tensors.
 * Supports sorting, batching, and VR complex construction.
 *
 * Storage:
 * - simplices_: [num_simplices, max_dim+1] int64 - vertex indices (-1 padded)
 * - values_: [num_simplices] double - filtration value
 * - dimensions_: [num_simplices] int64 - simplex dimension
 * - sorted_order_: [num_simplices] int64 - indices in filtration order
 */
class Filtration
{
private:
    // Core storage
    at::Tensor simplices_;  // [num_simplices, max_dim+1] - vertex indices
    at::Tensor values_;     // [num_simplices] - filtration values
    at::Tensor dimensions_; // [num_simplices] - dimension of each simplex

    // Sorting state
    at::Tensor sorted_order_; // [num_simplices] - sorted indices
    bool is_sorted_ = false;

    // Metadata
    int64_t num_simplices_ = 0;
    int64_t max_dimension_ = 0;
    int64_t max_vertices_per_simplex_ = 0; // Current max_dim + 1

    // Capacity for dynamic insertion
    int64_t capacity_ = 1024;
    static constexpr int64_t GROWTH_FACTOR = 2;

    void ensure_capacity(int64_t min_capacity);
    void update_max_dimension();

public:
    Filtration();

    /// Build from simplex tree
    explicit Filtration(const SimplexTree &tree);

    /// Build from explicit tensors
    Filtration(const at::Tensor &simplices, const at::Tensor &values, const at::Tensor &dimensions);

    // Copy/Move
    Filtration(const Filtration &other) = default;
    Filtration(Filtration &&other) noexcept = default;
    Filtration &operator=(const Filtration &other) = default;
    Filtration &operator=(Filtration &&other) noexcept = default;

    // Static Factory Methods (VR Construction)

    /// Build Vietoris-Rips filtration from point cloud
    static Filtration from_vietoris_rips(const at::Tensor &points, double max_radius,
                                         int64_t max_dim, at::Device device = at::kCPU);

    /// Build witness complex filtration
    static Filtration from_witness(const at::Tensor &points, const at::Tensor &landmarks,
                                   double max_radius, int64_t max_dim,
                                   at::Device device = at::kCPU);

    /// Build alpha complex filtration
    static Filtration from_alpha(const at::Tensor &points, at::Device device = at::kCPU);

    // Insertion

    /// Add a single simplex
    void append(const std::vector<int64_t> &vertices, double value);

    /// Add a batch of simplices
    void append_batch(const at::Tensor &vertices_list, const at::Tensor &values);

    /// Add a simplex from tensor
    void append_tensor(const at::Tensor &simplex, double value);

    // Sorting (for persistence algorithms)

    /// Sort by filtration value (ascending)
    void sort_by_filtration();

    /// Sort by dimension then filtration
    void sort_by_dimension_and_filtration();

    /// Get indices in sorted order
    [[nodiscard]] at::Tensor get_sorted_order() const;

    /// Check if sorted
    [[nodiscard]] bool is_sorted() const { return is_sorted_; }

    // Queries

    /// Get simplices in a specific dimension
    [[nodiscard]] at::Tensor get_simplices_in_dimension(int64_t dim) const;

    /// Get filtration values for simplices in a dimension
    [[nodiscard]] at::Tensor get_values_in_dimension(int64_t dim) const;

    /// Get count of simplices in each dimension
    [[nodiscard]] at::Tensor get_simplex_counts() const;

    /// Get all simplices as tensor
    [[nodiscard]] at::Tensor simplices() const { return simplices_.slice(0, 0, num_simplices_); }

    /// Get all filtration values
    [[nodiscard]] at::Tensor values() const { return values_.slice(0, 0, num_simplices_); }

    /// Get dimensions
    [[nodiscard]] at::Tensor dimensions() const { return dimensions_.slice(0, 0, num_simplices_); }

    /// Get filtration value of specific simplex by index
    [[nodiscard]] double get_value(int64_t idx) const;

    /// Get simplex vertices by index
    [[nodiscard]] at::Tensor get_simplex(int64_t idx) const;

    /// Get dimension of simplex
    [[nodiscard]] int64_t get_dimension(int64_t idx) const;

    // Subset Operations

    /// Get sub-filtration up to a filtration value
    [[nodiscard]] Filtration get_sublevel_set(double max_value) const;

    /// Get filtration for specific dimension only
    [[nodiscard]] Filtration get_dimension_filtration(int64_t dim) const;

    void to(at::Device device);
    [[nodiscard]] at::Device device() const { return simplices_.device(); }
    [[nodiscard]] bool is_cuda() const { return simplices_.is_cuda(); }

    // Batching Support

    /// Split into batch_size chunks
    [[nodiscard]] std::vector<Filtration> batch_split(int64_t batch_size) const;

    /// Concatenate multiple filtrations
    static Filtration batch_concat(const std::vector<Filtration> &filtrations);

    /// Create batched tensor [batch, max_simplices, max_dim+1]
    [[nodiscard]] at::Tensor to_batched_tensor() const;

    /// Get number of simplices
    [[nodiscard]] int64_t num_simplices() const { return num_simplices_; }

    /// Get max dimension
    [[nodiscard]] int64_t max_dimension() const { return max_dimension_; }

    /// Get max vertices per simplex
    [[nodiscard]] int64_t max_vertices_per_simplex() const { return max_vertices_per_simplex_; }

    // Statistics

    /// Get total persistence (sum of lifetimes)
    [[nodiscard]] double total_filtration_range() const;

    /// Get mean filtration value
    [[nodiscard]] double mean_filtration() const;

    /// Get filtration histogram
    [[nodiscard]] at::Tensor histogram(int64_t num_bins) const;
};

} // namespace nerve::torch
