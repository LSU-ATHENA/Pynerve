#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace nerve::torch
{

/**
 * @brief Simplex tree implementation using ATen tensor storage.
 *
 * Stores a simplicial complex as a tree structure with ATen tensors as the
 * underlying storage. This provides zero-copy interop with PyTorch, GPU
 * support, and compatibility with ML pipelines.
 *
 * Tree Structure:
 * - vertex_indices_: stores vertex labels for each node
 * - parent_pointers_: parent node index (-1 for roots)
 * - first_child_: index of first child (-1 if leaf)
 * - next_sibling_: index of next sibling (-1 if none)
 * - filtration_values_: filtration time for each simplex
 *
 * Example: Simplex [0,1,2] is stored as path root->0->1->2
 */
class SimplexTree
{
private:
    // Core storage (ATen tensors)
    at::Tensor vertex_indices_;    // [N] int64 - vertex label at each node
    at::Tensor parent_pointers_;   // [N] int64 - parent index (-1 for roots)
    at::Tensor first_child_;       // [N] int64 - first child index (-1 if leaf)
    at::Tensor next_sibling_;      // [N] int64 - next sibling index (-1 if none)
    at::Tensor filtration_values_; // [N] double - filtration value per node

    // Metadata
    int64_t root_idx_ = 0;
    int64_t max_dimension_ = 0;
    int64_t next_free_idx_ = 1; // Next available node index

    // Capacity management
    int64_t capacity_ = 1024;
    static constexpr int64_t GROWTH_FACTOR = 2;

    void ensure_capacity(int64_t min_capacity);
    int64_t allocate_node(int64_t vertex, int64_t parent, double filtration);

public:
    // Construction / Destruction

    SimplexTree();
    explicit SimplexTree(const at::Tensor &points, double max_radius = 0.0);

    // Copy/Move
    SimplexTree(const SimplexTree &other) = default;
    SimplexTree(SimplexTree &&other) noexcept = default;
    SimplexTree &operator=(const SimplexTree &other) = default;
    SimplexTree &operator=(SimplexTree &&other) noexcept = default;

    // Insertion / Modification

    /// Insert a simplex with given filtration value
    void insert(const std::vector<int64_t> &vertices, double filtration = 0.0);

    /// Insert a batch of simplices
    void insert_batch(const at::Tensor &simplices, const at::Tensor &filtration_values);

    /// Remove a simplex and its cofaces
    void remove(const std::vector<int64_t> &vertices);

    /// Set filtration value for a simplex
    void set_filtration(const std::vector<int64_t> &vertices, double value);

    // Queries

    /// Check if simplex exists in the tree
    [[nodiscard]] bool contains(const std::vector<int64_t> &vertices) const;

    /// Find node index for a simplex (-1 if not found)
    [[nodiscard]] int64_t find(const std::vector<int64_t> &vertices) const;

    /// Get vertices of a simplex from node index
    [[nodiscard]] std::vector<int64_t> get_vertices(int64_t node_idx) const;

    /// Get all cofaces of a simplex up to max_dim
    [[nodiscard]] std::vector<std::vector<int64_t>>
    get_cofaces(const std::vector<int64_t> &vertices, int64_t max_dim = -1) const;

    /// Get all faces of a simplex
    [[nodiscard]] std::vector<std::vector<int64_t>>
    get_faces(const std::vector<int64_t> &vertices) const;

    /// Get simplices in a specific dimension
    [[nodiscard]] std::vector<std::vector<int64_t>> get_simplices_by_dimension(int64_t dim) const;

    /// Get filtration value of a simplex
    [[nodiscard]] double get_filtration(const std::vector<int64_t> &vertices) const;

    // VR Complex Construction

    /// Build Vietoris-Rips complex from point cloud
    void build_vr(const at::Tensor &points, double max_radius, int64_t max_dim);

    /// Build witness complex
    void build_witness(const at::Tensor &points, const at::Tensor &landmarks, double max_radius,
                       int64_t max_dim);

    [[nodiscard]] int64_t num_simplices() const { return next_free_idx_ - 1; }
    [[nodiscard]] int64_t max_dimension() const { return max_dimension_; }

    /// Get all filtration values as tensor
    [[nodiscard]] at::Tensor filtration_values() const
    {
        return filtration_values_.slice(0, 0, next_free_idx_);
    }

    /// Get root index
    [[nodiscard]] int64_t root_idx() const { return root_idx_; }

    // PyTorch / Device Support

    /// Move all tensors to device
    void to(at::Device device);

    /// Get current device
    [[nodiscard]] at::Device device() const { return vertex_indices_.device(); }

    /// Check if on CUDA
    [[nodiscard]] bool is_cuda() const { return vertex_indices_.is_cuda(); }

    // Conversion

    /// Convert to boundary matrix for given dimension
    [[nodiscard]] at::Tensor to_boundary_matrix(int64_t dim) const;

    /// Get filtration-sorted simplex list
    [[nodiscard]] std::pair<at::Tensor, at::Tensor> get_sorted_simplices() const;

    /// Export to dense tensor representation [num_simplices, max_dim+1]
    [[nodiscard]] at::Tensor to_tensor() const;

    // Internal Access (for advanced users)

    [[nodiscard]] const at::Tensor &vertex_indices() const { return vertex_indices_; }
    [[nodiscard]] const at::Tensor &parent_pointers() const { return parent_pointers_; }
    [[nodiscard]] const at::Tensor &first_child() const { return first_child_; }
    [[nodiscard]] const at::Tensor &next_sibling() const { return next_sibling_; }
};

} // namespace nerve::torch
