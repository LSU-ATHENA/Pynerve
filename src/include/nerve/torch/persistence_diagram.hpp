#pragma once
#if __has_include(<torch/torch.h>)

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace nerve::torch
{

/**
 * @brief Persistence diagram using ATen tensor storage.
 *
 * Stores persistence pairs [birth, death] with associated metadata:
 * - diagram_: [num_pairs, 2] - [birth, death] values
 * - dimensions_: [num_pairs] - homology dimension for each pair
 * - birth_idx_: [num_pairs] - index of birth simplex in filtration
 * - death_idx_: [num_pairs] - index of death simplex (-1 for infinite)
 *
 * Fully differentiable when created through autograd-enabled operations.
 */
class PersistenceDiagram
{
private:
    // Core data
    at::Tensor diagram_;    // [num_pairs, 2] - [birth, death]
    at::Tensor dimensions_; // [num_pairs] - int64, dimension

    // Filtration indices (for mapping back)
    at::Tensor birth_idx_; // [num_pairs] - int64, index in filtration
    at::Tensor death_idx_; // [num_pairs] - int64, index (-1 for inf)

    // Optional: birth/death simplex vertices for reference
    at::Tensor birth_simplices_; // [num_pairs, max_dim] - vertex indices
    at::Tensor death_simplices_; // [num_pairs, max_dim] - vertex indices

    // Metadata
    int64_t max_dimension_ = 0;

public:
    PersistenceDiagram();

    /// Main constructor from computed diagram
    PersistenceDiagram(at::Tensor diagram, at::Tensor dimensions, at::Tensor birth_idx,
                       at::Tensor death_idx);

    /// Create empty diagram with capacity
    explicit PersistenceDiagram(int64_t capacity);

    // Copy/Move
    PersistenceDiagram(const PersistenceDiagram &other) = default;
    PersistenceDiagram(PersistenceDiagram &&other) noexcept = default;
    PersistenceDiagram &operator=(const PersistenceDiagram &other) = default;
    PersistenceDiagram &operator=(PersistenceDiagram &&other) noexcept = default;

    /// Get full diagram tensor [N, 2]
    [[nodiscard]] at::Tensor diagram() const { return diagram_; }

    /// Get birth times
    [[nodiscard]] at::Tensor births() const { return diagram_.select(1, 0); }

    /// Get death times
    [[nodiscard]] at::Tensor deaths() const { return diagram_.select(1, 1); }

    /// Get dimensions
    [[nodiscard]] at::Tensor dimensions() const { return dimensions_; }

    /// Get persistence (death - birth)
    [[nodiscard]] at::Tensor persistence_lengths() const;

    /// Get diagram for specific dimension
    [[nodiscard]] PersistenceDiagram get_dimension(int64_t dim) const;

    /// Get finite points only (death != inf)
    [[nodiscard]] PersistenceDiagram get_finite_points() const;

    /// Get infinite points only (birth but no death)
    [[nodiscard]] PersistenceDiagram get_infinite_points() const;

    /// Number of persistence pairs
    [[nodiscard]] int64_t num_pairs() const { return diagram_.size(0); }

    [[nodiscard]] int64_t max_dimension() const { return max_dimension_; }

    // Statistics

    /// Total persistence (sum of persistence lengths)
    [[nodiscard]] at::Tensor total_persistence() const;

    /// Sum of squared persistences
    [[nodiscard]] at::Tensor persistence_variance() const;

    /// Mean persistence per dimension
    [[nodiscard]] at::Tensor mean_persistence_by_dimension() const;

    /// Betti number at threshold (count of features alive at time)
    [[nodiscard]] int64_t betti_number(double threshold, int64_t dim = -1) const;

    /// Euler characteristic at threshold
    [[nodiscard]] double euler_characteristic(double threshold) const;

    /// Persistence entropy
    [[nodiscard]] double persistence_entropy() const;

    /// Persistence landscape norm
    [[nodiscard]] double landscape_norm(int64_t k) const;

    // ML Representations (for downstream tasks)

    /// Convert to persistence image [resolution, resolution]
    [[nodiscard]] at::Tensor to_persistence_image(int64_t resolution = 64, double sigma = 0.1,
                                                  double birth_min = 0.0,
                                                  double birth_max = -1.0, // auto
                                                  double death_min = 0.0,
                                                  double death_max = -1.0 // auto
    ) const;

    /// Convert to persistence landscape [k, num_samples]
    [[nodiscard]] at::Tensor to_persistence_landscape(int64_t k = 5, int64_t num_samples = 100,
                                                      double min_val = 0.0,
                                                      double max_val = -1.0 // auto
    ) const;

    /// Convert to Betti curve [num_samples]
    [[nodiscard]] at::Tensor to_betti_curve(int64_t num_samples = 100, double min_val = 0.0,
                                            double max_val = -1.0) const;

    /// Vectorized representation for ML
    [[nodiscard]] at::Tensor to_vector(int64_t bins_per_dim = 10) const;

    // Distances (for diagram comparison)

    /// Wasserstein distance to another diagram
    [[nodiscard]] double wasserstein_distance(const PersistenceDiagram &other,
                                              double p = 2.0) const;

    /// Bottleneck distance to another diagram
    [[nodiscard]] double bottleneck_distance(const PersistenceDiagram &other) const;

    /// Kernel methods
    [[nodiscard]] double persistence_kernel(const PersistenceDiagram &other,
                                            double sigma = 1.0) const;

    // Operations

    /// Concatenate with another diagram
    void append(const PersistenceDiagram &other);

    /// Filter by persistence threshold
    [[nodiscard]] PersistenceDiagram threshold(double min_persistence) const;

    /// Filter by dimension
    [[nodiscard]] PersistenceDiagram filter_by_dimension(int64_t dim) const;

    /// Sort by persistence (descending)
    void sort_by_persistence();

    /// Normalize birth/death to [0, 1]
    void normalize();

    void to(at::Device device);
    [[nodiscard]] at::Device device() const { return diagram_.device(); }
    [[nodiscard]] bool is_cuda() const { return diagram_.is_cuda(); }

    /// Stack diagrams (for batch processing)
    static PersistenceDiagram batch(const std::vector<PersistenceDiagram> &diagrams);

    /// Get item from batch
    [[nodiscard]] PersistenceDiagram get_batch_item(int64_t idx) const;

    /// Check if this is a batched diagram
    [[nodiscard]] bool is_batched() const;

    /// Batch size (1 if not batched)
    [[nodiscard]] int64_t batch_size() const;

    // I/O (for serialization)

    /// Export to dictionary of tensors
    [[nodiscard]] std::vector<std::pair<std::string, at::Tensor>> state_dict() const;

    /// Load from dictionary
    void load_state_dict(const std::vector<std::pair<std::string, at::Tensor>> &state_dict);

    // Differentiability Support

    /// Check if diagram supports gradients
    [[nodiscard]] bool requires_grad() const { return diagram_.requires_grad(); }

    /// Enable/disable gradients
    void set_requires_grad(bool requires_grad = true);

    /// Backward pass (if diagram is part of computational graph)
    void backward() const;

    /// Get gradient w.r.t. birth times (if available)
    [[nodiscard]] at::Tensor births_grad() const;

    /// Get gradient w.r.t. death times (if available)
    [[nodiscard]] at::Tensor deaths_grad() const;
};

} // namespace nerve::torch

#endif
