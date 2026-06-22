#pragma once

/// @file lazy_witness_complex.hpp

#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

/// Constructs a witness complex from landmarks using lazy evaluation.
/// The 1-skeleton is built explicitly, higher simplices are found via clique
/// enumeration.
class LazyWitnessComplex
{
public:
    LazyWitnessComplex(const std::vector<double> &all_points, size_t point_dim,
                       const std::vector<size_t> &landmarks, size_t max_dim, double max_radius);

    /// @brief Build the witness complex
    void buildComplex(algebra::SimplicialComplex &complex);

private:
    struct SimplexKeyHash
    {
        std::size_t operator()(const std::vector<size_t> &vertices) const noexcept
        {
            std::size_t seed = vertices.size();
            for (size_t vertex : vertices)
            {
                seed ^= std::hash<size_t>{}(vertex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    bool hasValidInput() const;
    double landmarkDistance(size_t i, size_t j) const;
    void buildHigherSimplices(const std::vector<std::vector<size_t>> &landmark_neighbors,
                              algebra::SimplicialComplex &complex);
    void expandCliques(std::vector<size_t> &current, std::vector<size_t> &candidates,
                       size_t target_size, const std::vector<std::vector<size_t>> &neighbors,
                       algebra::SimplicialComplex &complex,
                       std::unordered_set<std::vector<size_t>, SimplexKeyHash> &seen);

    const std::vector<double> all_points_;
    size_t point_dim_;
    const std::vector<size_t> landmarks_;
    size_t max_dim_;
    double max_radius_;
};

} // namespace nerve::persistence
