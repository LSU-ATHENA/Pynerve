#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::kernels
{

struct ExactDimensionResult
{
    std::vector<Pair> pairs;
    Size simplex_count = 0;
    Size reduction_operations = 0;
};

inline void validateSimplicialInput(const std::vector<std::vector<int>> &simplices,
                                    const std::vector<double> &filtration_values,
                                    const std::vector<int> &dimensions)
{
    if (simplices.size() != filtration_values.size() || simplices.size() != dimensions.size())
    {
        throw std::invalid_argument("simplicial input arrays must have equal size");
    }

    for (std::size_t i = 0; i < simplices.size(); ++i)
    {
        if (dimensions[i] < 0)
        {
            throw std::invalid_argument("simplex dimension must be non-negative");
        }
        if (!std::isfinite(filtration_values[i]))
        {
            throw std::invalid_argument("filtration values must be finite");
        }
        for (int vertex : simplices[i])
        {
            if (vertex < 0)
            {
                throw std::invalid_argument("simplex vertices must be non-negative");
            }
        }

        auto canonical = simplices[i];
        std::sort(canonical.begin(), canonical.end());
        const auto unique_end = std::unique(canonical.begin(), canonical.end());
        if (unique_end != canonical.end())
        {
            throw std::invalid_argument("simplex vertices must be unique");
        }

        const auto expected_vertices = static_cast<std::size_t>(dimensions[i]) + 1;
        if (simplices[i].size() != expected_vertices)
        {
            throw std::invalid_argument("simplex size does not match its dimension");
        }
    }
}

inline algebra::Simplex makeSimplex(const std::vector<int> &vertices)
{
    std::vector<Index> converted;
    converted.reserve(vertices.size());
    for (int vertex : vertices)
    {
        converted.push_back(static_cast<Index>(vertex));
    }
    return algebra::Simplex(converted);
}

inline void
insertClosure(const algebra::Simplex &simplex, double filtration,
              std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> *values)
{
    auto [it, inserted] = values->emplace(simplex, filtration);
    if (!inserted && filtration >= it->second)
    {
        return;
    }
    if (!inserted)
    {
        it->second = filtration;
    }
    if (simplex.numVertices() <= 1)
    {
        return;
    }

    for (const auto &face : simplex.faces(::nerve::core::DeterminismContract{}))
    {
        insertClosure(face, filtration, values);
    }
}

inline Size countSimplicesOfDimension(const std::vector<int> &dimensions,
                                      Dimension target_dimension)
{
    return static_cast<Size>(std::count(dimensions.begin(), dimensions.end(), target_dimension));
}

inline algebra::SimplicialComplex buildClosedComplex(const std::vector<std::vector<int>> &simplices,
                                                     const std::vector<double> &filtration_values,
                                                     const std::vector<int> &dimensions,
                                                     Dimension max_input_dimension)
{
    validateSimplicialInput(simplices, filtration_values, dimensions);

    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> values;
    values.reserve(simplices.size() * 2);
    for (std::size_t i = 0; i < simplices.size(); ++i)
    {
        if (dimensions[i] > max_input_dimension)
        {
            continue;
        }
        insertClosure(makeSimplex(simplices[i]), filtration_values[i], &values);
    }

    std::vector<std::pair<algebra::Simplex, double>> ordered;
    ordered.reserve(values.size());
    for (const auto &entry : values)
    {
        ordered.push_back(entry);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.second != rhs.second)
        {
            return lhs.second < rhs.second;
        }
        if (lhs.first.dimension() != rhs.first.dimension())
        {
            return lhs.first.dimension() < rhs.first.dimension();
        }
        return lhs.first < rhs.first;
    });

    algebra::SimplicialComplex complex;
    for (const auto &[simplex, filtration] : ordered)
    {
        complex.addSimplexWithFiltration(simplex, filtration);
    }
    return complex;
}

inline ExactDimensionResult
computeExactDimensionPairs(const std::vector<std::vector<int>> &simplices,
                           const std::vector<double> &filtration_values,
                           const std::vector<int> &dimensions, Dimension target_dimension)
{
    if (target_dimension < 0)
    {
        throw std::invalid_argument("target dimension must be non-negative");
    }

    ExactDimensionResult result;
    result.simplex_count = countSimplicesOfDimension(dimensions, target_dimension);

    const auto complex =
        buildClosedComplex(simplices, filtration_values, dimensions, target_dimension + 1);
    const auto exact = computeExactPersistenceZ2(complex, static_cast<Size>(target_dimension));

    result.reduction_operations = exact.reduction_operations;
    for (const auto &pair : exact.pairs)
    {
        if (pair.dimension == target_dimension)
        {
            result.pairs.push_back(pair);
        }
    }
    std::sort(result.pairs.begin(), result.pairs.end(), [](const Pair &lhs, const Pair &rhs) {
        if (lhs.birth != rhs.birth)
        {
            return lhs.birth < rhs.birth;
        }
        if (lhs.death != rhs.death)
        {
            return lhs.death < rhs.death;
        }
        return lhs.birth_index < rhs.birth_index;
    });
    return result;
}

} // namespace nerve::persistence::kernels
