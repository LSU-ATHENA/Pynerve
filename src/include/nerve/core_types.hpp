// Core type compatibility layer.
// This header forwards foundational types from `nerve/types.hpp`.

#pragma once

// Include types.hpp first to get Index, ConstSpan, etc.
#include "nerve/types.hpp"

#include <compare>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace nerve
{

// Field and Dimension defined in types.hpp - inherited via include

// Error codes - comprehensive set matching errors::ErrorCode
enum class ErrorCode : int
{
    Success = 0,
    SUCCESS = 0,
    InvalidArgument = -1,
    OutOfMemory = -2,
    GpuError = -3,
    NotImplemented = -4,
    Unknown = -99,
    // PH4/PH5/PH6 specific error codes
    E54_PH4_INVALID_INPUT = 0x00000604,
    E55_PH4_SPARSE_CONVERGENCE_FAIL = 0x00000605,
    E56_PH4_WITNESS_SAMPLING_ERROR = 0x00000606,
    E11_PH5_OVERFLOW = 0x00000607,
    E12_PH6_OVERFLOW = 0x00000608,
    E13_PH_HIGHDIM_PRECISION = 0x00000609,
    E41_RESOURCE_LIMIT = 0x00000501
};

// Forward declaration of persistence namespace
namespace persistence
{

// Persistence pair (birth, death)
struct PersistencePair
{
    double birth;
    double death;
    int dimension;
    PersistencePair(double b = 0.0, double d = 0.0, int dim = 0)
        : birth(b)
        , death(d)
        , dimension(dim)
    {}
};

// Persistence diagram - a collection of persistence pairs
struct PersistenceDiagram
{
    std::vector<PersistencePair> pairs;

    void addPair(double birth, double death, int dimension = 0)
    {
        pairs.emplace_back(birth, death, dimension);
    }

    [[nodiscard]] size_t size() const { return pairs.size(); }
    [[nodiscard]] bool empty() const { return pairs.empty(); }
    void clear() { pairs.clear(); }
};

// Point with coordinates for geometric operations
struct GeometricPoint
{
    std::vector<double> coords;

    GeometricPoint() = default;
    explicit GeometricPoint(const std::vector<double> &coordinates)
        : coords(coordinates)
    {}
    explicit GeometricPoint(std::vector<double> &&coordinates)
        : coords(std::move(coordinates))
    {}
};

// Configuration for simplification
struct SimplificationConfig
{
    double radius = 1.0;
    double edge_threshold = 0.5;
    bool use_simplification = true;
    int max_iterations = 100;
};

// Result of simplification
struct SimplificationResult
{
    std::vector<GeometricPoint> simplified_points;
    size_t original_points = 0;
    double simplification_ratio = 0.0;
    double estimated_speedup = 1.0;
    bool applied = false;
};

// Simplification benefit estimate
struct SimplificationEstimate
{
    double vertex_reduction = 0.0;
    double edge_reduction = 0.0;
    double simplex_reduction = 0.0;
    double estimated_speedup = 1.0;
    bool recommended = false;
};

namespace simplify
{

[[nodiscard]] SimplificationResult simplifyPointCloud(const std::vector<GeometricPoint> &points,
                                                      const SimplificationConfig &config);

[[nodiscard]] SimplificationEstimate estimateSimplificationBenefit(size_t num_points,
                                                                   size_t num_edges);

} // namespace simplify

} // namespace persistence

} // namespace nerve
