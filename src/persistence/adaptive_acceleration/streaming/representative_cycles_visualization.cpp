
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"

#include <utility>

namespace nerve::persistence::adaptive_acceleration::representative
{

class CycleVisualizer::Impl
{
public:
    explicit Impl(const RepresentativeConfig &config)
        : config(config)
    {}
    RepresentativeConfig config;
};

errors::ErrorResult<std::unique_ptr<CycleVisualizer>>
CycleVisualizer::create(const RepresentativeConfig &config)
{
    std::unique_ptr<CycleVisualizer> visualizer(new CycleVisualizer(config));
    return errors::ErrorResult<std::unique_ptr<CycleVisualizer>>::success(std::move(visualizer));
}

errors::ErrorResult<CycleVisualizationData> CycleVisualizer::generateVisualizationData(
    const Cycle &cycle, const core::BufferView<const double> &points, std::size_t point_dim)
{
    if (point_dim == 0 || points.empty() || points.size() % point_dim != 0 || !cycle.isValid())
    {
        return errors::ErrorResult<CycleVisualizationData>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    CycleVisualizationData data;
    data.dimension = cycle.dimension;
    data.persistence = cycle.persistence();
    data.visualization_format = "polyline";
    const std::size_t point_count = points.size() / point_dim;
    for (int vertex : cycle.vertices)
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= point_count)
        {
            continue;
        }
        std::vector<double> coords(point_dim, 0.0);
        const std::size_t offset = static_cast<std::size_t>(vertex) * point_dim;
        for (std::size_t dim = 0; dim < point_dim; ++dim)
        {
            coords[dim] = points[offset + dim];
        }
        data.vertices.push_back(std::move(coords));
    }
    for (std::size_t i = 1; i < data.vertices.size(); ++i)
    {
        data.edges.push_back({static_cast<int>(i - 1), static_cast<int>(i)});
        data.edge_weights.push_back(1.0);
    }
    if (!data.isValid())
    {
        return errors::ErrorResult<CycleVisualizationData>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return errors::ErrorResult<CycleVisualizationData>::success(std::move(data));
}

errors::ErrorResult<std::vector<CycleVisualizationData>>
CycleVisualizer::generateVisualizationData(const std::vector<Cycle> &cycles,
                                           const core::BufferView<const double> &points,
                                           std::size_t point_dim)
{
    std::vector<CycleVisualizationData> all;
    all.reserve(cycles.size());
    for (const Cycle &cycle : cycles)
    {
        auto one = generateVisualizationData(cycle, points, point_dim);
        if (one.isError())
        {
            continue;
        }
        all.push_back(one.value());
    }
    return errors::ErrorResult<std::vector<CycleVisualizationData>>::success(std::move(all));
}

CycleVisualizer::CycleVisualizer(const RepresentativeConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{}

CycleVisualizer::~CycleVisualizer() = default;

errors::ErrorResult<bool> CycleValidator::validateCycle(const Cycle &cycle,
                                                        const SparseMatrix &boundary_matrix)
{
    return errors::ErrorResult<bool>::success(checkCycleCondition(cycle, boundary_matrix) &&
                                              checkBoundaryCondition(cycle, boundary_matrix));
}

errors::ErrorResult<std::vector<bool>>
CycleValidator::validateCycles(const std::vector<Cycle> &cycles,
                               const SparseMatrix &boundary_matrix)
{
    std::vector<bool> results;
    results.reserve(cycles.size());
    for (const Cycle &cycle : cycles)
    {
        results.push_back(checkCycleCondition(cycle, boundary_matrix));
    }
    return errors::ErrorResult<std::vector<bool>>::success(std::move(results));
}

bool CycleValidator::isBoundaryCycle(const Cycle &cycle, const SparseMatrix &boundary_matrix)
{
    return checkBoundaryCondition(cycle, boundary_matrix);
}

bool CycleValidator::isCycle(const Cycle &cycle, const SparseMatrix &boundary_matrix)
{
    return checkCycleCondition(cycle, boundary_matrix);
}

bool CycleValidator::checkBoundaryCondition(const Cycle &cycle, const SparseMatrix &boundary_matrix)
{
    return cycle.isValid() && !boundary_matrix.rowIndices().empty();
}

bool CycleValidator::checkCycleCondition(const Cycle &cycle, const SparseMatrix &boundary_matrix)
{
    if (!cycle.isValid())
    {
        return false;
    }
    for (int vertex : cycle.vertices)
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= boundary_matrix.numRows())
        {
            return false;
        }
    }
    return true;
}

} // namespace nerve::persistence::adaptive_acceleration::representative
