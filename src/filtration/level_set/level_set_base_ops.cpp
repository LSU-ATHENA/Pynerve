
#include "nerve/algebra/simplex.hpp"
#include "nerve/filtration/level_set.hpp"
#ifdef USE_TORCH
#include <torch/torch.h>
#endif
#include "nerve/common.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <ranges>
#include <vector>
namespace nerve::filtration
{
LevelSet::LevelSet(const std::vector<Size> &grid_shape)
{
    setGridShape(grid_shape);
}
void LevelSet::setGridShape(const std::vector<Size> &shape)
{
    if (std::ranges::any_of(shape, [](Size dim) { return dim == 0; }))
    {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    grid_shape_ = shape;
}
void LevelSet::setFiltrationType(const std::string &type)
{
    if (type != "sublevel" && type != "superlevel" && type != "average")
    {
        throw std::invalid_argument("Unknown level-set filtration type");
    }
    filtration_type_ = type;
}
void LevelSet::setNumLevels(Size num_levels)
{
    if (num_levels == 0)
    {
        throw std::invalid_argument("Number of filtration levels must be positive");
    }
    num_levels_ = num_levels;
}
Size LevelSet::getNumLevels() const
{
    return num_levels_;
}
void LevelSet::setAdaptiveLevels(bool adaptive)
{
    adaptive_levels_ = adaptive;
}

errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
LevelSet::buildFiltration(core::BufferView<const double> scalar_field,
                          const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E30_DET_MISMATCH);
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());

    try
    {
        validateScalarField(fieldVector);
    }
    catch (const std::exception &)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    filtration_.clear();
    if (adaptive_levels_)
    {
        levels_ = computeAdaptiveLevels(scalar_field);
    }
    else
    {
        levels_ = computeLevels(scalar_field);
    }
    buildGridConnectivity();
    buildVertexSimplices(fieldVector);
    buildEdgeSimplices(fieldVector);
    if (grid_shape_.size() >= 2)
    {
        buildTriangleSimplices(fieldVector);
    }
    if (grid_shape_.size() >= 3)
    {
        buildTetrahedronSimplices(fieldVector);
    }
    sortFiltration();

    auto end = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end - start).count();
    return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::success(
        std::vector<std::pair<algebra::Simplex, double>>(filtration_));
}
#ifdef USE_TORCH
errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
LevelSet::buildFiltration(const at::Tensor &scalar_field, const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            "Cannot satisfy determinism contract");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> fieldVector;
    auto numel = scalar_field.numel();
    if (numel > 0)
    {
        fieldVector.resize(numel);
        at::Tensor t_cpu = scalar_field.contiguous().cpu();
        const double *tensor_data = t_cpu.data_ptr<double>();
        if (tensor_data)
        {
            std::copy(tensor_data, tensor_data + numel, fieldVector.begin());
        }
    }

    validateScalarField(fieldVector);
    filtration_.clear();
    if (adaptive_levels_)
    {
        core::BufferView<const double> fieldView(fieldVector.data(), fieldVector.size());
        levels_ = computeAdaptiveLevels(fieldView);
    }
    else
    {
        core::BufferView<const double> fieldView(fieldVector.data(), fieldVector.size());
        levels_ = computeLevels(fieldView);
    }
    buildGridConnectivity();
    buildVertexSimplices(fieldVector);
    buildEdgeSimplices(fieldVector);
    if (grid_shape_.size() >= 2)
    {
        buildTriangleSimplices(fieldVector);
    }
    if (grid_shape_.size() >= 3)
    {
        buildTetrahedronSimplices(fieldVector);
    }
    sortFiltration();

    auto end = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end - start).count();
    return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::success(
        std::vector<std::pair<algebra::Simplex, double>>(filtration_));
}
#endif // USE_TORCH
errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>
LevelSet::buildFiltration(core::BufferView<const double> scalar_field,
                          const std::vector<std::vector<Index>> &connectivity,
                          const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E30_DET_MISMATCH);
    }
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    std::vector<std::vector<Index>> connectivity_vector = connectivity;

    try
    {
        validateScalarField(fieldVector);
    }
    catch (const std::exception &)
    {
        return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::error(
            errors::ErrorCode::E88_INVALID_SIMPLICES);
    }
    filtration_.clear();
    buildMeshConnectivity(connectivity_vector);
    buildVertexSimplices(fieldVector);
    for (const auto &edge : connectivity_vector)
    {
        if (edge.size() == 2)
        {
            algebra::Simplex edgeSimplex(edge);
            double filtration_value = assignFiltrationValue(edgeSimplex, fieldVector);
            filtration_.emplace_back(edgeSimplex, filtration_value);
        }
    }
    sortFiltration();
    return errors::ErrorResult<std::vector<std::pair<algebra::Simplex, double>>>::success(
        std::vector<std::pair<algebra::Simplex, double>>(filtration_));
}
std::vector<std::pair<algebra::Simplex, double>>
LevelSet::build2dFiltration(core::BufferView<const double> scalar_field, Size width, Size height,
                            const core::DeterminismContract &contract)
{
    setGridShape({width, height});
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    core::BufferView<const double> fieldView(fieldVector.data(), fieldVector.size());
    auto result = buildFiltration(fieldView, contract);
    return result.isSuccess() ? result.value() : std::vector<std::pair<algebra::Simplex, double>>();
}
std::vector<std::pair<algebra::Simplex, double>>
LevelSet::build3dFiltration(core::BufferView<const double> scalar_field, Size width, Size height,
                            Size depth, const core::DeterminismContract &contract)
{
    setGridShape({width, height, depth});
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    core::BufferView<const double> fieldView(fieldVector.data(), fieldVector.size());
    auto result = buildFiltration(fieldView, contract);
    return result.isSuccess() ? result.value() : std::vector<std::pair<algebra::Simplex, double>>();
}
std::vector<double> LevelSet::computeLevels(core::BufferView<const double> scalar_field) const
{
    if (scalar_field.size() == 0)
        return {};
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    double min_val = *std::min_element(fieldVector.begin(), fieldVector.end());
    double max_val = *std::max_element(fieldVector.begin(), fieldVector.end());
    std::vector<double> levels;
    if (num_levels_ == 1)
    {
        return {min_val};
    }
    levels.reserve(num_levels_);
    for (Size i = 0; i < num_levels_; ++i)
    {
        double level = min_val + (max_val - min_val) * static_cast<double>(i) /
                                     static_cast<double>(num_levels_ - 1);
        levels.push_back(level);
    }
    return levels;
}
std::vector<double>
LevelSet::computeAdaptiveLevels(core::BufferView<const double> scalar_field) const
{
    if (scalar_field.size() == 0)
        return {};
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    std::vector<double> sorted_field = fieldVector;
    std::ranges::sort(sorted_field);
    std::vector<double> levels;
    levels.reserve(num_levels_);
    for (Size i = 0; i < num_levels_; ++i)
    {
        Size idx = num_levels_ == 1 ? 0 : i * (sorted_field.size() - 1) / (num_levels_ - 1);
        levels.push_back(sorted_field[idx]);
    }
    return levels;
}
std::vector<double> LevelSet::computeQuantileLevels(core::BufferView<const double> scalar_field,
                                                    Size num_quantiles) const
{
    if (scalar_field.size() == 0)
        return {};
    if (num_quantiles == 0)
        return {};
    std::vector<double> fieldVector(scalar_field.data(), scalar_field.data() + scalar_field.size());
    std::vector<double> sorted_field = fieldVector;
    std::ranges::sort(sorted_field);
    std::vector<double> quantiles;
    quantiles.reserve(num_quantiles + 1);
    for (Size i = 0; i <= num_quantiles; ++i)
    {
        Size idx = i * sorted_field.size() / num_quantiles;
        if (idx >= sorted_field.size())
            idx = sorted_field.size() - 1;
        quantiles.push_back(sorted_field[idx]);
    }
    return quantiles;
}
Size LevelSet::getNumSimplices() const
{
    return filtration_.size();
}
Size LevelSet::getNumSimplicesOfDimension(Size dim) const
{
    Size count = 0;
    for (const auto &[simplex, value] : filtration_)
    {
        if (simplex.dimension() == dim)
        {
            count++;
        }
    }
    return count;
}
std::vector<double> LevelSet::getFiltrationValues() const
{
    std::vector<double> values;
    values.reserve(filtration_.size());
    for (const auto &[simplex, value] : filtration_)
    {
        values.push_back(value);
    }
    return values;
}
std::map<Size, std::vector<algebra::Simplex>> LevelSet::getSimplicesByLevel() const
{
    std::map<Size, std::vector<algebra::Simplex>> result;
    for (const auto &[simplex, value] : filtration_)
    {
        Size level_index = 0;
        for (Size i = 0; i < levels_.size(); ++i)
        {
            if (value <= levels_[i])
            {
                level_index = i;
                break;
            }
        }
        result[level_index].push_back(simplex);
    }
    return result;
}
} // namespace nerve::filtration
