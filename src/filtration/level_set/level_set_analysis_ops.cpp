
#include "nerve/filtration/level_set.hpp"

#include <cmath>
#include <vector>

namespace nerve::filtration
{
std::vector<Index> LevelSet::findCriticalPoints(const std::vector<double> &scalar_field) const
{
    std::vector<Index> critical_points;
    for (Size i = 0; i < scalar_field.size(); ++i)
    {
        std::vector<Index> neighbors = getGridNeighbors(static_cast<Index>(i));
        bool is_minimum = true;
        bool is_maximum = true;
        for (Index neighbor : neighbors)
        {
            if (neighbor < static_cast<Index>(scalar_field.size()))
            {
                if (scalar_field[i] > scalar_field[neighbor])
                {
                    is_maximum = false;
                }
                if (scalar_field[i] < scalar_field[neighbor])
                {
                    is_minimum = false;
                }
            }
        }
        if (is_minimum || is_maximum)
        {
            critical_points.push_back(static_cast<Index>(i));
        }
    }
    return critical_points;
}
std::vector<Index> LevelSet::findMinima(const std::vector<double> &scalar_field) const
{
    std::vector<Index> minima;
    for (Size i = 0; i < scalar_field.size(); ++i)
    {
        std::vector<Index> neighbors = getGridNeighbors(static_cast<Index>(i));
        bool is_minimum = true;
        for (Index neighbor : neighbors)
        {
            if (static_cast<size_t>(neighbor) < scalar_field.size() &&
                scalar_field[i] >= scalar_field[neighbor])
            {
                is_minimum = false;
                break;
            }
        }
        if (is_minimum)
        {
            minima.push_back(static_cast<Index>(i));
        }
    }
    return minima;
}
std::vector<Index> LevelSet::findMaxima(const std::vector<double> &scalar_field) const
{
    std::vector<Index> maxima;
    for (Size i = 0; i < scalar_field.size(); ++i)
    {
        std::vector<Index> neighbors = getGridNeighbors(static_cast<Index>(i));
        bool is_maximum = true;
        for (Index neighbor : neighbors)
        {
            if (static_cast<size_t>(neighbor) < scalar_field.size() &&
                scalar_field[i] <= scalar_field[neighbor])
            {
                is_maximum = false;
                break;
            }
        }
        if (is_maximum)
        {
            maxima.push_back(static_cast<Index>(i));
        }
    }
    return maxima;
}
std::vector<Index> LevelSet::findSaddles(const std::vector<double> &scalar_field) const
{
    std::vector<Index> saddles;
    if (grid_shape_.size() != 2)
        return saddles;
    Size width = grid_shape_[0];
    Size height = grid_shape_[1];
    for (Size y = 1; y < height - 1; ++y)
    {
        for (Size x = 1; x < width - 1; ++x)
        {
            Index idx = static_cast<Index>(y * width + x);
            double dx = 0.0, dy = 0.0;
            if (x > 0 && x < width - 1)
            {
                dx = (scalar_field[idx + 1] - scalar_field[idx - 1]) / 2.0;
            }
            if (y > 0 && y < height - 1)
            {
                dy = (scalar_field[idx + width] - scalar_field[idx - width]) / 2.0;
            }
            double dxx = 0.0, dxy = 0.0, dyy = 0.0;
            if (x > 0 && x < width - 1)
            {
                dxx = (scalar_field[idx + 1] - 2 * scalar_field[idx] + scalar_field[idx - 1]) / 1.0;
            }
            if (y > 0 && y < height - 1)
            {
                dyy = (scalar_field[idx + width] - 2 * scalar_field[idx] +
                       scalar_field[idx - width]) /
                      1.0;
            }
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
            {
                dxy = (scalar_field[idx + width + 1] - scalar_field[idx - width - 1] -
                       scalar_field[idx + width - 1] + scalar_field[idx - width + 1]) /
                      4.0;
            }
            double det = dxx * dyy - dxy * dxy;
            double grad_magnitude = std::sqrt(dx * dx + dy * dy);
            if (det < -1e-10 && grad_magnitude > 1e-6)
            {
                saddles.push_back(idx);
            }
        }
    }
    return saddles;
}
double LevelSet::getComputationTime() const
{
    return computation_time_;
}
std::vector<Index> LevelSet::getNeighbors(Index point_index) const
{
    return getGridNeighbors(point_index);
}
std::vector<std::pair<algebra::Simplex, double>>
computeSublevelFiltration(const std::vector<double> &scalar_field, const std::vector<Size> &shape,
                          Size num_levels)
{
    LevelSet ls(shape);
    ls.setFiltrationType("sublevel");
    ls.setNumLevels(num_levels);
    core::BufferView<const double> fieldView(scalar_field.data(), scalar_field.size());
    auto result = ls.buildFiltration(fieldView);
    if (result.isSuccess())
    {
        return result.value();
    }
    else
    {
        return {};
    }
}
std::vector<std::pair<algebra::Simplex, double>>
computeSuperlevelFiltration(const std::vector<double> &scalar_field, const std::vector<Size> &shape,
                            Size num_levels)
{
    LevelSet ls(shape);
    ls.setFiltrationType("superlevel");
    ls.setNumLevels(num_levels);
    core::BufferView<const double> fieldView(scalar_field.data(), scalar_field.size());
    auto result = ls.buildFiltration(fieldView);
    if (result.isSuccess())
    {
        return result.value();
    }
    else
    {
        return {};
    }
}
errors::ErrorResult<Vector<std::pair<algebra::Simplex, double>>>
computeAdaptiveFiltration(const Vector<double> &scalar_field, const Vector<Size> &shape)
{
    LevelSet ls(shape);
    ls.setAdaptiveLevels(true);
    core::BufferView<const double> fieldView(scalar_field.data(), scalar_field.size());
    return ls.buildFiltration(fieldView);
}
Vector<Index> findCriticalPoints2d(const Vector<double> &scalar_field, Size width, Size height)
{
    LevelSet ls({width, height});
    return ls.findCriticalPoints(scalar_field);
}
Vector<Index> findCriticalPoints3d(const Vector<double> &scalar_field, Size width, Size height,
                                   Size depth)
{
    LevelSet ls({width, height, depth});
    return ls.findCriticalPoints(scalar_field);
}
Vector<Vector<double>> computeGradientField(const Vector<double> &scalar_field,
                                            const Vector<Size> &shape)
{
    Vector<Vector<double>> gradient(scalar_field.size());
    if (shape.size() == 2)
    {
        Size width = shape[0];
        Size height = shape[1];
        for (Size y = 0; y < height; ++y)
        {
            for (Size x = 0; x < width; ++x)
            {
                Index idx = static_cast<Index>(y * width + x);
                gradient[idx] = {0.0, 0.0};
                if (x > 0 && x < width - 1)
                {
                    gradient[idx][0] = (scalar_field[idx + 1] - scalar_field[idx - 1]) / 2.0;
                }
                else if (x > 0)
                {
                    gradient[idx][0] = scalar_field[idx] - scalar_field[idx - 1];
                }
                else if (x < width - 1)
                {
                    gradient[idx][0] = scalar_field[idx + 1] - scalar_field[idx];
                }
                if (y > 0 && y < height - 1)
                {
                    gradient[idx][1] =
                        (scalar_field[idx + width] - scalar_field[idx - width]) / 2.0;
                }
                else if (y > 0)
                {
                    gradient[idx][1] = scalar_field[idx] - scalar_field[idx - width];
                }
                else if (y < height - 1)
                {
                    gradient[idx][1] = scalar_field[idx + width] - scalar_field[idx];
                }
            }
        }
    }
    return gradient;
}
std::vector<std::vector<Index>> computeWatershed2d(const std::vector<double> &scalar_field,
                                                   Size width, Size height)
{
    LevelSet ls({width, height});
    auto minima = ls.findMinima(scalar_field);
    std::vector<std::vector<Index>> basins(minima.size());
    std::vector<Index> labels(scalar_field.size(), static_cast<Index>(-1));
    for (Size i = 0; i < minima.size(); ++i)
    {
        labels[minima[i]] = static_cast<Index>(i);
        basins[i].push_back(minima[i]);
    }
    for (Size i = 0; i < scalar_field.size(); ++i)
    {
        if (labels[i] == static_cast<Index>(-1))
        {
            std::vector<Index> neighbors = ls.getNeighbors(static_cast<Index>(i));
            for (Index neighbor : neighbors)
            {
                if (neighbor < static_cast<Index>(labels.size()) &&
                    labels[neighbor] != static_cast<Index>(-1))
                {
                    labels[i] = labels[neighbor];
                    basins[labels[neighbor]].push_back(static_cast<Index>(i));
                    break;
                }
            }
        }
    }
    return basins;
}
} // namespace nerve::filtration
