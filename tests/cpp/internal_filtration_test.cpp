
#include "nerve/filtration/detail/filtration_detail.hpp"
#include "nerve/filtration/level_set.hpp"
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;

bool check_level_set_connectivity_2x2_grid()
{
    nerve::filtration::LevelSet ls({2, 2});
    std::vector<double> values = {0.0, 1.0, 1.0, 0.0};

    ls.buildGridConnectivity();
    auto neighbors = ls.getGridNeighbors(0);
    if (neighbors.size() != 2)
    {
        return false;
    }

    ls.buildVertexSimplices(values);
    ls.buildEdgeSimplices(values);
    ls.buildTriangleSimplices(values);
    ls.sortFiltration();

    Size total = ls.getNumSimplices();
    return total == 7;
}

bool check_level_set_num_levels_config()
{
    nerve::filtration::LevelSet ls({3, 3});
    ls.setNumLevels(5);
    ls.setAdaptiveLevels(false);
    ls.setFiltrationType("sublevel");
    return ls.getNumLevels() == 5;
}

bool check_level_set_grid_dimension()
{
    nerve::filtration::LevelSet ls({4, 5, 6});
    Size dim = ls.getGridDimension();
    if (dim != 3)
    {
        return false;
    }
    Size total = ls.getTotalGridPoints();
    return total == 120;
}

bool check_level_set_index_conversion()
{
    nerve::filtration::LevelSet ls({3, 4});
    std::vector<Index> coords = {1, 2};
    Index linear = ls.gridIndexToLinear(coords);
    if (linear != 7)
    {
        return false;
    }
    auto back = ls.linearIndexToGrid(linear);
    return back.size() == 2 && back[0] == 1 && back[1] == 2;
}

bool check_vr_builder_filtration_vertex_count()
{
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    auto result = nerve::filtration::computeVietorisRipsFiltration(points, 2, 2.0, 1);
    if (result.empty())
    {
        return false;
    }
    Size vertex_count = 0;
    for (const auto &[simplex, value] : result)
    {
        (void)value;
        if (simplex.dimension() == 0)
        {
            ++vertex_count;
        }
    }
    return vertex_count == 4;
}

bool check_vr_builder_all_pair_distances()
{
    std::vector<double> points = {0.0, 0.0, 3.0, 4.0};
    auto buf = nerve::filtration::computeAllPairDistances(points, 2);
    if (buf.size() == 0)
    {
        return false;
    }
    const double *data = static_cast<const double *>(buf.data());
    if (!data)
    {
        return false;
    }
    return std::abs(data[0] - 0.0) < 1e-10 && std::abs(data[3] - 5.0) < 1e-10;
}

bool check_vr_builder_knn()
{
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    auto neighbors = nerve::filtration::findKNearestNeighbors(points, 2, 0, 2);
    return neighbors.size() == 2;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("level_set_connectivity_2x2_grid", check_level_set_connectivity_2x2_grid());
    run("level_set_num_levels_config", check_level_set_num_levels_config());
    run("level_set_grid_dimension", check_level_set_grid_dimension());
    run("level_set_index_conversion", check_level_set_index_conversion());
    run("vr_builder_filtration_vertex_count", check_vr_builder_filtration_vertex_count());
    run("vr_builder_all_pair_distances", check_vr_builder_all_pair_distances());
    run("vr_builder_knn", check_vr_builder_knn());

    return failures > 0 ? 1 : 0;
}
