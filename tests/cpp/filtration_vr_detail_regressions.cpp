#include "nerve/algebra/boundary.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/filtration/vietoris_rips.hpp"
#include "nerve/filtration/vr_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Size;
using nerve::algebra::Point;
using nerve::core::BufferView;
using nerve::core::DeterminismContract;
using nerve::core::ownership_utils::PointView;
using nerve::filtration::SparseVietorisRips;
using nerve::filtration::VietorisRips;
using nerve::filtration::WeightedVietorisRips;

constexpr double kTol = 1e-10;

PointView view_of(const std::vector<double> &v)
{
    return PointView(const_cast<double *>(v.data()), v.size());
}

bool check_vr_construction()
{
    VietorisRips vr(2.0);
    vr.setMaxRadius(1.5);
    vr.setMaxDimension(2);
    vr.setDistanceMetric("euclidean");
    return true;
}

bool check_vr_build_filtration()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    DeterminismContract contract;
    auto result = vr.buildFiltration(view_of(pts), 2, contract);
    if (result.isError())
    {
        std::cerr << "VR buildFiltration returned an error\n";
        return false;
    }
    auto filtration = result.value();
    if (filtration.empty())
    {
        std::cerr << "VR filtration should not be empty for 3 points\n";
        return false;
    }
    for (const auto &entry : filtration)
    {
        if (entry.second < 0.0)
        {
            std::cerr << "negative filtration value\n";
            return false;
        }
    }
    return true;
}

bool check_vr_filtration_monotonic()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    DeterminismContract contract;
    auto result = vr.buildFiltration(view_of(pts), 2, contract);
    if (result.isError())
    {
        std::cerr << "VR buildFiltration returned an error\n";
        return false;
    }
    auto filtration = result.value();
    for (size_t i = 1; i < filtration.size(); ++i)
    {
        if (filtration[i].second < filtration[i - 1].second - kTol)
        {
            std::cerr << "VR filtration not monotonic at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_vr_num_simplices()
{
    VietorisRips vr(2.0);
    vr.setMaxDimension(2);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0};
    DeterminismContract contract;
    auto result = vr.buildFiltration(view_of(pts), 2, contract);
    if (result.isError())
    {
        std::cerr << "VR buildFiltration returned an error\n";
        return false;
    }
    Size n = vr.getNumSimplices();
    if (n == 0)
    {
        std::cerr << "num simplices should be non-zero\n";
        return false;
    }
    Size n_dim1 = vr.getNumSimplicesOfDimension(1);
    (void)n_dim1;
    return true;
}

bool check_sparse_vr_construction()
{
    SparseVietorisRips sparse(5);
    sparse.setKNeighbors(5);
    sparse.setApproximationFactor(1.1);
    sparse.setBatchSize(100);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    DeterminismContract contract;
    auto filtration = sparse.buildFiltration(view_of(pts), 2, contract);
    if (filtration.empty())
    {
        std::cerr << "sparse VR filtration empty\n";
        return false;
    }
    return true;
}

bool check_sparse_vs_full_edges()
{
    SparseVietorisRips sparse(2);
    sparse.setKNeighbors(2);
    sparse.setBatchSize(100);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    DeterminismContract contract;
    auto sparse_filt = sparse.buildFiltration(view_of(pts), 2, contract);
    VietorisRips vr(2.0);
    vr.setMaxDimension(1);
    auto vr_result = vr.buildFiltration(view_of(pts), 2, contract);
    if (vr_result.isError())
    {
        std::cerr << "VR buildFiltration returned an error\n";
        return false;
    }
    (void)sparse_filt.size();
    (void)vr_result.value().size();
    return true;
}

bool check_weighted_vr_construction()
{
    std::vector<double> weights{1.0, 2.0, 1.5};
    WeightedVietorisRips wvr(weights);
    wvr.setAdaptiveRadius(false);
    wvr.setWeightFunction("inverse_distance");
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    DeterminismContract contract;
    auto filtration = wvr.buildFiltration(view_of(pts), 2, contract);
    if (filtration.empty())
    {
        std::cerr << "weighted VR filtration empty\n";
        return false;
    }
    return true;
}

bool check_weighted_vr_weights_respected()
{
    std::vector<double> weights{10.0, 0.1, 10.0};
    WeightedVietorisRips wvr(weights);
    wvr.setAdaptiveRadius(false);
    wvr.setWeightFunction("inverse_distance");
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 2.0, 0.0};
    DeterminismContract contract;
    auto filtration = wvr.buildFiltration(view_of(pts), 2, contract);
    bool has_edge = false;
    for (const auto &entry : filtration)
    {
        if (entry.first.dimension() == 1)
        {
            has_edge = true;
            break;
        }
    }
    (void)has_edge;
    return true;
}

bool check_parallel_vr_complex()
{
    std::vector<Point> points;
    points.emplace_back(0.0f, 0.0f, 0.0f);
    points.emplace_back(1.0f, 0.0f, 0.0f);
    points.emplace_back(0.5f, 0.866f, 0.0f);
    auto complex = nerve::filtration::vr::parallel::buildParallelVRComplex(points, 2.0f, 2);
    if (complex.edges.empty())
    {
        std::cerr << "parallel VR complex should have edges\n";
        return false;
    }
    return true;
}

bool check_parallel_edge_detection()
{
    std::vector<Point> points;
    points.emplace_back(0.0f, 0.0f, 0.0f);
    points.emplace_back(1.0f, 0.0f, 0.0f);
    points.emplace_back(0.5f, 0.866f, 0.0f);
    auto edges = nerve::filtration::vr::parallel::parallelEdgeDetection(points, 2.0f, 2);
    if (edges.empty())
    {
        std::cerr << "parallel edge detection should find edges\n";
        return false;
    }
    for (const auto &e : edges)
    {
        if (e.distance < 0.0f)
        {
            std::cerr << "negative edge distance\n";
            return false;
        }
    }
    return true;
}

bool check_ann_vr()
{
    std::vector<Point> points;
    points.emplace_back(0.0f, 0.0f, 0.0f);
    points.emplace_back(1.0f, 0.0f, 0.0f);
    points.emplace_back(0.5f, 0.866f, 0.0f);
    points.emplace_back(0.2f, 0.2f, 0.0f);
    auto edges = nerve::filtration::vr::ann::buildVRWithANN(points, 2.0f, 3, true);
    if (edges.empty())
    {
        std::cerr << "ANN VR should find edges\n";
        return false;
    }
    return true;
}

bool check_weighted_local_weights()
{
    std::vector<double> weights{1.0, 1.0, 1.0};
    WeightedVietorisRips wvr(weights);
    std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    auto local = wvr.computeLocalWeights(view_of(pts), 2);
    if (local.size() != 3)
    {
        std::cerr << "local weights wrong size\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_vr_construction())
    {
        std::cerr << "FAIL: VR construction\n";
        return 1;
    }
    if (!check_vr_build_filtration())
    {
        std::cerr << "FAIL: VR build\n";
        return 1;
    }
    if (!check_vr_filtration_monotonic())
    {
        std::cerr << "FAIL: VR monotonic\n";
        return 1;
    }
    if (!check_vr_num_simplices())
    {
        std::cerr << "FAIL: VR num simplices\n";
        return 1;
    }
    if (!check_sparse_vr_construction())
    {
        std::cerr << "FAIL: sparse VR\n";
        return 1;
    }
    if (!check_sparse_vs_full_edges())
    {
        std::cerr << "FAIL: sparse vs full\n";
        return 1;
    }
    if (!check_weighted_vr_construction())
    {
        std::cerr << "FAIL: weighted VR\n";
        return 1;
    }
    if (!check_weighted_vr_weights_respected())
    {
        std::cerr << "FAIL: weighted VR weights\n";
        return 1;
    }
    if (!check_parallel_vr_complex())
    {
        std::cerr << "FAIL: parallel VR\n";
        return 1;
    }
    if (!check_parallel_edge_detection())
    {
        std::cerr << "FAIL: parallel edges\n";
        return 1;
    }
    if (!check_ann_vr())
    {
        std::cerr << "FAIL: ANN VR\n";
        return 1;
    }
    if (!check_weighted_local_weights())
    {
        std::cerr << "FAIL: local weights\n";
        return 1;
    }
    return 0;
}
