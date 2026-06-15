
#include "nerve/core_types.hpp"
#include "nerve/streaming/streaming_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;
using nerve::streaming::IncrementalGraphLaplacian;
using nerve::streaming::IncrementalLaplacianConfig;
using nerve::streaming::StreamingLaplacianProcessor;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

bool check_incremental_laplacian_construction()
{
    IncrementalLaplacianConfig config;
    config.max_dimension = 2;
    config.smoothing_sigma = 0.1;
    config.eigenvalue_tolerance = 1e-10;
    IncrementalGraphLaplacian lap(config);
    if (lap.getVertexCount() != 0)
    {
        std::cerr << "new laplacian should have 0 vertices\n";
        return false;
    }
    if (lap.getEdgeCount() != 0)
    {
        std::cerr << "new laplacian should have 0 edges\n";
        return false;
    }
    return true;
}

bool check_add_vertices_and_edges()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addVertex(2);
    if (lap.getVertexCount() != 3)
    {
        std::cerr << "expected 3 vertices, got " << lap.getVertexCount() << "\n";
        return false;
    }
    lap.addEdge(0, 1, 1.0);
    lap.addEdge(1, 2, 1.0);
    if (lap.getEdgeCount() != 2)
    {
        std::cerr << "expected 2 edges, got " << lap.getEdgeCount() << "\n";
        return false;
    }
    return true;
}

bool check_laplacian_spectrum()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addEdge(0, 1, 1.0);
    auto spectrum = lap.computeSpectrum();
    if (spectrum.eigenvalues.empty())
    {
        std::cerr << "spectrum should have eigenvalues\n";
        return false;
    }
    for (double ev : spectrum.eigenvalues)
    {
        if (!std::isfinite(ev))
        {
            std::cerr << "eigenvalue not finite\n";
            return false;
        }
    }
    if (!std::isfinite(spectrum.trace))
    {
        std::cerr << "trace not finite\n";
        return false;
    }
    return true;
}

bool check_remove_vertex()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addEdge(0, 1, 1.0);
    lap.removeVertex(1);
    if (lap.getVertexCount() != 1)
    {
        std::cerr << "after removal expected 1 vertex, got " << lap.getVertexCount() << "\n";
        return false;
    }
    return true;
}

bool check_remove_edge()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addEdge(0, 1, 1.0);
    lap.removeEdge(0, 1);
    if (lap.getEdgeCount() != 0)
    {
        std::cerr << "after edge removal expected 0, got " << lap.getEdgeCount() << "\n";
        return false;
    }
    return true;
}

bool check_spectral_gap()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addEdge(0, 1, 1.0);
    double gap = lap.computeSpectralGap();
    if (!std::isfinite(gap))
    {
        std::cerr << "spectral gap not finite\n";
        return false;
    }
    if (gap < 0.0)
    {
        std::cerr << "spectral gap negative\n";
        return false;
    }
    return true;
}

bool check_fiedler_vector()
{
    IncrementalGraphLaplacian lap;
    lap.addVertex(0);
    lap.addVertex(1);
    lap.addEdge(0, 1, 1.0);
    auto fiedler = lap.computeFiedlerVector();
    if (fiedler.empty())
    {
        std::cerr << "fiedler vector should not be empty\n";
        return false;
    }
    return true;
}

bool check_streaming_laplacian_processor()
{
    IncrementalLaplacianConfig config;
    StreamingLaplacianProcessor processor(config);
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    std::vector<Index> indices = {0, 1, 2};
    processor.ingestPoints(points, 2, indices);
    Size count = processor.getProcessedCount();
    if (count == 0)
    {
        std::cerr << "processed count should be > 0\n";
        return false;
    }
    return true;
}

bool check_streaming_laplacian_spectrum_timeline()
{
    IncrementalLaplacianConfig config;
    StreamingLaplacianProcessor processor(config);
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    std::vector<Index> indices = {0, 1, 2};
    processor.ingestPoints(points, 2, indices);
    auto latest = processor.getLatestSpectrum();
    if (!latest.eigenvalues.empty())
    {
        for (double ev : latest.eigenvalues)
        {
            if (!std::isfinite(ev))
            {
                std::cerr << "non-finite eigenvalue in latest spectrum\n";
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_incremental_laplacian_construction())
    {
        std::cerr << "FAIL: laplacian construction\n";
        return 1;
    }
    if (!check_add_vertices_and_edges())
    {
        std::cerr << "FAIL: add vertices/edges\n";
        return 1;
    }
    if (!check_laplacian_spectrum())
    {
        std::cerr << "FAIL: laplacian spectrum\n";
        return 1;
    }
    if (!check_remove_vertex())
    {
        std::cerr << "FAIL: remove vertex\n";
        return 1;
    }
    if (!check_remove_edge())
    {
        std::cerr << "FAIL: remove edge\n";
        return 1;
    }
    if (!check_spectral_gap())
    {
        std::cerr << "FAIL: spectral gap\n";
        return 1;
    }
    if (!check_fiedler_vector())
    {
        std::cerr << "FAIL: fiedler vector\n";
        return 1;
    }
    if (!check_streaming_laplacian_processor())
    {
        std::cerr << "FAIL: streaming processor\n";
        return 1;
    }
    if (!check_streaming_laplacian_spectrum_timeline())
    {
        std::cerr << "FAIL: spectrum timeline\n";
        return 1;
    }
    return 0;
}
