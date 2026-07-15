#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/kernels/ph4_ops.hpp"
#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::CompactSummary;
using nerve::persistence::Pair;
using nerve::persistence::PH5PH6Engine;
using nerve::persistence::SparseBoundaryMatrix;
using namespace nerve::test;

bool check_sparse_boundary_matrix_construction()
{
    SparseBoundaryMatrix<> mat(3, 3);
    if (mat.numRows() != 3 || mat.numCols() != 3)
    {
        std::cerr << "SparseBoundaryMatrix dimensions mismatch\n";
        return false;
    }
    return true;
}

bool check_sparse_boundary_matrix_add_column()
{
    SparseBoundaryMatrix<> mat(3, 3);
    mat.addColumn(0, {{0, 1}, {2, 1}});
    if (mat.numNonzero() != 2)
    {
        std::cerr << "expected 2 non-zeros after addColumn, got " << mat.numNonzero() << "\n";
        return false;
    }
    return true;
}

bool check_sparse_boundary_matrix_add_row()
{
    SparseBoundaryMatrix<> mat(3, 3);
    mat.addRow(1, {{0, 1}, {2, 1}});
    if (mat.numNonzero() != 2)
    {
        std::cerr << "expected 2 non-zeros after addRow, got " << mat.numNonzero() << "\n";
        return false;
    }
    return true;
}

bool check_sparse_boundary_matrix_get_set()
{
    SparseBoundaryMatrix<> mat(3, 3);
    mat.set(0, 1, 1);
    auto val = mat.get(0, 1);
    if (val != 1)
    {
        std::cerr << "expected get(0,1)=1, got " << val << "\n";
        return false;
    }
    if (!mat.isNonzero(0, 1))
    {
        std::cerr << "isNonzero(0,1) should be true\n";
        return false;
    }
    if (mat.isNonzero(2, 2))
    {
        std::cerr << "isNonzero(2,2) should be false\n";
        return false;
    }
    return true;
}

bool check_compact_summary_add_pair()
{
    CompactSummary summary;
    Pair p1{1.0, 3.0, 0};
    Pair p2{0.0, std::numeric_limits<double>::infinity(), 0};
    Pair p3{0.5, 1.5, 1};

    summary.addPair(p1);
    summary.addPair(p2);
    summary.addPair(p3);

    if (summary.getTotalPairs() != 3)
    {
        std::cerr << "expected 3 total pairs, got " << summary.getTotalPairs() << "\n";
        return false;
    }
    return true;
}

bool check_compact_summary_get_top_pairs()
{
    CompactSummary summary;
    summary.addPair(Pair{1.0, 5.0, 0});
    summary.addPair(Pair{2.0, 3.0, 0});
    summary.addPair(Pair{0.0, 10.0, 1});

    auto top = summary.getTopPairs(2);
    if (top.size() > 2)
    {
        std::cerr << "getTopPairs(2) returned " << top.size() << " pairs\n";
        return false;
    }
    return true;
}

bool check_compact_summary_compression_ratio()
{
    CompactSummary summary;
    Pair p{0.0, 1.0, 0};
    summary.addPair(p);
    auto ratio = summary.getCompressionRatio();
    if (ratio < 0.0)
    {
        std::cerr << "negative compression ratio\n";
        return false;
    }
    return true;
}

bool check_ph5_ph6_engine_construction()
{
    PH5PH6Engine<std::vector<double>>::Config config;
    config.numerical_tolerance = 1e-12;
    config.max_iterations = 10;

    PH5PH6Engine<std::vector<double>> engine(config);
    (void)engine;
    return true;
}

bool check_ph5_ph6_engine_compute()
{
    PH5PH6Engine<std::vector<double>> engine;
    std::vector<std::vector<double>> points = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};

    auto result = engine.computePersistenceCohomology(points, 1);
    if (!result.has_value())
    {
        return true;
    }

    const auto &pairs = result.value();
    for (const auto &p : pairs)
    {
        if (p.second < 0)
        {
            std::cerr << "negative scalar in PH5 result\n";
            return false;
        }
    }
    return true;
}

bool check_ph5_ph6_engine_metrics()
{
    PH5PH6Engine<std::vector<double>> engine;
    std::vector<std::vector<double>> points = {{0.0, 0.0}, {1.0, 0.0}};
    engine.computePersistenceCohomology(points, 1);

    auto metrics = engine.getComputationMetrics();
    if (metrics.computation_time_ms < 0.0)
    {
        std::cerr << "negative computation time\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_sparse_boundary_matrix_construction())
    {
        std::cerr << "FAIL: SparseBoundaryMatrix construction\n";
        return 1;
    }
    if (!check_sparse_boundary_matrix_add_column())
    {
        std::cerr << "FAIL: SparseBoundaryMatrix addColumn\n";
        return 1;
    }
    if (!check_sparse_boundary_matrix_add_row())
    {
        std::cerr << "FAIL: SparseBoundaryMatrix addRow\n";
        return 1;
    }
    if (!check_sparse_boundary_matrix_get_set())
    {
        std::cerr << "FAIL: SparseBoundaryMatrix get/set\n";
        return 1;
    }
    if (!check_compact_summary_add_pair())
    {
        std::cerr << "FAIL: CompactSummary addPair\n";
        return 1;
    }
    if (!check_compact_summary_get_top_pairs())
    {
        std::cerr << "FAIL: CompactSummary getTopPairs\n";
        return 1;
    }
    if (!check_compact_summary_compression_ratio())
    {
        std::cerr << "FAIL: CompactSummary compression ratio\n";
        return 1;
    }
    if (!check_ph5_ph6_engine_construction())
    {
        std::cerr << "FAIL: PH5PH6Engine construction\n";
        return 1;
    }
    if (!check_ph5_ph6_engine_compute())
    {
        std::cerr << "FAIL: PH5PH6Engine compute\n";
        return 1;
    }
    if (!check_ph5_ph6_engine_metrics())
    {
        std::cerr << "FAIL: PH5PH6Engine metrics\n";
        return 1;
    }
    return 0;
}
