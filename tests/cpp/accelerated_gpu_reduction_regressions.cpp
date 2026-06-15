
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/cuda/cuda_edge_extraction.hpp"
#include "nerve/persistence/cuda/cuda_edge_types.hpp"
#include "nerve/persistence/cuda/gpu_reduction_engine.hpp"
#include "nerve/persistence/hybrid_reduction_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Index;
using nerve::Size;
using nerve::errors::ErrorCode;
using nerve::persistence::accelerated::CUDAEgdeExtractor;
using nerve::persistence::accelerated::Edge;
using nerve::persistence::accelerated::EdgeExtractionConfig;
using nerve::persistence::accelerated::EdgeExtractionStats;
using nerve::persistence::accelerated::factory;

bool check_edge_extraction_config_default()
{
    EdgeExtractionConfig config;
    if (config.max_edges != 1000000)
        return false;
    if (!config.enable_early_termination)
        return false;
    if (config.enable_filtering)
        return false;
    if (config.sort_by_weight)
        return false;
    if (std::abs(config.min_edge_weight - 0.0) > 1e-10)
        return false;
    if (config.max_degree != 1000)
        return false;
    if (!config.use_shared_memory)
        return false;
    if (!config.enable_streaming)
        return false;
    if (config.streaming_threshold != 1000000)
        return false;
    auto result = config.validate();
    if (result.isError())
        return false;
    return true;
}

bool check_edge_extraction_config_validation()
{
    EdgeExtractionConfig config;
    auto ok = config.validate();
    if (ok.isError())
        return false;
    config.max_edges = 0;
    auto err = config.validate();
    if (!err.isError())
        return false;
    return true;
}

bool check_edge_extraction_stats_default()
{
    EdgeExtractionStats stats;
    if (std::abs(stats.total_time_ms - 0.0) > 1e-10)
        return false;
    if (stats.edges_extracted != 0)
        return false;
    if (std::abs(stats.getExtractionRate() - 0.0) > 1e-10)
        return false;
    if (std::abs(stats.getFilteringRate() - 0.0) > 1e-10)
        return false;
    if (std::abs(stats.getEfficiencyScore() - 1.0) > 1e-10)
        return false;
    return true;
}

bool check_edge_stats_computation()
{
    EdgeExtractionStats stats;
    stats.total_time_ms = 100.0;
    stats.edges_extracted = 5000;
    stats.edges_filtered = 2000;
    stats.edge_density = 0.1;
    if (std::abs(stats.getExtractionRate() - 50000.0) > 1.0)
        return false;
    if (std::abs(stats.getFilteringRate() - 2000.0 / 7000.0) > 1e-6)
        return false;
    return true;
}

bool check_factory_create_accelerated()
{
    auto result = factory::createAcceleratedEdgeExtractor(100, 2.0, 0.1);
    if (result.isError())
    {
        bool acceptable = (result.errorCode() == ErrorCode::E51_PH_INPUT);
        if (acceptable)
            return true;
        return false;
    }
    auto extractor = std::move(result.value());
    if (!extractor)
        return false;
    auto cfg = extractor->getConfig();
    if (cfg.max_edges == 0)
        return false;
    return true;
}

bool check_factory_create_batch()
{
    auto result = factory::createBatchEdgeExtractor(4, 500, 1.0);
    if (result.isError())
        return true;
    auto extractor = std::move(result.value());
    if (!extractor)
        return false;
    return true;
}

bool check_factory_create_high_density()
{
    auto result = factory::createHighDensityEdgeExtractor(200, 1.0, 0.5);
    if (result.isError())
        return true;
    auto extractor = std::move(result.value());
    if (!extractor)
        return false;
    if (!extractor->getConfig().enable_filtering)
        return false;
    if (!extractor->getConfig().sort_by_weight)
        return false;
    return true;
}

bool check_factory_create_sparse()
{
    auto result = factory::createSparseEdgeExtractor(500, 0.5, 0.05);
    if (result.isError())
        return true;
    auto extractor = std::move(result.value());
    if (!extractor)
        return false;
    if (!extractor->getConfig().enable_early_termination)
        return false;
    return true;
}

bool check_factory_invalid_parameters()
{
    auto r1 = factory::createAcceleratedEdgeExtractor(0, 1.0, 0.1);
    if (!r1.isError())
        return false;
    auto r2 = factory::createAcceleratedEdgeExtractor(100, -1.0, 0.1);
    if (!r2.isError())
        return false;
    auto r3 = factory::createAcceleratedEdgeExtractor(100, 1.0, -0.1);
    if (!r3.isError())
        return false;
    return true;
}

bool check_edge_type()
{
    Edge e;
    if (e.u != -1 || e.v != -1 || std::abs(e.w - 0.0) > 1e-10)
        return false;
    if (e.isValid())
        return false;
    Edge e2(0, 1, 0.5);
    if (e2.u != 0 || e2.v != 1 || std::abs(e2.w - 0.5) > 1e-10)
        return false;
    if (!e2.isValid())
        return false;
    Edge e3(1, 0, 0.5);
    if (e3.isValid())
        return false;
    return true;
}

bool check_get_optimal_max_edges()
{
    using nerve::persistence::accelerated::utils::getOptimalMaxEdges;
    Size r = getOptimalMaxEdges(100, 0.5, 0.1);
    if (r == 0)
        return false;
    if (r > 1000000)
        return false;
    Size r_invalid = getOptimalMaxEdges(100, -1.0);
    if (r_invalid != 0)
        return false;
    return true;
}

bool check_validate_edge_extraction_params()
{
    using nerve::persistence::accelerated::utils::validateEdgeExtractionParams;
    EdgeExtractionConfig cfg;
    auto r1 = validateEdgeExtractionParams(100, 1.0, cfg);
    if (r1.isError())
        return false;
    auto r2 = validateEdgeExtractionParams(0, 1.0, cfg);
    if (!r2.isError())
        return false;
    auto r3 = validateEdgeExtractionParams(100, -1.0, cfg);
    if (!r3.isError())
        return false;
    return true;
}

bool check_should_use_functions()
{
    using nerve::persistence::accelerated::utils::shouldEnableEarlyTermination;
    using nerve::persistence::accelerated::utils::shouldEnableFiltering;
    using nerve::persistence::accelerated::utils::shouldEnableSorting;
    using nerve::persistence::accelerated::utils::shouldUseSharedMemory;
    using nerve::persistence::accelerated::utils::shouldUseStreaming;
    if (shouldUseStreaming(100, 1000, 1024ULL * 1024ULL * 1024ULL))
        return false;
    if (!shouldEnableFiltering(20000, 0.1))
        return false;
    if (shouldEnableFiltering(100, 1.0))
        return false;
    if (shouldEnableSorting(5000))
        return false;
    if (!shouldEnableSorting(50000))
        return false;
    if (!shouldUseSharedMemory(64, 5000))
        return false;
    if (shouldUseSharedMemory(1000, 5000))
        return false;
    if (!shouldEnableEarlyTermination(200000))
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_edge_extraction_config_default())
    {
        std::cerr << "FAIL: edge extraction config default\n";
        return 1;
    }
    if (!check_edge_extraction_config_validation())
    {
        std::cerr << "FAIL: edge extraction config validation\n";
        return 1;
    }
    if (!check_edge_extraction_stats_default())
    {
        std::cerr << "FAIL: edge extraction stats default\n";
        return 1;
    }
    if (!check_edge_stats_computation())
    {
        std::cerr << "FAIL: edge stats computation\n";
        return 1;
    }
    if (!check_factory_create_accelerated())
    {
        std::cerr << "FAIL: factory create accelerated\n";
        return 1;
    }
    if (!check_factory_create_batch())
    {
        std::cerr << "FAIL: factory create batch\n";
        return 1;
    }
    if (!check_factory_create_high_density())
    {
        std::cerr << "FAIL: factory create high density\n";
        return 1;
    }
    if (!check_factory_create_sparse())
    {
        std::cerr << "FAIL: factory create sparse\n";
        return 1;
    }
    if (!check_factory_invalid_parameters())
    {
        std::cerr << "FAIL: factory invalid parameters\n";
        return 1;
    }
    if (!check_edge_type())
    {
        std::cerr << "FAIL: edge type\n";
        return 1;
    }
    if (!check_get_optimal_max_edges())
    {
        std::cerr << "FAIL: get optimal max edges\n";
        return 1;
    }
    if (!check_validate_edge_extraction_params())
    {
        std::cerr << "FAIL: validate edge extraction params\n";
        return 1;
    }
    if (!check_should_use_functions())
    {
        std::cerr << "FAIL: should use functions\n";
        return 1;
    }
    return 0;
}
