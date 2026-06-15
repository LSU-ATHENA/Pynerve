
#include "nerve/persistence/cuda/thread_block_cluster.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::accelerated::AdvancedClusterConfig;
using nerve::persistence::accelerated::CLUSTER_DEFAULT_SIZE_X;
using nerve::persistence::accelerated::CLUSTER_DEFAULT_SIZE_Y;
using nerve::persistence::accelerated::CLUSTER_DEFAULT_SIZE_Z;
using nerve::persistence::accelerated::CLUSTER_MAX_SIZE;
using nerve::persistence::accelerated::ClusterBenchmarkConfig;
using nerve::persistence::accelerated::ClusterBenchmarkResult;

bool check_cluster_constants()
{
    if (CLUSTER_MAX_SIZE != 16)
        return false;
    if (CLUSTER_DEFAULT_SIZE_X != 8)
        return false;
    if (CLUSTER_DEFAULT_SIZE_Y != 1)
        return false;
    if (CLUSTER_DEFAULT_SIZE_Z != 1)
        return false;
    return true;
}

bool check_advanced_cluster_config_default()
{
    AdvancedClusterConfig cfg;
    if (cfg.clusterSizeX != CLUSTER_DEFAULT_SIZE_X)
        return false;
    if (cfg.clusterSizeY != CLUSTER_DEFAULT_SIZE_Y)
        return false;
    if (cfg.clusterSizeZ != CLUSTER_DEFAULT_SIZE_Z)
        return false;
    if (cfg.useNonPortable)
        return false;
    if (!cfg.useMulticast)
        return false;
    if (!cfg.useDistributedL2)
        return false;
    if (cfg.sharedMemPerBlock == 0)
        return false;
    if (cfg.l2PromotionSize <= 0)
        return false;
    if (cfg.numPipelineStages <= 0)
        return false;
    int total = cfg.totalClusterSize();
    if (total != CLUSTER_DEFAULT_SIZE_X * CLUSTER_DEFAULT_SIZE_Y * CLUSTER_DEFAULT_SIZE_Z)
        return false;
    return true;
}

bool check_advanced_cluster_config_total_size()
{
    AdvancedClusterConfig cfg;
    cfg.clusterSizeX = 4;
    cfg.clusterSizeY = 2;
    cfg.clusterSizeZ = 1;
    if (cfg.totalClusterSize() != 8)
        return false;
    cfg.clusterSizeX = 16;
    cfg.clusterSizeY = 1;
    cfg.clusterSizeZ = 1;
    if (cfg.totalClusterSize() != 16)
        return false;
    return true;
}

bool check_advanced_cluster_config_is_valid()
{
    AdvancedClusterConfig cfg;
    cfg.clusterSizeX = 8;
    cfg.clusterSizeY = 1;
    cfg.clusterSizeZ = 1;
    bool valid = cfg.isValid(90);
    if (!valid)
        return false;
    return true;
}

bool check_advanced_cluster_config_invalid_negative()
{
    AdvancedClusterConfig cfg;
    cfg.clusterSizeX = -1;
    bool valid = cfg.isValid(90);
    if (valid)
        return false;
    return true;
}

bool check_advanced_cluster_config_invalid_too_large()
{
    AdvancedClusterConfig cfg;
    cfg.clusterSizeX = 16;
    cfg.clusterSizeY = 2;
    cfg.clusterSizeZ = 1;
    bool valid = cfg.isValid(90);
    if (valid)
        return false;
    return true;
}

bool check_advanced_cluster_config_invalid_smem()
{
    AdvancedClusterConfig cfg;
    cfg.sharedMemPerBlock = 0;
    bool valid = cfg.isValid(90);
    if (valid)
        return false;
    return true;
}

bool check_advanced_cluster_config_invalid_stages()
{
    AdvancedClusterConfig cfg;
    cfg.numPipelineStages = 0;
    bool valid = cfg.isValid(90);
    if (valid)
        return false;
    return true;
}

bool check_advanced_cluster_total_distributed_smem()
{
    AdvancedClusterConfig cfg;
    cfg.clusterSizeX = 4;
    cfg.clusterSizeY = 1;
    cfg.clusterSizeZ = 1;
    cfg.sharedMemPerBlock = 65536;
    size_t total = cfg.totalDistributedSmem();
    if (total != 65536ULL * 4)
        return false;
    return true;
}

bool check_benchmark_config_default()
{
    ClusterBenchmarkConfig cfg;
    if (cfg.minClusterSizeX != 2)
        return false;
    if (cfg.maxClusterSizeX != 16)
        return false;
    if (cfg.minClusterSizeY != 1)
        return false;
    if (cfg.maxClusterSizeY != 2)
        return false;
    if (cfg.numTrials != 10)
        return false;
    if (!cfg.testNonPortable)
        return false;
    return true;
}

bool check_benchmark_result_default()
{
    ClusterBenchmarkResult result;
    if (result.optimalClusterSizeX != 1)
        return false;
    if (result.optimalClusterSizeY != 1)
        return false;
    if (result.optimalClusterSizeZ != 1)
        return false;
    if (result.useNonPortable)
        return false;
    if (result.optimalTimeMs != std::numeric_limits<float>::infinity())
        return false;
    if (result.blackwellRequired)
        return false;
    return true;
}

bool check_cluster_features_available()
{
    bool avail = nerve::persistence::accelerated::clusterFeaturesAvailable();
    static_cast<void>(avail);
    return true;
}

bool check_non_portable_clusters_available()
{
    bool avail = nerve::persistence::accelerated::nonPortableClustersAvailable();
    static_cast<void>(avail);
    return true;
}

bool check_get_or_benchmark_cluster_config()
{
    auto result = nerve::persistence::accelerated::getOrBenchmarkClusterConfig(0, 0, "test_key");
    if (result.optimalClusterSizeX != 1)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_cluster_constants())
    {
        std::cerr << "FAIL: cluster constants\n";
        return 1;
    }
    if (!check_advanced_cluster_config_default())
    {
        std::cerr << "FAIL: advanced cluster config default\n";
        return 1;
    }
    if (!check_advanced_cluster_config_total_size())
    {
        std::cerr << "FAIL: advanced cluster config total size\n";
        return 1;
    }
    if (!check_advanced_cluster_config_is_valid())
    {
        std::cerr << "FAIL: advanced cluster config is valid\n";
        return 1;
    }
    if (!check_advanced_cluster_config_invalid_negative())
    {
        std::cerr << "FAIL: advanced cluster config invalid negative\n";
        return 1;
    }
    if (!check_advanced_cluster_config_invalid_too_large())
    {
        std::cerr << "FAIL: advanced cluster config invalid too large\n";
        return 1;
    }
    if (!check_advanced_cluster_config_invalid_smem())
    {
        std::cerr << "FAIL: advanced cluster config invalid smem\n";
        return 1;
    }
    if (!check_advanced_cluster_config_invalid_stages())
    {
        std::cerr << "FAIL: advanced cluster config invalid stages\n";
        return 1;
    }
    if (!check_advanced_cluster_total_distributed_smem())
    {
        std::cerr << "FAIL: advanced cluster total distributed smem\n";
        return 1;
    }
    if (!check_benchmark_config_default())
    {
        std::cerr << "FAIL: benchmark config default\n";
        return 1;
    }
    if (!check_benchmark_result_default())
    {
        std::cerr << "FAIL: benchmark result default\n";
        return 1;
    }
    if (!check_cluster_features_available())
    {
        std::cerr << "FAIL: cluster features available\n";
        return 1;
    }
    if (!check_non_portable_clusters_available())
    {
        std::cerr << "FAIL: non portable clusters available\n";
        return 1;
    }
    if (!check_get_or_benchmark_cluster_config())
    {
        std::cerr << "FAIL: get or benchmark cluster config\n";
        return 1;
    }
    return 0;
}
