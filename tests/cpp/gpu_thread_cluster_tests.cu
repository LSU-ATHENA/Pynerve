#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/thread_block_cluster.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU thread block cluster tests\n";
        return 0;
    }

    // Cluster: AdvancedClusterConfig defaults
    {
        nerve::persistence::accelerated::AdvancedClusterConfig cfg;
        assert(cfg.clusterSizeX == 8);
        assert(cfg.clusterSizeY == 1);
        assert(cfg.totalClusterSize() == 8);
        assert(cfg.useMulticast == true);
        assert(cfg.useDistributedL2 == true);
        std::cout << "PASSED: AdvancedClusterConfig defaults\n";
    }

    // Cluster: benchmarkClusterConfig
    {
        auto result = nerve::persistence::accelerated::benchmarkClusterConfig(64, 3);
        assert(result.optimalTimeMs >= 0.0f);
        std::cout << "PASSED: benchmarkClusterConfig (64 points, dim=3)\n";
    }

    // Cluster: getOrBenchmarkClusterConfig
    {
        auto result = nerve::persistence::accelerated::getOrBenchmarkClusterConfig(32, 2, "test_key");
        assert(result.optimalTimeMs >= 0.0f);
        std::cout << "PASSED: getOrBenchmarkClusterConfig (32 points, dim=2)\n";
    }

    // Cluster: clusterFeaturesAvailable
    {
        bool avail = nerve::persistence::accelerated::clusterFeaturesAvailable();
        std::cout << "PASSED: clusterFeaturesAvailable=" << avail << "\n";
    }

    // Cluster: nonPortableClustersAvailable
    {
        bool avail = nerve::persistence::accelerated::nonPortableClustersAvailable();
        std::cout << "PASSED: nonPortableClustersAvailable=" << avail << "\n";
    }

    return 0;
}
