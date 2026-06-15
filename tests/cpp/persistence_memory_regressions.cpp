#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/memory/clear_and_compress.hpp"
#include "nerve/persistence/memory/numa_memory_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::persistence::clearcompress::Chunk;
using nerve::persistence::clearcompress::ChunkColumn;
using nerve::persistence::clearcompress::ChunkConfig;
using nerve::persistence::clearcompress::ChunkReductionResult;
using nerve::persistence::clearcompress::ClearCompressResult;
using nerve::persistence::clearcompress::ClearCompressSpeedup;
using nerve::persistence::clearcompress::PersistencePair;

bool check_chunk_config_defaults()
{
    ChunkConfig config;
    if (config.chunk_size != 65536)
    {
        std::cerr << "default chunk_size expected 65536, got " << config.chunk_size << "\n";
        return false;
    }
    if (!config.compress_after_reduction)
    {
        std::cerr << "default compress_after_reduction should be true\n";
        return false;
    }
    if (config.num_threads != 0)
    {
        std::cerr << "default num_threads should be 0\n";
        return false;
    }
    return true;
}

bool check_chunk_defaults()
{
    Chunk chunk;
    if (chunk.chunk_index != -1)
    {
        std::cerr << "default chunk_index should be -1\n";
        return false;
    }
    if (chunk.start_column != 0)
    {
        std::cerr << "default start_column should be 0\n";
        return false;
    }
    if (chunk.is_compressed)
    {
        std::cerr << "default is_compressed should be false\n";
        return false;
    }
    return true;
}

bool check_chunk_column_defaults()
{
    ChunkColumn col;
    if (col.global_index != -1)
    {
        std::cerr << "default global_index should be -1\n";
        return false;
    }
    if (col.is_cleared)
    {
        std::cerr << "default is_cleared should be false\n";
        return false;
    }
    return true;
}

bool check_persistence_pair_defaults()
{
    PersistencePair pp;
    if (pp.birth_index != -1)
    {
        std::cerr << "default birth_index should be -1\n";
        return false;
    }
    if (pp.death_index != -1)
    {
        std::cerr << "default death_index should be -1\n";
        return false;
    }
    if (pp.birth_time != 0.0 || pp.death_time != 0.0)
    {
        std::cerr << "default times should be 0.0\n";
        return false;
    }
    return true;
}

bool check_clear_compress_result_defaults()
{
    ClearCompressResult result;
    if (!result.all_pairs.empty())
    {
        std::cerr << "default all_pairs should be empty\n";
        return false;
    }
    if (result.total_time_ms != 0.0)
    {
        std::cerr << "default total_time_ms should be 0.0\n";
        return false;
    }
    if (result.num_chunks != 0)
    {
        std::cerr << "default num_chunks should be 0\n";
        return false;
    }
    return true;
}

bool check_chunk_reduction_result_defaults()
{
    ChunkReductionResult result;
    if (result.chunk_index != -1)
    {
        std::cerr << "default chunk_index should be -1\n";
        return false;
    }
    if (!result.pairs.empty())
    {
        std::cerr << "default pairs should be empty\n";
        return false;
    }
    if (result.xor_operations != 0)
    {
        std::cerr << "default xor_operations should be 0\n";
        return false;
    }
    return true;
}

bool check_clear_compress_speedup_defaults()
{
    ClearCompressSpeedup speedup;
    if (std::abs(speedup.total_speedup - 1.0) > 1e-12)
    {
        std::cerr << "default total_speedup should be 1.0\n";
        return false;
    }
    if (speedup.memory_reduction != 0.0)
    {
        std::cerr << "default memory_reduction should be 0.0\n";
        return false;
    }
    return true;
}

bool check_build_chunks_small()
{
    std::vector<std::vector<int>> matrix = {{0, 1}, {1, 2}, {0, 2}};
    ChunkConfig config;
    config.chunk_size = 2;

    auto chunks = nerve::persistence::clearcompress::buildChunks(matrix, config);
    if (chunks.empty())
    {
        std::cerr << "buildChunks should produce at least 1 chunk\n";
        return false;
    }
    if (chunks[0].columns.empty())
    {
        std::cerr << "chunk should have columns\n";
        return false;
    }
    return true;
}

bool check_optimal_chunk_config()
{
    auto config = nerve::persistence::clearcompress::getOptimalChunkConfig(1000, 64);
    if (config.chunk_size == 0)
    {
        std::cerr << "optimal chunk_size should be positive\n";
        return false;
    }
    return true;
}

bool check_estimate_clear_compress_speedup()
{
    auto speedup = nerve::persistence::clearcompress::estimateClearCompressSpeedup(1000, 64, 10);
    if (speedup.total_speedup <= 0.0)
    {
        std::cerr << "speedup estimate should be positive\n";
        return false;
    }
    return true;
}

bool check_should_use_clear_compress()
{
    bool use_large = nerve::persistence::clearcompress::shouldUseClearCompress(100000, 10);
    bool use_small = nerve::persistence::clearcompress::shouldUseClearCompress(10, 1);

    if (use_small)
    {
        std::cerr << "shouldUseClearCompress(10,1) expected false\n";
        return false;
    }
    (void)use_large;
    return true;
}

bool check_numa_is_available()
{
    bool available = nerve::persistence::numa::isNumaAvailable();
    (void)available;
    return true;
}

bool check_numa_get_current_node()
{
    int node = nerve::persistence::numa::getCurrentNode();
    (void)node;
    return true;
}

} // namespace

int main()
{
    if (!check_chunk_config_defaults())
    {
        std::cerr << "FAIL: chunk config defaults\n";
        return 1;
    }
    if (!check_chunk_defaults())
    {
        std::cerr << "FAIL: chunk defaults\n";
        return 1;
    }
    if (!check_chunk_column_defaults())
    {
        std::cerr << "FAIL: chunk column defaults\n";
        return 1;
    }
    if (!check_persistence_pair_defaults())
    {
        std::cerr << "FAIL: persistence pair defaults\n";
        return 1;
    }
    if (!check_clear_compress_result_defaults())
    {
        std::cerr << "FAIL: clear compress result defaults\n";
        return 1;
    }
    if (!check_chunk_reduction_result_defaults())
    {
        std::cerr << "FAIL: chunk reduction result defaults\n";
        return 1;
    }
    if (!check_clear_compress_speedup_defaults())
    {
        std::cerr << "FAIL: clear compress speedup defaults\n";
        return 1;
    }
    if (!check_build_chunks_small())
    {
        std::cerr << "FAIL: build chunks small\n";
        return 1;
    }
    if (!check_optimal_chunk_config())
    {
        std::cerr << "FAIL: optimal chunk config\n";
        return 1;
    }
    if (!check_estimate_clear_compress_speedup())
    {
        std::cerr << "FAIL: estimate clear compress speedup\n";
        return 1;
    }
    if (!check_should_use_clear_compress())
    {
        std::cerr << "FAIL: should use clear compress\n";
        return 1;
    }
    if (!check_numa_is_available())
    {
        std::cerr << "FAIL: numa is available\n";
        return 1;
    }
    if (!check_numa_get_current_node())
    {
        std::cerr << "FAIL: numa get current node\n";
        return 1;
    }
    return 0;
}
