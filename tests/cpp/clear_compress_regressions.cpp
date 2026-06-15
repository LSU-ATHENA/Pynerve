#include "nerve/persistence/memory/clear_and_compress.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

int main()
{
    using namespace nerve::persistence::clearcompress;

    ChunkConfig config;
    config.chunk_size = 2;

    const std::vector<std::vector<int>> columns{{1, 0, 1}, {}, {2}};
    auto chunks = buildChunks(columns, config);
    assert(chunks.size() == 2);
    assert(chunks[0].columns[0].indices.size() == 3);

    std::unordered_map<int, int> pivots;
    auto reduced = reduceChunk(chunks[0], pivots, config);
    assert(reduced.chunk_index == 0);
    assert(reduced.xor_operations >= 0);
    assert(std::isfinite(reduced.reduction_time_ms));
    assert(std::isfinite(reduced.memory_reduction));
    assert(chunks[0].is_compressed);

    auto result = reduceMatrixClearCompress(columns, config);
    assert(result.num_chunks == 2);
    assert(result.total_xor_operations >= 0);
    assert(std::isfinite(result.total_time_ms));
    assert(std::isfinite(result.chunk_build_time_ms));
    assert(std::isfinite(result.total_memory_reduction));

    bool rejected_zero_chunk_size = false;
    try
    {
        ChunkConfig invalid = config;
        invalid.chunk_size = 0;
        (void)buildChunks(columns, invalid);
    }
    catch (const std::invalid_argument &)
    {
        rejected_zero_chunk_size = true;
    }
    assert(rejected_zero_chunk_size);

    bool rejected_negative_build_row = false;
    try
    {
        (void)buildChunks({{0, -1}}, config);
    }
    catch (const std::invalid_argument &)
    {
        rejected_negative_build_row = true;
    }
    assert(rejected_negative_build_row);

    bool rejected_negative_reduce_row = false;
    try
    {
        Chunk bad_chunk;
        bad_chunk.chunk_index = 0;
        bad_chunk.start_column = 0;
        bad_chunk.end_column = 1;
        bad_chunk.columns.push_back(ChunkColumn{0, {0, -1}, false});
        (void)reduceChunk(bad_chunk, pivots, config);
    }
    catch (const std::invalid_argument &)
    {
        rejected_negative_reduce_row = true;
    }
    assert(rejected_negative_reduce_row);

    return 0;
}
