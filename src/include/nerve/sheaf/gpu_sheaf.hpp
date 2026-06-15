
#pragma once

#include "nerve/core.hpp"

#include <future>
#include <memory>
#include <vector>

namespace nerve::sheaf
{

struct Point
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

namespace gpu
{

struct SheafGPUBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_stalks;
    int stalk_dim;
};

SheafGPUBenchmark benchmarkSheafGPU(int num_stalks, int stalk_dim);

} // namespace gpu

namespace parallel
{

struct alignas(64) StalkData
{
    int id = 0;
    int dimension = 0;
    std::vector<float> data;

    StalkData() = default;

    StalkData(int stalk_id, int stalk_dim)
        : id(stalk_id)
        , dimension(stalk_dim > 0 ? stalk_dim : 0)
        , data(static_cast<size_t>(dimension), 0.0f)
    {}
};

class StalkSpatialHash
{
public:
    explicit StalkSpatialHash(float cell_size = 1.0f);
    ~StalkSpatialHash();

    void insertStalk(int stalk_id, const Point &position);
    std::vector<int> getNearbyStalks(const Point &position) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class ParallelSheafBuilder
{
public:
    struct SheafConfig
    {
        int num_stalks = 0;
        int stalk_dimension = 0;
        bool use_simd = true;
        int num_threads = 0;
    };

    explicit ParallelSheafBuilder(const SheafConfig &config);
    ~ParallelSheafBuilder();

    void build();
    std::vector<struct StalkData> getStalks() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class SIMDStalkOperations
{
public:
    static void addStalks(const struct StalkData &a, const struct StalkData &b,
                          struct StalkData &result);

    static void scaleStalk(const struct StalkData &stalk, float scalar, struct StalkData &result);

    static float dotProduct(const struct StalkData &a, const struct StalkData &b);

    static void normalizeStalk(struct StalkData &stalk);
};

struct SheafParallelBenchmark
{
    double sequential_time_ms = 0.0;
    double parallel_time_ms = 0.0;
    double simd_time_ms = 0.0;
    double speedup_parallel = 0.0;
    double speedup_simd = 0.0;
    int num_stalks = 0;
    int stalk_dim = 0;
    int num_threads = 0;
};

SheafParallelBenchmark benchmarkParallelSheaf(int num_stalks, int stalk_dim, int num_threads);

} // namespace parallel

namespace morphism
{

struct SparseMorphism
{
    int from_dim = 0;
    int to_dim = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;

    void apply(const std::vector<float> &input, std::vector<float> &output) const;
    void applySIMD(const float *input, float *output) const;
};

class MorphismMemoryPool
{
public:
    explicit MorphismMemoryPool(size_t initial_size = 1024ULL * 1024ULL);
    ~MorphismMemoryPool();

    float *allocateMorphism(int nnz);
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class BatchedMorphismComputer
{
public:
    explicit BatchedMorphismComputer(int cache_block_size = 256);
    ~BatchedMorphismComputer();

    void addMorphism(int from_stalk, int to_stalk, const SparseMorphism &morphism);

    void computeBatch(const std::vector<int> &stalk_order,
                      const std::vector<std::vector<float>> &stalk_data,
                      std::vector<std::vector<float>> &output_data);

    void computeBatchSIMD(const std::vector<int> &stalk_order,
                          const std::vector<float *> &stalk_data, std::vector<float *> &output_data,
                          int num_stalks);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class MorphismCompositionOptimizer
{
public:
    MorphismCompositionOptimizer();
    ~MorphismCompositionOptimizer();

    void registerChain(const std::vector<int> &stalk_chain);
    SparseMorphism getComposed(int from, int to) const;
    void addMorphism(int from, int to, const SparseMorphism &morphism);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class AsyncMorphismQueue
{
public:
    AsyncMorphismQueue();
    ~AsyncMorphismQueue();

    void submit(int from_stalk, int to_stalk, const std::vector<float> &input,
                std::promise<std::vector<float>> promise);

    void start(int num_workers = 2);
    void stop();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace morphism

struct SheafConfig
{
    int num_stalks = 0;
    int stalk_dimension = 0;
    bool use_parallel = true;
    bool use_simd = true;
    int num_threads = 0;
    int gpu_batch_size = 256;
    bool use_memory_pool = true;
    size_t memory_pool_size = 64ULL * 1024ULL * 1024ULL;
};

struct SheafResult
{
    std::vector<float> cohomology;
    double computation_time_ms = 0.0;
    bool success = false;
};

class SheafEngine
{
public:
    explicit SheafEngine(const SheafConfig &config = {});
    ~SheafEngine();

    void buildSheaf(const std::vector<Point> &stalk_positions,
                    const std::vector<int> &stalk_dimensions);

    SheafResult computeCohomology(const std::vector<float> &cocycle);

    void applyMorphism(int from_stalk, int to_stalk, const std::vector<float> &input,
                       std::vector<float> &output);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct SheafHardwareInfo
{
    bool has_gpu = false;
    bool has_avx512 = false;
    bool has_avx2 = false;
    int num_cores = 0;
};

SheafHardwareInfo detectSheafHardware();

} // namespace nerve::sheaf
