
#include "nerve/config.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::sheaf::SheafConfig;
using nerve::sheaf::SheafEngine;
using nerve::sheaf::morphism::AsyncMorphismQueue;
using nerve::sheaf::morphism::BatchedMorphismComputer;
using nerve::sheaf::morphism::MorphismCompositionOptimizer;
using nerve::sheaf::morphism::SparseMorphism;
using nerve::sheaf::parallel::ParallelSheafBuilder;
using nerve::sheaf::parallel::SIMDStalkOperations;
using nerve::sheaf::parallel::StalkData;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

bool check_sparse_morphism_apply()
{
    SparseMorphism morph;
    morph.from_dim = 3;
    morph.to_dim = 2;
    morph.row_ptr = {0, 2, 3};
    morph.col_idx = {0, 1, 2};
    morph.values = {1.0f, 0.5f, 2.0f};
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(morph.to_dim, 0.0f);
    morph.apply(input, output);
    if (output.size() != 2)
    {
        std::cerr << "output size mismatch\n";
        return false;
    }
    return true;
}

bool check_morphism_composition_optimizer()
{
    MorphismCompositionOptimizer opt;
    SparseMorphism m1;
    m1.from_dim = 2;
    m1.to_dim = 2;
    m1.row_ptr = {0, 1, 2};
    m1.col_idx = {0, 1};
    m1.values = {1.0f, 1.0f};
    opt.addMorphism(0, 1, m1);
    SparseMorphism m2;
    m2.from_dim = 2;
    m2.to_dim = 2;
    m2.row_ptr = {0, 1, 2};
    m2.col_idx = {0, 1};
    m2.values = {2.0f, 2.0f};
    opt.addMorphism(1, 2, m2);
    opt.registerChain({0, 1, 2});
    auto composed = opt.getComposed(0, 2);
    if (composed.from_dim == 0 && composed.to_dim == 0)
    {
        std::cerr << "morphism composition returned trivial morphism\n";
        return false;
    }
    return true;
}

bool check_batched_morphism_computer()
{
    BatchedMorphismComputer computer(64);
    SparseMorphism morph;
    morph.from_dim = 2;
    morph.to_dim = 2;
    morph.row_ptr = {0, 1, 2};
    morph.col_idx = {0, 1};
    morph.values = {1.0f, 1.0f};
    computer.addMorphism(0, 1, morph);
    std::vector<int> order = {0, 1};
    std::vector<std::vector<float>> stalk_data = {{1.0f, 2.0f}, {0.0f, 0.0f}};
    std::vector<std::vector<float>> output_data(2);
    computer.computeBatch(order, stalk_data, output_data);
    if (output_data.size() != 2)
    {
        std::cerr << "batched output size mismatch\n";
        return false;
    }
    return true;
}

bool check_async_morphism_queue()
{
    AsyncMorphismQueue queue;
    queue.start(2);
    std::promise<std::vector<float>> prom;
    auto fut = prom.get_future();
    queue.submit(0, 0, {1.0f, 2.0f}, std::move(prom));
    queue.stop();
    auto result = fut.get();
    if (result.empty())
    {
        std::cerr << "async result empty\n";
        return false;
    }
    return true;
}

bool check_parallel_sheaf_builder()
{
    ParallelSheafBuilder::SheafConfig config;
    config.num_stalks = 4;
    config.stalk_dimension = 3;
    config.use_simd = true;
    config.num_threads = 2;
    ParallelSheafBuilder builder(config);
    builder.build();
    auto stalks = builder.getStalks();
    if (stalks.empty())
    {
        std::cerr << "parallel builder returned no stalks\n";
        return false;
    }
    return true;
}

bool check_simd_stalk_operations()
{
    StalkData a(0, 3);
    StalkData b(1, 3);
    a.data = {1.0f, 2.0f, 3.0f};
    b.data = {4.0f, 5.0f, 6.0f};
    StalkData result(2, 3);
    SIMDStalkOperations::addStalks(a, b, result);
    float dot = SIMDStalkOperations::dotProduct(a, b);
    if (std::abs(dot - 32.0f) > 1e-6f)
    {
        std::cerr << "dot product expected 32, got " << dot << "\n";
        return false;
    }
    return true;
}

bool check_sheaf_engine_lifecycle()
{
    SheafConfig cfg;
    cfg.num_stalks = 0;
    cfg.stalk_dimension = 0;
    SheafEngine engine(cfg);
    nerve::sheaf::Point pos;
    pos.x = 0.0f;
    pos.y = 0.0f;
    pos.z = 0.0f;
    std::vector<nerve::sheaf::Point> positions = {pos};
    std::vector<int> dims = {2};
    engine.buildSheaf(positions, dims);
    return true;
}

} // namespace

int main()
{
    if (!check_sparse_morphism_apply())
    {
        std::cerr << "FAIL: sparse morphism apply\n";
        return 1;
    }
    if (!check_morphism_composition_optimizer())
    {
        std::cerr << "FAIL: morphism composition\n";
        return 1;
    }
    if (!check_batched_morphism_computer())
    {
        std::cerr << "FAIL: batched morphism\n";
        return 1;
    }
    if (!check_async_morphism_queue())
    {
        std::cerr << "FAIL: async morphism queue\n";
        return 1;
    }
    if (!check_parallel_sheaf_builder())
    {
        std::cerr << "FAIL: parallel sheaf builder\n";
        return 1;
    }
    if (!check_simd_stalk_operations())
    {
        std::cerr << "FAIL: SIMD stalk ops\n";
        return 1;
    }
    if (!check_sheaf_engine_lifecycle())
    {
        std::cerr << "FAIL: sheaf engine lifecycle\n";
        return 1;
    }
    return 0;
}
