#include "nerve/core_types.hpp"
#include "nerve/sheaf/detail/sheaf_detail.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"
#include "nerve/sheaf/sheaf_laplacian.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::sheaf::detectSheafHardware;
using nerve::sheaf::Point;
using nerve::sheaf::SheafConfig;
using nerve::sheaf::SheafEngine;

bool check_engine_basic_construction()
{
    SheafConfig config;
    config.num_stalks = 4;
    config.stalk_dimension = 3;
    config.num_threads = 2;
    SheafEngine engine(config);
    std::vector<Point> positions = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    std::vector<int> dims = {3, 3, 3, 3};
    engine.buildSheaf(positions, dims);
    std::vector<float> cocycle(12, 1.0f);
    auto result = engine.computeCohomology(cocycle);
    if (!result.success)
    {
        std::cerr << "engine computation did not succeed\n";
        return false;
    }
    if (result.cohomology.size() != 12)
    {
        std::cerr << "engine cohomology size mismatch\n";
        return false;
    }
    return true;
}

bool check_morphism_optimizer_basic_config()
{
    using nerve::sheaf::morphism::MorphismCompositionOptimizer;
    using nerve::sheaf::morphism::MorphismMemoryPool;
    using nerve::sheaf::morphism::SparseMorphism;

    SparseMorphism f;
    f.from_dim = 2;
    f.to_dim = 3;
    f.row_ptr = {0, 2, 3, 4};
    f.col_idx = {0, 1, 0, 1};
    f.values = {1.0f, 0.0f, 0.0f, 1.0f};

    SparseMorphism g;
    g.from_dim = 3;
    g.to_dim = 2;
    g.row_ptr = {0, 1, 2};
    g.col_idx = {0, 2};
    g.values = {1.0f, 1.0f};

    MorphismCompositionOptimizer opt;
    opt.addMorphism(0, 1, f);
    opt.addMorphism(1, 2, g);
    opt.registerChain({0, 1, 2});
    auto composed = opt.getComposed(0, 2);
    if (composed.from_dim != 2 || composed.to_dim != 2)
    {
        std::cerr << "morphism composition dimension mismatch\n";
        return false;
    }
    if (composed.row_ptr.empty())
    {
        std::cerr << "morphism composition produced empty result\n";
        return false;
    }

    MorphismMemoryPool pool(4096);
    float *allocated = pool.allocateMorphism(16);
    if (allocated == nullptr)
    {
        std::cerr << "morphism pool allocation returned null\n";
        return false;
    }
    return true;
}

bool check_parallel_sheaf_builder()
{
    using nerve::sheaf::parallel::ParallelSheafBuilder;
    using nerve::sheaf::parallel::SIMDStalkOperations;
    using nerve::sheaf::parallel::StalkData;

    ParallelSheafBuilder::SheafConfig builder_config;
    builder_config.num_stalks = 8;
    builder_config.stalk_dimension = 4;
    builder_config.num_threads = 2;
    ParallelSheafBuilder builder(builder_config);
    builder.build();
    auto stalks = builder.getStalks();
    if (stalks.size() != 8)
    {
        std::cerr << "parallel builder stalk count mismatch\n";
        return false;
    }
    for (const auto &s : stalks)
    {
        if (s.data.size() != 4)
        {
            std::cerr << "parallel builder stalk dimension mismatch\n";
            return false;
        }
    }

    StalkData a(0, 4);
    StalkData b(1, 4);
    a.data = {1.0f, 2.0f, 3.0f, 4.0f};
    b.data = {5.0f, 6.0f, 7.0f, 8.0f};
    StalkData sum(2, 4);
    SIMDStalkOperations::addStalks(a, b, sum);
    if (std::abs(sum.data[0] - 6.0f) > 1e-6f || std::abs(sum.data[3] - 12.0f) > 1e-6f)
    {
        std::cerr << "SIMD stalk addition produced wrong result\n";
        return false;
    }

    float dp = SIMDStalkOperations::dotProduct(a, b);
    float expected = 1.0f * 5.0f + 2.0f * 6.0f + 3.0f * 7.0f + 4.0f * 8.0f;
    if (std::abs(dp - expected) > 1e-6f)
    {
        std::cerr << "SIMD dot product mismatch\n";
        return false;
    }
    return true;
}

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
bool check_laplacian_factory_config_validation()
{
    using nerve::sheaf::SheafLaplacianFactory;
    using nerve::sheaf::SheafLaplacianRuntime;

    SheafLaplacianRuntime::SheafConfig laplacian_config;
    laplacian_config.num_attributes = 2;
    laplacian_config.attribute_weight = 1.0;
    laplacian_config.topological_weight = 1.0;
    laplacian_config.numerical_tolerance = 1e-12;

    SheafLaplacianFactory::FactoryConfig factory_config;
    SheafLaplacianFactory factory(factory_config);
    if (!factory.validateConfig(laplacian_config))
    {
        std::cerr << "valid laplacian config failed validation\n";
        return false;
    }

    auto errors = factory.getConfigErrors(laplacian_config);
    if (!errors.empty())
    {
        std::cerr << "valid laplacian config produced errors\n";
        return false;
    }

    auto runtime = factory.createProductSheaf(laplacian_config);
    if (!runtime)
    {
        std::cerr << "factory returned null for product sheaf\n";
        return false;
    }
    return true;
}

bool check_laplacian_matrix_dimensions()
{
    using nerve::sheaf::SheafLaplacianRuntime;

    SheafLaplacianRuntime::SheafConfig config;
    config.num_attributes = 1;
    config.attribute_weight = 0.5;
    config.topological_weight = 0.5;
    config.numerical_tolerance = 1e-12;
    SheafLaplacianRuntime runtime(config);

    for (uint32_t i = 0; i < 3; ++i)
    {
        SheafLaplacianRuntime::SheafNode node;
        node.node_id = i;
        node.position = {static_cast<double>(i), 0.0};
        node.attributes = {static_cast<double>(i)};
        node.attribute_names = {"val"};
        node.local_laplacian = 0.0;
        node.neighbors = {};
        node.edge_weights = {};
        node.stalk_values = {1.0};
        node.restriction_maps = {};
        if (i > 0)
        {
            node.neighbors = {i - 1};
            node.edge_weights = {1.0};
            node.stalk_values = {1.0};
            node.restriction_maps = {1.0};
        }
        runtime.addNode(node);
    }
    auto result = runtime.buildSheafLaplacian();
    if (result.matrix_size == 0)
    {
        std::cerr << "laplacian matrix has zero size\n";
        return false;
    }
    if (!std::isfinite(result.frobenius_norm))
    {
        std::cerr << "laplacian frobenius norm not finite\n";
        return false;
    }
    if (!result.isStable)
    {
        std::cerr << "laplacian result should be stable by default\n";
        return false;
    }
    return true;
}
#endif

} // namespace

int main()
{
    if (!check_engine_basic_construction())
    {
        std::cerr << "FAIL: engine basic construction\n";
        return 1;
    }
    if (!check_morphism_optimizer_basic_config())
    {
        std::cerr << "FAIL: morphism optimizer basic config\n";
        return 1;
    }
    if (!check_parallel_sheaf_builder())
    {
        std::cerr << "FAIL: parallel sheaf builder\n";
        return 1;
    }
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
    if (!check_laplacian_factory_config_validation())
    {
        std::cerr << "FAIL: laplacian factory config validation\n";
        return 1;
    }
    if (!check_laplacian_matrix_dimensions())
    {
        std::cerr << "FAIL: laplacian matrix dimensions\n";
        return 1;
    }
#endif
    return 0;
}
