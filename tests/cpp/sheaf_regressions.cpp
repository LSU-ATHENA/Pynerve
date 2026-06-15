
#include "nerve/config.hpp"
#include "nerve/sheaf/sheaf_laplacian.hpp"
#include "nerve/sheaf/sheaf_learning.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Size;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)

using nerve::sheaf::SheafLaplacianFactory;
using nerve::sheaf::SheafLaplacianRuntime;

bool check_sheaf_construction()
{
    SheafLaplacianRuntime::SheafConfig config;
    config.num_attributes = 2;
    SheafLaplacianRuntime runtime(config);
    SheafLaplacianRuntime::SheafNode node;
    node.node_id = 0;
    node.position = {0.0, 0.0};
    node.attributes = {1.0, 2.0};
    node.attribute_names = {"a", "b"};
    node.neighbors = {1};
    node.edge_weights = {1.0};
    node.stalk_values = {1.0, 1.0};
    node.restriction_maps = {1.0, 0.0, 0.0, 1.0};
    node.local_laplacian = 0.0;
    runtime.addNode(node);
    if (!runtime.validateSheafStructure())
    {
        std::cerr << "sheaf structure validation failed\n";
        return false;
    }
    return true;
}

bool check_sheaf_laplacian_computation()
{
    SheafLaplacianRuntime::SheafConfig config;
    config.num_attributes = 1;
    SheafLaplacianRuntime runtime(config);
    for (int i = 0; i < 3; ++i)
    {
        SheafLaplacianRuntime::SheafNode node;
        node.node_id = static_cast<uint32_t>(i);
        node.position = {static_cast<double>(i), 0.0};
        node.attributes = {static_cast<double>(i)};
        node.attribute_names = {"val"};
        node.local_laplacian = 0.0;
        if (i > 0)
        {
            node.neighbors = {static_cast<uint32_t>(i - 1)};
            node.edge_weights = {1.0};
            node.stalk_values = {1.0};
            node.restriction_maps = {1.0};
        }
        runtime.addNode(node);
    }
    auto result = runtime.buildSheafLaplacian();
    if (result.matrix_size == 0)
    {
        std::cerr << "sheaf laplacian matrix is empty\n";
        return false;
    }
    if (!std::isfinite(result.frobenius_norm))
    {
        std::cerr << "frobenius norm not finite\n";
        return false;
    }
    return true;
}

bool check_sheaf_morphism_create()
{
    SheafLaplacianFactory::FactoryConfig factory_config;
    factory_config.default_type = SheafLaplacianFactory::SheafType::PRODUCT_SHEAF;
    SheafLaplacianFactory factory(factory_config);
    SheafLaplacianRuntime::SheafConfig config;
    auto ptr = factory.createProductSheaf(config);
    if (!ptr)
    {
        std::cerr << "factory returned null\n";
        return false;
    }
    return true;
}

bool check_sheaf_isStable()
{
    SheafLaplacianRuntime::SheafConfig config;
    SheafLaplacianRuntime runtime(config);
    if (!runtime.isStable())
    {
        std::cerr << "default runtime should be stable\n";
        return false;
    }
    return true;
}

bool check_sheaf_numerical_residual()
{
    SheafLaplacianRuntime::SheafConfig config;
    config.numerical_tolerance = 1e-12;
    SheafLaplacianRuntime runtime(config);
    double residual = runtime.getNumericalResidual();
    if (!std::isfinite(residual))
    {
        std::cerr << "residual must be finite\n";
        return false;
    }
    return true;
}

#endif

} // namespace

int main()
{
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
    if (!check_sheaf_construction())
    {
        std::cerr << "FAIL: sheaf construction\n";
        return 1;
    }
    if (!check_sheaf_laplacian_computation())
    {
        std::cerr << "FAIL: sheaf laplacian\n";
        return 1;
    }
    if (!check_sheaf_morphism_create())
    {
        std::cerr << "FAIL: sheaf morphism\n";
        return 1;
    }
    if (!check_sheaf_isStable())
    {
        std::cerr << "FAIL: sheaf isStable\n";
        return 1;
    }
    if (!check_sheaf_numerical_residual())
    {
        std::cerr << "FAIL: sheaf residual\n";
        return 1;
    }
#endif
    return 0;
}
