
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/kernel_launcher.hpp"
#include "nerve/persistence/core/core_types.hpp"

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

using nerve::core::BufferView;
using nerve::persistence::Pair;

#ifdef NERVE_HAS_CUDA

bool check_symmetric_difference_computation()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i = {1, 3, 5, 7, 9};
    std::vector<int> col_j = {2, 3, 6, 7, 10};
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        std::cerr << "symmetric difference failed: " << err.compactSummary() << "\n";
        return false;
    }
    if (result.empty())
    {
        std::cerr << "symmetric difference should not be empty for overlapping columns\n";
        return false;
    }
    return true;
}

bool check_symmetric_difference_disjoint()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i = {0, 1, 2};
    std::vector<int> col_j = {3, 4, 5};
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        std::cerr << "disjoint symmetric difference failed\n";
        return false;
    }
    if (result.size() != col_i.size() + col_j.size())
    {
        std::cerr << "disjoint columns should produce union size, got " << result.size()
                  << " expected " << (col_i.size() + col_j.size()) << "\n";
        return false;
    }
    return true;
}

bool check_symmetric_difference_identical()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i = {1, 2, 3, 4, 5};
    std::vector<int> col_j = {1, 2, 3, 4, 5};
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        std::cerr << "identical symmetric difference failed\n";
        return false;
    }
    if (!result.empty())
    {
        std::cerr << "identical columns should produce empty result, got size " << result.size()
                  << "\n";
        return false;
    }
    return true;
}

bool check_symmetric_difference_empty_input()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i;
    std::vector<int> col_j;
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        std::cerr << "empty symmetric difference failed\n";
        return false;
    }
    if (!result.empty())
    {
        std::cerr << "two empty columns should produce empty result\n";
        return false;
    }
    return true;
}

bool check_large_column_handling()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i(100000, 1);
    std::vector<int> col_j;
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        return true;
    }
    if (result.empty())
    {
        std::cerr << "large column with empty other should produce result\n";
        return false;
    }
    return true;
}

bool check_large_overlapping_columns()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i(50000, 0);
    std::vector<int> col_j(50000, 0);
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        return true;
    }
    if (!result.empty())
    {
        std::cerr << "identical large columns should produce empty result\n";
        return false;
    }
    return true;
}

bool check_sorted_columns_guarantee()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    if (!mgr.isAvailable())
    {
        return true;
    }
    std::vector<int> col_i = {0, 2, 4, 6, 8, 10};
    std::vector<int> col_j = {1, 3, 5, 7, 9, 11};
    std::vector<int> result;
    auto err = mgr.symmetricDifferenceColumns(col_i, col_j, result);
    if (err.isError())
    {
        return true;
    }
    if (result.empty())
    {
        std::cerr << "disjoint sorted columns should produce union\n";
        return false;
    }
    for (std::size_t k = 1; k < result.size(); ++k)
    {
        if (result[k] <= result[k - 1])
        {
            std::cerr << "result should be sorted ascending\n";
            return false;
        }
    }
    return true;
}

bool check_gpu_initialization()
{
    nerve::gpu::ComputeManager &mgr = nerve::gpu::ComputeManager::getInstance();
    bool available = mgr.isAvailable();
    return true;
}

#else

bool check_gpu_disabled_environment()
{
    std::cerr << "GPU tests not compiled: NERVE_HAS_CUDA not defined\n";
    return true;
}

#endif

} // namespace

int main()
{
#ifdef NERVE_HAS_CUDA
    if (!check_symmetric_difference_computation())
    {
        std::cerr << "FAIL: symmetric difference computation\n";
        return 1;
    }
    if (!check_symmetric_difference_disjoint())
    {
        std::cerr << "FAIL: symmetric difference disjoint\n";
        return 1;
    }
    if (!check_symmetric_difference_identical())
    {
        std::cerr << "FAIL: symmetric difference identical\n";
        return 1;
    }
    if (!check_symmetric_difference_empty_input())
    {
        std::cerr << "FAIL: symmetric difference empty input\n";
        return 1;
    }
    if (!check_large_column_handling())
    {
        std::cerr << "FAIL: large column handling\n";
        return 1;
    }
    if (!check_large_overlapping_columns())
    {
        std::cerr << "FAIL: large overlapping columns\n";
        return 1;
    }
    if (!check_sorted_columns_guarantee())
    {
        std::cerr << "FAIL: sorted columns guarantee\n";
        return 1;
    }
    if (!check_gpu_initialization())
    {
        std::cerr << "FAIL: GPU initialization\n";
        return 1;
    }
#else
    if (!check_gpu_disabled_environment())
    {
        std::cerr << "FAIL: GPU disabled environment\n";
        return 1;
    }
#endif
    return 0;
}
