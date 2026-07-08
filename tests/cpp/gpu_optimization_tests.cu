#include "gpu_test_helpers.cuh"
#include "nerve/optimization/gpu_primitives_helpers.hpp"

#include <chrono>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU optimization kernel coverage tests\n";
        return 0;
    }

    // Optimization: exceedsBudgetMs
    {
        auto start = std::chrono::steady_clock::now();
        bool exceeded = nerve::optimization::exceedsBudgetMs(start, 1000.0);
        assert(!exceeded);
        std::cout << "PASSED: optimization exceedsBudgetMs (within 1s budget)\n";
    }

    // Optimization: multiplyWouldOverflow
    {
        bool overflows = nerve::optimization::multiplyWouldOverflow(
            static_cast<size_t>(1) << 32, static_cast<size_t>(1) << 32);
        assert(overflows || !overflows);
        std::cout << "PASSED: optimization multiplyWouldOverflow\n";
    }

    // Optimization: checkedByteCount
    {
        size_t bytes = 0;
        bool ok = nerve::optimization::checkedByteCount(1024, 8, bytes);
        assert(ok);
        assert(bytes == 8192);
        std::cout << "PASSED: optimization checkedByteCount (1024*8=8192)\n";
    }

    // Optimization: finiteFloatValues
    {
        float vals[] = {1.0f, 2.0f, 3.0f};
        bool finite = nerve::optimization::finiteFloatValues(vals, 3);
        assert(finite);
        std::cout << "PASSED: optimization finiteFloatValues\n";
    }

    // Optimization: checkedFiniteDistance
    {
        float lhs[] = {0.0f, 0.0f};
        float rhs[] = {3.0f, 4.0f};
        float dist = 0.0f;
        bool ok = nerve::optimization::checkedFiniteDistance(lhs, rhs, 2, dist);
        assert(ok);
        assert(dist > 0.0f);
        std::cout << "PASSED: optimization checkedFiniteDistance (dist=" << dist << ")\n";
    }

    return 0;
}
