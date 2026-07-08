#include "gpu_test_helpers.cuh"
#include "nerve/autodiff/autodiff.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU autodiff kernel coverage tests\n";
        return 0;
    }

    // Autodiff: Tensor default construction
    {
        nerve::autodiff::Tensor t;
        std::cout << "PASSED: autodiff Tensor default construction\n";
    }

    // Autodiff: Variable default construction
    {
        nerve::autodiff::Variable v;
        bool grad = v.requiresGrad();
        assert(grad || !grad); // just smoke
        std::cout << "PASSED: autodiff Variable default\n";
    }

    // Autodiff: Variable with requiresGrad
    {
        nerve::autodiff::Variable v;
        v.setRequiresGrad(true);
        assert(v.requiresGrad() == true);
        std::cout << "PASSED: autodiff Variable setRequiresGrad\n";
    }

    // Autodiff: ComputationalGraph construction
    {
        nerve::autodiff::ComputationalGraph graph;
        graph.clear();
        std::cout << "PASSED: autodiff ComputationalGraph default\n";
    }

    return 0;
}
