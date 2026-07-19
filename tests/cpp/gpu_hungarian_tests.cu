#include "gpu_test_helpers.cuh"
#include "math/persistence_metrics/hungarian_solver.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU hungarian kernel coverage tests\n";
        return 0;
    }

    // Hungarian: basic solve on 2x2 cost matrix
    {
        std::vector<std::vector<double>> cost = {{1.0, 2.0}, {3.0, 4.0}};
        nerve::math::HungarianSolver solver(cost);
        auto result = solver.solve();
        assert(!result.isError());
        auto &assignment = result.value();
        assert(assignment.num_assigned == 2);
        assert(assignment.total_cost > 0.0);
        assert(assignment.pairs.size() == 2);
        std::cout << "PASSED: hungarian solve (2x2, cost=" << assignment.total_cost << ")\n";
    }

    // Hungarian: solve on 3x3 identity-ish matrix
    {
        std::vector<std::vector<double>> cost = {
            {1.0, 10.0, 10.0}, {10.0, 1.0, 10.0}, {10.0, 10.0, 1.0}};
        nerve::math::HungarianSolver solver(cost);
        auto result = solver.solve();
        assert(!result.isError());
        auto &assignment = result.value();
        assert(assignment.num_assigned == 3);
        assert(assignment.total_cost < 10.0); // diagonal assignment
        std::cout << "PASSED: hungarian solve (3x3 diagonal, cost=" << assignment.total_cost
                  << ")\n";
    }

    // Hungarian: solve on 4x3 rectangular matrix
    {
        std::vector<std::vector<double>> cost = {
            {2.0, 3.0, 4.0}, {3.0, 2.0, 5.0}, {1.0, 6.0, 3.0}, {4.0, 2.0, 1.0}};
        nerve::math::HungarianSolver solver(cost);
        auto result = solver.solve();
        assert(!result.isError());
        auto &assignment = result.value();
        assert(assignment.num_assigned > 0);
        assert(assignment.pairs.size() == assignment.num_assigned);
        std::cout << "PASSED: hungarian solve (4x3, " << assignment.num_assigned << " assigned)\n";
    }

    // Hungarian: solve on 1x1 singleton
    {
        std::vector<std::vector<double>> cost = {{5.0}};
        nerve::math::HungarianSolver solver(cost);
        auto result = solver.solve();
        assert(!result.isError());
        auto &assignment = result.value();
        assert(assignment.num_assigned == 1);
        assert(assignment.total_cost == 5.0);
        std::cout << "PASSED: hungarian solve (1x1, cost=5.0)\n";
    }

    return 0;
}
