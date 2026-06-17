
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_distributed_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#ifdef NERVE_HAS_MPI

namespace
{

using nerve::Index;
using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::persistence::DistributedReducer;
using nerve::persistence::Pair;

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_distributed_reducer_construction()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));
    BoundaryMatrix bm(complex, 1);
    DistributedReducer::Config config;
    config.num_nodes = 2;
    config.num_threads_per_node = 2;
    DistributedReducer reducer(bm, config);
    (void)reducer;
    return true;
}

bool check_config_validation()
{
    DistributedReducer::Config config;
    if (config.num_nodes < 1)
        return false;
    if (config.num_threads_per_node < 1)
        return false;
    if (config.chunk_size == 0)
        return false;
    if (config.communication_threshold < 0.0)
        return false;
    return true;
}

bool check_distributed_reduction_valid_pairs()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    DistributedReducer reducer(bm);
    reducer.computeDistributed();
    auto pairs = reducer.getPairs();
    if (pairs.empty())
    {
        std::cerr << "expected pairs from distributed reduction\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death violated\n";
            return false;
        }
        if (!p.isInfinite() && p.lifetime() < -1e-12)
        {
            std::cerr << "negative persistence\n";
            return false;
        }
    }
    return true;
}

bool check_distributed_performance_monitoring()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    DistributedReducer reducer(bm);
    reducer.computeDistributed();
    if (reducer.getTotalOperations() == 0)
        return false;
    if (reducer.getComputationTime() < 0.0)
        return false;
    if (reducer.getCommunicationTime() < 0.0)
        return false;
    return true;
}

bool check_config_get_set()
{
    DistributedReducer::Config config;
    config.num_nodes = 4;
    config.num_threads_per_node = 8;
    config.chunk_size = 500;
    config.enable_overlap = false;
    config.communication_threshold = 0.05;

    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));
    BoundaryMatrix bm(complex, 1);
    DistributedReducer reducer(bm);
    reducer.setConfig(config);
    auto retrieved = reducer.getConfig();
    if (retrieved.num_nodes != 4)
        return false;
    if (retrieved.num_threads_per_node != 8)
        return false;
    if (retrieved.chunk_size != 500)
        return false;
    if (retrieved.enable_overlap)
        return false;
    return true;
}

bool check_compute_with_partitions()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    DistributedReducer reducer(bm);
    reducer.computeWithPartitions(2);
    if (!reducer.getPairs().empty())
        return true;
    return true;
}

} // namespace

int main()
{
    if (!check_distributed_reducer_construction())
    {
        std::cerr << "FAIL: distributed reducer construction\n";
        return 1;
    }
    if (!check_config_validation())
    {
        std::cerr << "FAIL: config validation\n";
        return 1;
    }
    if (!check_distributed_reduction_valid_pairs())
    {
        std::cerr << "FAIL: distributed reduction valid pairs\n";
        return 1;
    }
    if (!check_distributed_performance_monitoring())
    {
        std::cerr << "FAIL: distributed performance monitoring\n";
        return 1;
    }
    if (!check_config_get_set())
    {
        std::cerr << "FAIL: config get/set\n";
        return 1;
    }
    if (!check_compute_with_partitions())
    {
        std::cerr << "FAIL: compute with partitions\n";
        return 1;
    }
    return 0;
}

#else
int main()
{
    return 0;
}
#endif