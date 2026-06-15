#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace
{

nerve::common::VRConfig testConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    return cfg;
}

} // namespace

#ifdef NERVE_ENABLE_REPRESENTATIVE_CYCLES

TEST(SOTARepresentativeCyclesTest, CycleComputerCreation)
{
    nerve::persistence::adaptive_acceleration::representative::RepresentativeConfig rconfig;
    rconfig.use_matrix_multiplication = true;
    rconfig.enable_parallel_computation = false;

    auto computer = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleComputer::create(rconfig);
    EXPECT_TRUE(computer.isSuccess());
}

TEST(SOTARepresentativeCyclesTest, ComputeFromSparseMatrix)
{
    const std::vector<double> dense{1.0, 0.0, 0.0, 1.0};
    auto matrix =
        nerve::persistence::adaptive_acceleration::SparseMatrix::fromDenseMatrix(dense, 2, 2);
    ASSERT_TRUE(matrix.isSuccess());

    nerve::persistence::adaptive_acceleration::representative::RepresentativeConfig rconfig;
    rconfig.use_matrix_multiplication = true;
    rconfig.enable_parallel_computation = false;

    auto computer = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleComputer::create(rconfig);
    ASSERT_TRUE(computer.isSuccess());

    auto cycles = computer.value()->computeRepresentativesFast(matrix.value());
    EXPECT_TRUE(cycles.isSuccess());
}

TEST(SOTARepresentativeCyclesTest, CycleHasValidProperties)
{
    const std::vector<double> dense{1.0, 0.5, 0.5, 1.0};
    auto matrix =
        nerve::persistence::adaptive_acceleration::SparseMatrix::fromDenseMatrix(dense, 2, 2);
    ASSERT_TRUE(matrix.isSuccess());

    nerve::persistence::adaptive_acceleration::representative::RepresentativeConfig rconfig;
    rconfig.use_matrix_multiplication = true;

    auto computer = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleComputer::create(rconfig);
    ASSERT_TRUE(computer.isSuccess());

    auto cycles_result = computer.value()->computeRepresentativesFast(matrix.value());
    ASSERT_TRUE(cycles_result.isSuccess());

    for (const auto &cycle : cycles_result.value())
    {
        EXPECT_GE(cycle.dimension, 0);
        EXPECT_TRUE(std::isfinite(cycle.birth_time));
        EXPECT_GE(cycle.death_time, cycle.birth_time);
        EXPECT_FALSE(cycle.vertices.empty());
        EXPECT_EQ(cycle.vertices.size(), cycle.coefficients.size());
    }
}

TEST(SOTARepresentativeCyclesTest, CycleFactoryCreation)
{
    nerve::persistence::adaptive_acceleration::representative::RepresentativeConfig rconfig;
    rconfig.enable_parallel_computation = false;

    auto fast = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleFactory::createFast(rconfig);
    EXPECT_TRUE(fast.isSuccess());

    auto parallel = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleFactory::createParallel(rconfig);
    EXPECT_TRUE(parallel.isSuccess());

    auto viz = nerve::persistence::adaptive_acceleration::representative::
        RepresentativeCycleFactory::createForVisualization(rconfig);
    EXPECT_TRUE(viz.isSuccess());
}

TEST(SOTARepresentativeCyclesTest, CycleValidatorValidates)
{
    const std::vector<double> dense{1.0, 0.0, 0.0, 1.0};
    auto matrix =
        nerve::persistence::adaptive_acceleration::SparseMatrix::fromDenseMatrix(dense, 2, 2);
    ASSERT_TRUE(matrix.isSuccess());

    nerve::persistence::adaptive_acceleration::representative::Cycle cycle;
    cycle.vertices = {0, 1};
    cycle.coefficients = {1.0, 1.0};
    cycle.dimension = 1;
    cycle.birth_time = 0.0;
    cycle.death_time = 1.0;

    EXPECT_TRUE(cycle.isValid());

    auto valid =
        nerve::persistence::adaptive_acceleration::representative::CycleValidator::validateCycle(
            cycle, matrix.value());
    EXPECT_TRUE(valid.isSuccess());
}

#else

TEST(SOTARepresentativeCyclesTest, RepresentativeCyclesNotEnabled)
{
    GTEST_SKIP() << "NERVE_ENABLE_REPRESENTATIVE_CYCLES is not defined";
}

#endif
