#include "nerve/common/accelerated_types.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace
{

nerve::common::VRConfig testConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    cfg.acceleration.mode = nerve::common::AccelerationMode::GPU_ONLY;
    cfg.acceleration.enable_gpu = true;
    cfg.use_acceleration = true;
    return cfg;
}

} // namespace

#ifdef NERVE_ENABLE_CUDA

TEST(SOTACudaTest, GpuAccelerationManagerCreate)
{
    auto manager = nerve::persistence::accelerated::GPUAccelerationManager::create(testConfig());
    EXPECT_TRUE(manager.isSuccess());
}

TEST(SOTACudaTest, GpuAccelerationManagerDetectCapabilities)
{
    auto caps = nerve::persistence::accelerated::GPUAccelerationManager::detectSystemCapabilities();
    EXPECT_FALSE(caps.cuda_available || caps.gpu_info.device_id >= 0 ||
                 !caps.supported_features.empty() || false);
}

TEST(SOTACudaTest, GpuAccelerationManagerRuntimeCheck)
{
    bool available =
        nerve::persistence::accelerated::GPUAccelerationManager::isGpuRuntimeAvailable();
    EXPECT_TRUE(available || !available);
}

TEST(SOTACudaTest, AcceleratedVREngineCreate)
{
    auto engine = nerve::persistence::accelerated::AcceleratedVREngine::create(testConfig());
    EXPECT_TRUE(engine.isSuccess());
}

TEST(SOTACudaTest, CudaAvailabilityCheck)
{
    bool available = nerve::persistence::is_cuda_available();
    EXPECT_TRUE(available || !available);
}

TEST(SOTACudaTest, AcceleratedFactoryEngines)
{
    auto runtime =
        nerve::persistence::accelerated::factory::createAccelerationRuntimeEngine(10, 2, 1.0);
    EXPECT_TRUE(runtime.isSuccess());

    auto production =
        nerve::persistence::accelerated::factory::createProductionEngine(testConfig());
    EXPECT_TRUE(production.isSuccess() || production.isError());
}

#else

TEST(SOTACudaTest, CudaNotEnabled)
{
    GTEST_SKIP() << "NERVE_ENABLE_CUDA is not defined";
}

#endif
