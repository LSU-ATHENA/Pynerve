
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/detail/vr_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <vector>

namespace
{

bool check_engine_config_defaults()
{
    nerve::common::VRConfig config;
    if (config.max_dim != 2)
        return false;
    if (config.max_radius != 1.0)
        return false;
    if (config.acceleration.mode != nerve::common::AccelerationMode::CPU_ONLY)
        return false;

        return false;
    return true;
}

bool check_factory_creates_engine()
{
    auto engine =
        nerve::persistence::accelerated::factory::createAccelerationRuntimeEngine(100, 3, 1.0);
    if (engine.isError())
        return false;
    if (!engine.value())
        return false;
    auto prod =
        nerve::persistence::accelerated::factory::createProductionEngine(nerve::common::VRConfig{});
    if (prod.isError())
        return false;
    return true;
}

bool check_impl_helpers()
{
    using nerve::persistence::accelerated::detail::bytesToMb;
    using nerve::persistence::accelerated::detail::estimateProblemOps;
    if (bytesToMb(1048576) != 1.0)
        return false;
    if (estimateProblemOps(0, 3, 2) != 0.0)
        return false;
    if (estimateProblemOps(3, 2, 2) <= 0.0)
        return false;
    return true;
}

bool check_runtime_initialization()
{
    using nerve::persistence::getOptimalFastvrConfig;
    auto cfg = getOptimalFastvrConfig(50, 3);
    if (cfg.max_dim != 2)
        return false;
    if (cfg.max_radius != 1.0)
        return false;
    auto cfg2 = getOptimalFastvrConfig(1000, 3);
    if (cfg2.algorithm != nerve::common::VRAlgorithmSelection::EXACT_STANDARD)
        return false;
    return true;
}

bool check_vr_computation_non_empty()
{
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    nerve::core::BufferView<const double> view(pts.data(), pts.size());
    nerve::common::VRConfig config;
    config.max_dim = 1;
    config.max_radius = 2.0;
    auto result = nerve::persistence::accelerated::computeVrPersistenceFast(view, 2, config);
    if (result.isError())
        return false;
    return !result.value().empty();
}

bool check_saturated_helpers()
{
    namespace acc = nerve::persistence::accelerated::utils;
    if (acc::estimateMemoryRequirements(10, 3, 2, nerve::common::VRConfig{}) == 0)
        return false;
    if (acc::isAccelerationBeneficial(0, 2, 1.0))
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_engine_config_defaults())
        return 1;
    if (!check_factory_creates_engine())
        return 1;
    if (!check_impl_helpers())
        return 1;
    if (!check_runtime_initialization())
        return 1;
    if (!check_vr_computation_non_empty())
        return 1;
    if (!check_saturated_helpers())
        return 1;
    return 0;
}
