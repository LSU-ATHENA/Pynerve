
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
reduceHybridExact(const core::BufferView<const double> &points, size_t point_dim,
                  const HeterogeneousFastVR::Config &config,
                  const core::DeterminismContract &contract)
{
    auto engine_result = HeterogeneousFastVR::create(config);
    if (engine_result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(engine_result.errorCode());
    }
    return engine_result.value()->computeVrPersistence(points, point_dim, contract);
}

} // namespace nerve::persistence::accelerated
