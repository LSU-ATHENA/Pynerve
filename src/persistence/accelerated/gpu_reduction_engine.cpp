
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
reduceFiltrationCpu(const core::BufferView<const double> &points, size_t point_dim,
                    const VRConfig &config)
{
    return computeVrPersistenceFast(points, point_dim, config);
}

} // namespace nerve::persistence::accelerated
