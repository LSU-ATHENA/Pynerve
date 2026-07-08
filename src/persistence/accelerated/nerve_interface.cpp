
#include "nerve/persistence/accelerated/nerve_interface.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
NerveVRInterface::computeVrPersistenceNerve(core::BufferView<const double>points,
                                            size_t point_dim, const core::DeterminismContract &)
{
    try
    {
        return computeVrPersistenceFast(points, point_dim, config_);
    }
    catch (const std::exception &e)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                             e.what());
    }
}

} // namespace nerve::persistence::accelerated
