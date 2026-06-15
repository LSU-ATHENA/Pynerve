
#include "nerve/gpu/distributed_tuning.hpp"
#include "nerve/gpu/hybrid_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace nerve::gpu::tuning
{

#if defined(NERVE_HAS_MPI)
#include "detail/tuning_distributed_coordinator_core.inl"
#include "detail/tuning_distributed_coordinator_mpi.inl"
#include "detail/tuning_distributed_multigpu.inl"
#endif // NERVE_HAS_MPI

} // namespace nerve::gpu::tuning
