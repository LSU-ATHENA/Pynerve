#include <cstdint>
#include <cstring>
#include <vector>

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>
#endif
#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

#include "nerve/acceleration_fwd.hpp"

namespace nerve::runtime::mpi
{
void broadcastCalibration(double *p, Size n, int r)
{
    if (!p || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Bcast(p, static_cast<int>(n), MPI_DOUBLE, r, MPI_COMM_WORLD);
#else
    (void)r;
#endif
}
} // namespace nerve::runtime::mpi
