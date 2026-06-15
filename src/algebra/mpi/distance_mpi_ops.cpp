#include "nerve/config.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

namespace nerve::algebra::mpi
{

#ifdef NERVE_HAS_MPI

void distributedDistanceMatrix(const double *local_points, Size n_local, Size dim,
                               double *local_dist)
{
    int rank, size;
    int mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_rank failed in distributedDistanceMatrix" << std::endl;
        return;
    }
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_size failed in distributedDistanceMatrix" << std::endl;
        return;
    }

    std::vector<int> counts(size);
    mpi_err = MPI_Allgather(&n_local, 1, sizeof(Size) == 8 ? MPI_INT64_T : MPI_INT, counts.data(),
                            1, sizeof(Size) == 8 ? MPI_INT64_T : MPI_INT, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allgather failed in distributedDistanceMatrix" << std::endl;
        return;
    }

    std::vector<int> displs(size, 0);
    Size total = 0;
    for (int i = 0; i < size; ++i)
    {
        displs[i] = static_cast<int>(total * dim);
        total += counts[i];
    }

    std::vector<double> all_points(total * dim);
    mpi_err = MPI_Allgatherv(local_points, n_local * dim, MPI_DOUBLE, all_points.data(),
                             counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allgatherv failed in distributedDistanceMatrix" << std::endl;
        return;
    }

    for (Size i = 0; i < n_local; ++i)
    {
        for (Size j = 0; j < total; ++j)
        {
            double sum = 0.0;
            for (Size d = 0; d < dim; ++d)
            {
                double diff = local_points[i * dim + d] - all_points[j * dim + d];
                sum += diff * diff;
            }
            local_dist[i * total + j] = std::sqrt(sum);
        }
    }
}

#else

void distributedDistanceMatrix(const double *local, Size n, Size dim, double *dist)
{
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (Size d = 0; d < dim; ++d)
            {
                double diff = local[i * dim + d] - local[j * dim + d];
                sum += diff * diff;
            }
            dist[i * n + j] = std::sqrt(sum);
        }
    }
}

#endif

} // namespace nerve::algebra::mpi
