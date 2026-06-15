#include "nerve/config.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

namespace nerve::filtration::mpi
{

#ifdef NERVE_HAS_MPI

static int rank_ = 0;
static int size_ = 1;

void initFiltrationMPI(int *argc, char ***argv)
{
    int mpi_rc = MPI_Init(argc, argv);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in initFiltrationMPI (MPI_Init): %s\n", buf);
        return;
    }
    mpi_rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in initFiltrationMPI (MPI_Comm_rank): %s\n", buf);
        return;
    }
    mpi_rc = MPI_Comm_size(MPI_COMM_WORLD, &size_);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in initFiltrationMPI (MPI_Comm_size): %s\n", buf);
        return;
    }
}

int mpiRank()
{
    return rank_;
}
int mpiSize()
{
    return size_;
}

std::vector<double> gatherFiltrationValues(const std::vector<double> &local, int root)
{
    int n = static_cast<int>(local.size());
    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;

    if (root == mpiRank())
    {
        counts.resize(mpiSize());
        displs.resize(mpiSize());
    }

    int mpi_err = MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Gather failed in gatherFiltrationValues" << std::endl;
        return {};
    }

    if (root == mpiRank())
    {
        for (int i = 0; i < mpiSize(); ++i)
        {
            displs[i] = total;
            total += counts[i];
        }
    }

    mpi_err = MPI_Bcast(&total, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Bcast failed in gatherFiltrationValues" << std::endl;
        return {};
    }
    std::vector<double> all_values(total);
    mpi_err = MPI_Gatherv(local.data(), n, MPI_DOUBLE, all_values.data(), counts.data(),
                          displs.data(), MPI_DOUBLE, root, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Gatherv failed in gatherFiltrationValues" << std::endl;
        return {};
    }
    return all_values;
}

std::vector<Pair> gatherPairsMPI(const std::vector<Pair> &local_pairs)
{
    int nlocal = static_cast<int>(local_pairs.size());

    int total_pairs = 0;
    int mpi_rc = MPI_Allreduce(&nlocal, &total_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in gatherPairsMPI (MPI_Allreduce): %s\n", buf);
        return {};
    }

    std::vector<int> recvcounts(static_cast<std::size_t>(size_));
    mpi_rc = MPI_Allgather(&nlocal, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in gatherPairsMPI (MPI_Allgather): %s\n", buf);
        return {};
    }

    std::vector<int> displs(static_cast<std::size_t>(size_));
    int offset = 0;
    for (int i = 0; i < size_; ++i)
    {
        displs[i] = offset;
        offset += recvcounts[i];
    }

    std::vector<Pair> all_pairs(static_cast<std::size_t>(total_pairs));
    mpi_rc = MPI_Allgatherv(local_pairs.data(), nlocal * static_cast<int>(sizeof(Pair)), MPI_BYTE,
                            all_pairs.data(), recvcounts.data(), displs.data(), MPI_BYTE,
                            MPI_COMM_WORLD);
    if (mpi_rc != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_rc, buf, &len);
        fprintf(stderr, "MPI error in gatherPairsMPI (MPI_Allgatherv): %s\n", buf);
        return {};
    }

    return all_pairs;
}

bool isPowerOfTwo(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

#else

void initFiltrationMPI(int *, char ***) {}
int mpiRank()
{
    return 0;
}
int mpiSize()
{
    return 1;
}
std::vector<double> gatherFiltrationValues(const std::vector<double> &v, int root)
{
    if (root != 0)
        throw std::invalid_argument(
            "gatherFiltrationValues: root must be 0 in single-process mode");
    return v;
}
std::vector<Pair> gatherPairsMPI(const std::vector<Pair> &p)
{
    return p;
}

#endif

} // namespace nerve::filtration::mpi
