#include "nerve/config.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

namespace nerve::graphs::mpi
{

#ifdef NERVE_HAS_MPI

struct DistributedGraph
{
    std::vector<int> local_vertices;
    std::vector<std::pair<int, int>> local_edges;
    int rank;
    int size;

    DistributedGraph()
    {
        int mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (mpi_err != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Comm_rank failed in DistributedGraph constructor");
        }
        mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (mpi_err != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Comm_size failed in DistributedGraph constructor");
        }
    }

    void allGatherVertices()
    {
        int nlocal = static_cast<int>(local_vertices.size());
        std::vector<int> counts(size);
        int mpi_err = MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgather failed in allGatherVertices" << std::endl;
            return;
        }
        std::vector<int> displs(size, 0);
        int total = 0;
        for (int i = 0; i < size; ++i)
        {
            displs[i] = total;
            total += counts[i];
        }
        std::vector<int> all_verts(total);
        mpi_err = MPI_Allgatherv(local_vertices.data(), nlocal, MPI_INT, all_verts.data(),
                                 counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgatherv failed in allGatherVertices" << std::endl;
            return;
        }
        local_vertices = std::move(all_verts);
    }

    void distributeEdges(const std::vector<std::pair<int, int>> &all_edges)
    {
        int nedges = static_cast<int>(all_edges.size());
        int edges_per_rank = nedges / size;
        int extra = nedges % size;
        int start = rank * edges_per_rank + std::min(rank, extra);
        int count = edges_per_rank + (rank < extra ? 1 : 0);
        local_edges.assign(all_edges.begin() + start, all_edges.begin() + start + count);
    }
};

DistributedGraph &globalGraph()
{
    static DistributedGraph g;
    return g;
}

void initGraphMPI(int *argc, char ***argv)
{
    int mpi_err = MPI_Init(argc, argv);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Init failed in initGraphMPI" << std::endl;
    }
}

int graphRank()
{
    return globalGraph().rank;
}
int graphSize()
{
    return globalGraph().size;
}

void distributeGraphEdges(const std::vector<std::pair<int, int>> &edges)
{
    globalGraph().distributeEdges(edges);
}

std::vector<int> gatherGraphResults(const std::vector<int> &local)
{
    std::vector<int> all;
    int nlocal = static_cast<int>(local.size());
    int total = 0;
    int mpi_err = MPI_Reduce(&nlocal, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Reduce failed in gatherGraphResults" << std::endl;
        return {};
    }
    if (globalGraph().rank == 0)
        all.resize(total);
    std::vector<int> counts(globalGraph().size, 0);
    mpi_err = MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Gather failed in gatherGraphResults" << std::endl;
        return {};
    }
    std::vector<int> displs(globalGraph().size, 0);
    for (int i = 1; i < globalGraph().size; ++i)
        displs[i] = displs[i - 1] + counts[i - 1];
    mpi_err = MPI_Gatherv(local.data(), nlocal, MPI_INT, all.data(), counts.data(), displs.data(),
                          MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Gatherv failed in gatherGraphResults" << std::endl;
        return {};
    }
    return all;
}

#else

void initGraphMPI(int *, char ***) {}
int graphRank()
{
    return 0;
}
int graphSize()
{
    return 1;
}
void distributeGraphEdges(const std::vector<std::pair<int, int>> &edges)
{
    (void)edges;
}
std::vector<int> gatherGraphResults(const std::vector<int> &v)
{
    return v;
}

#endif

} // namespace nerve::graphs::mpi
