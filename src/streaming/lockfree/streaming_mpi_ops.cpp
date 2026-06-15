#include "nerve/config.hpp"
#include "nerve/streaming/incremental.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

namespace nerve::streaming
{

#ifdef NERVE_HAS_MPI

class MpiDistributedStreamer
{
public:
    explicit MpiDistributedStreamer(int window_size = 1000)
        : window_size_(window_size)
        , rank_(0)
        , size_(1)
        , initialized_(false)
    {}

    void init(int *argc, char ***argv)
    {
        int mpi_err = MPI_Init(argc, argv);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Init failed in MpiDistributedStreamer::init" << std::endl;
            return;
        }
        mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Comm_rank failed in MpiDistributedStreamer::init" << std::endl;
            return;
        }
        mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size_);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Comm_size failed in MpiDistributedStreamer::init" << std::endl;
            return;
        }
        initialized_ = true;
    }

    void finalize()
    {
        if (initialized_)
        {
            int mpi_err = MPI_Finalize();
            if (mpi_err != MPI_SUCCESS)
            {
                std::cerr << "MPI_Finalize failed in MpiDistributedStreamer::finalize" << std::endl;
            }
        }
        initialized_ = false;
    }

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool isInitialized() const { return initialized_; }

    void distributePoints(const std::vector<double> &local_points, Size dim,
                          std::vector<double> &all_points, Size &total_points)
    {
        int local_count = static_cast<int>(local_points.size() / dim);
        std::vector<int> counts(size_);
        int mpi_err =
            MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgather failed in distributePoints" << std::endl;
            return;
        }

        total_points = 0;
        for (int c : counts)
            total_points += static_cast<Size>(c);

        std::vector<int> displs(size_, 0);
        for (int i = 1; i < size_; ++i)
            displs[i] = displs[i - 1] + counts[i - 1] * static_cast<int>(dim);

        all_points.resize(total_points * dim);
        mpi_err = MPI_Allgatherv(local_points.data(), local_count * static_cast<int>(dim),
                                 MPI_DOUBLE, all_points.data(), counts.data(), displs.data(),
                                 MPI_DOUBLE, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgatherv failed in distributePoints" << std::endl;
        }
    }

    void allGatherPairs(const std::vector<Pair> &local_pairs, std::vector<Pair> &global_pairs)
    {
        int local_size = static_cast<int>(local_pairs.size());
        std::vector<int> counts(size_);
        int mpi_err =
            MPI_Allgather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgather failed in allGatherPairs" << std::endl;
            return;
        }

        std::vector<int> displs(size_, 0);
        int total = 0;
        for (int i = 0; i < size_; ++i)
        {
            displs[i] = total;
            total += counts[i];
        }

        struct MpiPair
        {
            double birth, death;
            int dim;
        };
        std::vector<MpiPair> local_mpi(local_size);
        for (int i = 0; i < local_size; ++i)
        {
            local_mpi[i].birth = local_pairs[i].birth;
            local_mpi[i].death = local_pairs[i].death;
            local_mpi[i].dim = local_pairs[i].dimension;
        }
        int local_byte_count = local_size * static_cast<int>(sizeof(MpiPair));
        std::vector<int> byte_counts(size_);
        std::vector<int> byte_displs(size_);
        for (int i = 0; i < size_; ++i)
        {
            byte_counts[i] = counts[i] * static_cast<int>(sizeof(MpiPair));
            byte_displs[i] = displs[i] * static_cast<int>(sizeof(MpiPair));
        }

        std::vector<MpiPair> global_mpi(total);
        mpi_err = MPI_Allgatherv(local_mpi.data(), local_byte_count, MPI_BYTE, global_mpi.data(),
                                 byte_counts.data(), byte_displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allgatherv failed in allGatherPairs" << std::endl;
            return;
        }
        global_pairs.resize(total);
        for (int i = 0; i < total; ++i)
        {
            global_pairs[i].birth = global_mpi[i].birth;
            global_pairs[i].death = global_mpi[i].death;
            global_pairs[i].dimension = static_cast<Size>(global_mpi[i].dim);
        }
    }

    void broadcastWindow(std::vector<double> &window, Size dim)
    {
        int count = static_cast<int>(window.size());
        int mpi_err = MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Bcast failed in broadcastWindow" << std::endl;
            return;
        }
        window.resize(count);
        mpi_err = MPI_Bcast(window.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Bcast failed in broadcastWindow" << std::endl;
        }
    }

    void reduceStabilityScores(const std::vector<double> &local_scores,
                               std::vector<double> &global_scores)
    {
        global_scores.resize(local_scores.size());
        int mpi_err = MPI_Reduce(local_scores.data(), global_scores.data(),
                                 static_cast<int>(local_scores.size()), MPI_DOUBLE, MPI_SUM, 0,
                                 MPI_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Reduce failed in reduceStabilityScores" << std::endl;
            return;
        }
        if (rank_ == 0)
        {
            for (double &s : global_scores)
                s /= static_cast<double>(size_);
        }
    }

private:
    int window_size_;
    int rank_;
    int size_;
    bool initialized_;
};

static MpiDistributedStreamer &globalStreamer()
{
    static MpiDistributedStreamer streamer;
    return streamer;
}

void initDistributedStreaming(int *argc, char ***argv)
{
    globalStreamer().init(argc, argv);
}

void finalizeDistributedStreaming()
{
    globalStreamer().finalize();
}

int distributedRank()
{
    return globalStreamer().rank();
}
int distributedSize()
{
    return globalStreamer().size();
}

void allGatherStreamResults(const std::vector<Pair> &local, std::vector<Pair> &global)
{
    globalStreamer().allGatherPairs(local, global);
}

#else

void initDistributedStreaming(int *, char ***) {}
void finalizeDistributedStreaming() {}
int distributedRank()
{
    return 0;
}
int distributedSize()
{
    return 1;
}
void allGatherStreamResults(const std::vector<Pair> &local, std::vector<Pair> &global)
{
    if (&global != &local)
        global = local;
}

#endif

} // namespace nerve::streaming
