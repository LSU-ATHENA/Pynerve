#include "nerve/config.hpp"
#include "nerve/distributed/mpi_persistence.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nerve::distributed
{

int checkedCount(int count, const char *context)
{
    if (count < 0)
    {
        throw std::length_error(context);
    }
    return count;
}

void validateRootRank(int root, int world_size)
{
    if (root < 0 || root >= world_size)
    {
        throw std::invalid_argument("MPI broadcast root is out of range");
    }
}

#if !HAS_MPI

MPICommunicator::MPICommunicator()
    : world_rank_(0)
    , world_size_(1)
{}
MPICommunicator::MPICommunicator(MPICommunicator &&other) noexcept
    : world_rank_(other.world_rank_)
    , world_size_(other.world_size_)
{
    other.world_rank_ = -1;
    other.world_size_ = -1;
}
MPICommunicator &MPICommunicator::operator=(MPICommunicator &&other) noexcept
{
    if (this != &other)
    {
        world_rank_ = other.world_rank_;
        world_size_ = other.world_size_;
        other.world_rank_ = -1;
        other.world_size_ = -1;
    }
    return *this;
}
MPICommunicator::~MPICommunicator() {}

int MPICommunicator::rank() const
{
    return world_rank_;
}
int MPICommunicator::size() const
{
    return world_size_;
}
bool MPICommunicator::is_root() const
{
    return world_rank_ == 0;
}

template <typename T>
void MPICommunicator::broadcast(T *data, int count, int root)
{
    int checked = checkedCount(count, "MPI broadcast count cannot be negative");
    validateRootRank(root, world_size_);
    if (checked > 0 && data == nullptr)
    {
        throw std::invalid_argument("MPI broadcast data cannot be null when count is positive");
    }
}

template <typename T>
nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast(T *data, int count, int root)
{
    int checked = checkedCount(count, "MPI broadcast count cannot be negative");
    if (root < 0 || root >= world_size_)
    {
        return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
                                                       "MPI broadcast root is out of range");
    }
    if (checked > 0 && data == nullptr)
    {
        return nerve::errors::ErrorResult<void>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI broadcast data cannot be null when count is positive");
    }
    return nerve::errors::ErrorResult<void>::success();
}

template <typename T>
void MPICommunicator::allgather(const T *send_data, int send_count, T *recv_data, int recv_count)
{
    int checked_send = checkedCount(send_count, "MPI allgather send count cannot be negative");
    int checked_recv = checkedCount(recv_count, "MPI allgather recv count cannot be negative");
    if (checked_send != checked_recv)
    {
        throw std::invalid_argument("MPI allgather send and recv counts must match");
    }
    if (checked_send > 0 && send_data == nullptr)
    {
        throw std::invalid_argument(
            "MPI allgather send data cannot be null when count is positive");
    }
    if (checked_recv > 0 && recv_data == nullptr)
    {
        throw std::invalid_argument(
            "MPI allgather recv data cannot be null when count is positive");
    }
    if (checked_send == 0)
    {
        return;
    }
    std::copy(send_data, send_data + checked_send, recv_data);
}

template <typename T>
nerve::errors::ErrorResult<void> MPICommunicator::try_allgather(const T *send_data, int send_count,
                                                                T *recv_data, int recv_count)
{
    int checked_send = checkedCount(send_count, "MPI allgather send count cannot be negative");
    int checked_recv = checkedCount(recv_count, "MPI allgather recv count cannot be negative");
    if (checked_send != checked_recv)
    {
        return nerve::errors::ErrorResult<void>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI allgather send and recv counts must match");
    }
    if (checked_send > 0 && send_data == nullptr)
    {
        return nerve::errors::ErrorResult<void>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI allgather send data cannot be null when count is positive");
    }
    if (checked_recv > 0 && recv_data == nullptr)
    {
        return nerve::errors::ErrorResult<void>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI allgather recv data cannot be null when count is positive");
    }
    if (checked_send > 0)
    {
        std::copy(send_data, send_data + checked_send, recv_data);
    }
    return nerve::errors::ErrorResult<void>::success();
}

void MPICommunicator::barrier() {}

nerve::errors::ErrorResult<void> MPICommunicator::try_barrier()
{
    return nerve::errors::ErrorResult<void>::success();
}

template <typename T>
MPIRequest MPICommunicator::isend(const T *data, int count, int dest, int)
{
    if (count < 0)
    {
        throw std::length_error("MPI isend count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        throw std::invalid_argument("MPI isend data cannot be null when count is positive");
    }
    if (dest != 0)
    {
        throw std::invalid_argument("MPI isend destination must be 0 in single-process mode");
    }
    return MPIRequest{};
}

template <typename T>
MPIRequest MPICommunicator::irecv(T *data, int count, int source, int)
{
    if (count < 0)
    {
        throw std::length_error("MPI irecv count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        throw std::invalid_argument("MPI irecv data cannot be null when count is positive");
    }
    if (source != 0)
    {
        throw std::invalid_argument("MPI irecv source must be 0 in single-process mode");
    }
    return MPIRequest{};
}

template <typename T>
nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_isend(const T *data, int count,
                                                                  int dest, int tag)
{
    if (count < 0)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT, "MPI isend count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI isend data cannot be null when count is positive");
    }
    if (dest != 0)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
            "MPI isend destination must be 0 in single-process mode");
    }
    (void)tag;
    return nerve::errors::ErrorResult<MPIRequest>::success(MPIRequest{});
}

template <typename T>
nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_irecv(T *data, int count, int source,
                                                                  int tag)
{
    if (count < 0)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT, "MPI irecv count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E81_MATRIX_EMPTY,
            "MPI irecv data cannot be null when count is positive");
    }
    if (source != 0)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
            "MPI irecv source must be 0 in single-process mode");
    }
    (void)tag;
    return nerve::errors::ErrorResult<MPIRequest>::success(MPIRequest{});
}

#endif // !HAS_MPI

} // namespace nerve::distributed
