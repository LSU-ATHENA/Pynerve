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

#if HAS_MPI

#include <mutex>

namespace
{

std::mutex &communicatorLifecycleMutex()
{
    static std::mutex mutex;
    return mutex;
}

int &communicatorLifecycleRefCount()
{
    static int ref_count = 0;
    return ref_count;
}

nerve::errors::ErrorResult<void> checkMpiResult(int status, const char *context)
{
    if (status == MPI_SUCCESS)
    {
        return nerve::errors::ErrorResult<void>::success();
    }
    auto err = nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::UNKNOWN,
                                                       std::string(context) + " (MPI error code " +
                                                           std::to_string(status) + ")");
    return err;
}

void checkMpiSuccess(int status, const char *context)
{
    auto res = checkMpiResult(status, context);
    if (res.isError())
    {
        throw std::runtime_error(res.error().message);
    }
}

template <typename T>
MPI_Datatype mpiDatatype();

template <>
MPI_Datatype mpiDatatype<int>()
{
    return MPI_INT;
}

template <>
MPI_Datatype mpiDatatype<unsigned>()
{
    return MPI_UNSIGNED;
}

template <>
MPI_Datatype mpiDatatype<long long>()
{
    return MPI_LONG_LONG;
}

template <>
MPI_Datatype mpiDatatype<float>()
{
    return MPI_FLOAT;
}

template <>
MPI_Datatype mpiDatatype<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype mpiDatatype<char>()
{
    return MPI_CHAR;
}

template <>
MPI_Datatype mpiDatatype<std::uint64_t>()
{
    return MPI_UINT64_T;
}

template <>
MPI_Datatype mpiDatatype<std::int64_t>()
{
    return MPI_INT64_T;
}

} // namespace

MPICommunicator::MPICommunicator()
{
    std::lock_guard<std::mutex> lock(communicatorLifecycleMutex());
    int finalized = 0;
    checkMpiSuccess(MPI_Finalized(&finalized), "MPI_Finalized failed");
    if (finalized != 0)
    {
        throw std::runtime_error("MPI has already been finalized");
    }
    int initialized = 0;
    checkMpiSuccess(MPI_Initialized(&initialized), "MPI_Initialized failed");
    if (!initialized)
    {
        int provided = 0;
        checkMpiSuccess(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided),
                        "MPI_Init_thread failed");
        if (provided < MPI_THREAD_SERIALIZED)
        {
            throw std::runtime_error("MPI implementation provides insufficient thread support "
                                     "(need MPI_THREAD_SERIALIZED)");
        }
    }
    communicatorLifecycleRefCount() += 1;
    checkMpiSuccess(MPI_Comm_size(MPI_COMM_WORLD, &world_size_), "MPI_Comm_size failed");
    checkMpiSuccess(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_), "MPI_Comm_rank failed");
}

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
        if (world_rank_ >= 0)
        {
            std::lock_guard<std::mutex> lock(communicatorLifecycleMutex());
            if (communicatorLifecycleRefCount() > 0)
            {
                communicatorLifecycleRefCount() -= 1;
            }
        }
        world_rank_ = other.world_rank_;
        world_size_ = other.world_size_;
        other.world_rank_ = -1;
        other.world_size_ = -1;
    }
    return *this;
}

MPICommunicator::~MPICommunicator()
{
    if (world_rank_ < 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(communicatorLifecycleMutex());
    if (communicatorLifecycleRefCount() > 0)
    {
        communicatorLifecycleRefCount() -= 1;
    }
}

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
    if (checked == 0)
    {
        return;
    }
    checkMpiSuccess(MPI_Bcast(data, checked, mpiDatatype<T>(), root, MPI_COMM_WORLD),
                    "MPI_Bcast failed");
}

template <typename T>
nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast(T *data, int count, int root)
{
    auto checked = checkedCount(count, "MPI broadcast count cannot be negative");
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
    if (checked == 0)
    {
        return nerve::errors::ErrorResult<void>::success();
    }
    return checkMpiResult(MPI_Bcast(data, checked, mpiDatatype<T>(), root, MPI_COMM_WORLD),
                          "MPI_Bcast failed");
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
    checkMpiSuccess(MPI_Allgather(send_data, checked_send, mpiDatatype<T>(), recv_data,
                                  checked_recv, mpiDatatype<T>(), MPI_COMM_WORLD),
                    "MPI_Allgather failed");
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
    if (checked_send == 0)
    {
        return nerve::errors::ErrorResult<void>::success();
    }
    return checkMpiResult(MPI_Allgather(send_data, checked_send, mpiDatatype<T>(), recv_data,
                                        checked_recv, mpiDatatype<T>(), MPI_COMM_WORLD),
                          "MPI_Allgather failed");
}

void MPICommunicator::barrier()
{
    checkMpiSuccess(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed");
}

nerve::errors::ErrorResult<void> MPICommunicator::try_barrier()
{
    return checkMpiResult(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed");
}

template <typename T>
MPIRequest MPICommunicator::isend(const T *data, int count, int dest, int tag)
{
    if (count < 0)
    {
        throw std::length_error("MPI isend count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        throw std::invalid_argument("MPI isend data cannot be null when count is positive");
    }
    if (dest < 0 || dest >= world_size_)
    {
        throw std::invalid_argument("MPI isend destination rank out of range");
    }
    MPIRequest req;
    checkMpiSuccess(MPI_Isend(const_cast<T *>(data), count, mpiDatatype<T>(), dest, tag,
                              MPI_COMM_WORLD, &req.request),
                    "MPI_Isend failed");
    req.active = true;
    return req;
}

template <typename T>
MPIRequest MPICommunicator::irecv(T *data, int count, int source, int tag)
{
    if (count < 0)
    {
        throw std::length_error("MPI irecv count cannot be negative");
    }
    if (count > 0 && data == nullptr)
    {
        throw std::invalid_argument("MPI irecv data cannot be null when count is positive");
    }
    if (source < 0 || source >= world_size_)
    {
        throw std::invalid_argument("MPI irecv source rank out of range");
    }
    MPIRequest req;
    checkMpiSuccess(
        MPI_Irecv(data, count, mpiDatatype<T>(), source, tag, MPI_COMM_WORLD, &req.request),
        "MPI_Irecv failed");
    req.active = true;
    return req;
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
    if (dest < 0 || dest >= world_size_)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
            "MPI isend destination rank out of range");
    }
    MPIRequest req;
    auto res = checkMpiResult(MPI_Isend(const_cast<T *>(data), count, mpiDatatype<T>(), dest, tag,
                                        MPI_COMM_WORLD, &req.request),
                              "MPI_Isend failed");
    if (res.isError())
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(res.errorCode(), res.error().message);
    }
    req.active = true;
    return nerve::errors::ErrorResult<MPIRequest>::success(std::move(req));
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
    if (source < 0 || source >= world_size_)
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(
            nerve::errors::ErrorCode::E41_RESOURCE_LIMIT, "MPI irecv source rank out of range");
    }
    MPIRequest req;
    auto res = checkMpiResult(
        MPI_Irecv(data, count, mpiDatatype<T>(), source, tag, MPI_COMM_WORLD, &req.request),
        "MPI_Irecv failed");
    if (res.isError())
    {
        return nerve::errors::ErrorResult<MPIRequest>::error(res.errorCode(), res.error().message);
    }
    req.active = true;
    return nerve::errors::ErrorResult<MPIRequest>::success(std::move(req));
}

void MPICommunicator::wait(MPIRequest &req)
{
    if (!req.active)
    {
        return;
    }
    MPI_Status status;
    checkMpiSuccess(MPI_Wait(&req.request, &status), "MPI_Wait failed");
    req.active = false;
}

void MPICommunicator::waitall(std::vector<MPIRequest> &reqs)
{
    if (reqs.empty())
    {
        return;
    }
    std::vector<MPI_Request> raw;
    raw.reserve(reqs.size());
    for (auto &r : reqs)
    {
        if (r.active)
        {
            raw.push_back(r.request);
        }
    }
    if (raw.empty())
    {
        return;
    }
    std::vector<MPI_Status> statuses(raw.size());
    checkMpiSuccess(MPI_Waitall(static_cast<int>(raw.size()), raw.data(), statuses.data()),
                    "MPI_Waitall failed");
    for (auto &r : reqs)
    {
        r.active = false;
    }
}

template <typename T>
MPIRequest MPICommunicator::iallgather(const T *sendbuf, int sendcount, T *recvbuf, int recvcount)
{
    if (sendcount > 0 && sendbuf == nullptr)
        throw std::invalid_argument("MPI iallgather sendbuf is null with positive sendcount");
    MPIRequest req;
    checkMpiSuccess(MPI_Iallgather(sendbuf, sendcount, mpiDatatype<T>(), recvbuf, recvcount,
                                   mpiDatatype<T>(), MPI_COMM_WORLD, &req.request),
                    "MPI_Iallgather failed");
    req.active = true;
    return req;
}

template <typename T>
MPIRequest MPICommunicator::ibroadcast(T *buffer, int count, int root)
{
    validateRootRank(root, world_size_);
    MPIRequest req;
    checkMpiSuccess(MPI_Ibcast(buffer, count, mpiDatatype<T>(), root, MPI_COMM_WORLD, &req.request),
                    "MPI_Ibcast failed");
    req.active = true;
    return req;
}

template <typename T>
MPIRequest MPICommunicator::ireduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root)
{
    validateRootRank(root, world_size_);
    MPIRequest req;
    checkMpiSuccess(MPI_Ireduce(sendbuf, recvbuf, count, mpiDatatype<T>(), op, root, MPI_COMM_WORLD,
                                &req.request),
                    "MPI_Ireduce failed");
    req.active = true;
    return req;
}

// Explicit template instantiations
template void MPICommunicator::broadcast<int>(int *, int, int);
template void MPICommunicator::broadcast<unsigned>(unsigned *, int, int);
template void MPICommunicator::broadcast<long>(long *, int, int);
template void MPICommunicator::broadcast<long long>(long long *, int, int);
template void MPICommunicator::broadcast<float>(float *, int, int);
template void MPICommunicator::broadcast<double>(double *, int, int);
template void MPICommunicator::broadcast<char>(char *, int, int);
template void MPICommunicator::broadcast<std::uint64_t>(std::uint64_t *, int, int);

template nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast<int>(int *, int, int);
template nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast<float>(float *, int, int);
template nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast<double>(double *, int,
                                                                                 int);

template void MPICommunicator::allgather<int>(const int *, int, int *, int);
template void MPICommunicator::allgather<unsigned>(const unsigned *, int, unsigned *, int);
template void MPICommunicator::allgather<long>(const long *, int, long *, int);
template void MPICommunicator::allgather<long long>(const long long *, int, long long *, int);
template void MPICommunicator::allgather<float>(const float *, int, float *, int);
template void MPICommunicator::allgather<double>(const double *, int, double *, int);
template void MPICommunicator::allgather<char>(const char *, int, char *, int);

template nerve::errors::ErrorResult<void> MPICommunicator::try_allgather<int>(const int *, int,
                                                                              int *, int);
template nerve::errors::ErrorResult<void> MPICommunicator::try_allgather<float>(const float *, int,
                                                                                float *, int);
template nerve::errors::ErrorResult<void>
MPICommunicator::try_allgather<double>(const double *, int, double *, int);

template MPIRequest MPICommunicator::isend<int>(const int *, int, int, int);
template MPIRequest MPICommunicator::isend<float>(const float *, int, int, int);
template MPIRequest MPICommunicator::isend<double>(const double *, int, int, int);
template MPIRequest MPICommunicator::isend<char>(const char *, int, int, int);

template MPIRequest MPICommunicator::irecv<int>(int *, int, int, int);
template MPIRequest MPICommunicator::irecv<float>(float *, int, int, int);
template MPIRequest MPICommunicator::irecv<double>(double *, int, int, int);
template MPIRequest MPICommunicator::irecv<char>(char *, int, int, int);

template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_isend<int>(const int *, int,
                                                                                int, int);
template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_isend<float>(const float *,
                                                                                  int, int, int);
template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_isend<double>(const double *,
                                                                                   int, int, int);

template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_irecv<int>(int *, int, int,
                                                                                int);
template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_irecv<float>(float *, int, int,
                                                                                  int);
template nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_irecv<double>(double *, int,
                                                                                   int, int);

template MPIRequest MPICommunicator::iallgather<int>(const int *, int, int *, int);
template MPIRequest MPICommunicator::ibroadcast<int>(int *, int, int);
template MPIRequest MPICommunicator::ireduce<int>(const int *, int *, int, MPI_Op, int);

#else // !HAS_MPI

MPICommunicator::MPICommunicator()
    : world_rank_(-1)
    , world_size_(-1)
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

int MPICommunicator::rank() const
{
    throw std::logic_error("MPICommunicator::rank() requires MPI support");
}

int MPICommunicator::size() const
{
    throw std::logic_error("MPICommunicator::size() requires MPI support");
}

bool MPICommunicator::is_root() const
{
    throw std::logic_error("MPICommunicator::is_root() requires MPI support");
}

template <typename T>
void MPICommunicator::broadcast(T *, int, int)
{
    throw std::logic_error("MPICommunicator::broadcast() requires MPI support");
}

template <typename T>
nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast(T *, int, int)
{
    return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                   "MPI broadcast requires MPI support");
}

template <typename T>
void MPICommunicator::allgather(const T *, int, T *, int)
{
    throw std::logic_error("MPICommunicator::allgather() requires MPI support");
}

template <typename T>
nerve::errors::ErrorResult<void> MPICommunicator::try_allgather(const T *, int, T *, int)
{
    return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                   "MPI allgather requires MPI support");
}

void MPICommunicator::barrier()
{
    throw std::logic_error("MPICommunicator::barrier() requires MPI support");
}

nerve::errors::ErrorResult<void> MPICommunicator::try_barrier()
{
    return nerve::errors::ErrorResult<void>::error(nerve::errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                   "MPI barrier requires MPI support");
}

template <typename T>
MPIRequest MPICommunicator::isend(const T *, int, int, int)
{
    throw std::logic_error("MPICommunicator::isend() requires MPI support");
}

template <typename T>
MPIRequest MPICommunicator::irecv(T *, int, int, int)
{
    throw std::logic_error("MPICommunicator::irecv() requires MPI support");
}

template <typename T>
nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_isend(const T *, int, int, int)
{
    return nerve::errors::ErrorResult<MPIRequest>::error(
        nerve::errors::ErrorCode::E41_RESOURCE_LIMIT, "MPI isend requires MPI support");
}

template <typename T>
nerve::errors::ErrorResult<MPIRequest> MPICommunicator::try_irecv(T *, int, int, int)
{
    return nerve::errors::ErrorResult<MPIRequest>::error(
        nerve::errors::ErrorCode::E41_RESOURCE_LIMIT, "MPI irecv requires MPI support");
}

template <typename T>
MPIRequest MPICommunicator::iallgather(const T *, int, T *, int)
{
    throw std::logic_error("MPICommunicator::iallgather() requires MPI support");
}

template <typename T>
MPIRequest MPICommunicator::ibroadcast(T *, int, int)
{
    throw std::logic_error("MPICommunicator::ibroadcast() requires MPI support");
}

template <typename T>
MPIRequest MPICommunicator::ireduce(const T *, T *, int, MPI_Op, int)
{
    throw std::logic_error("MPICommunicator::ireduce() requires MPI support");
}

// Explicit template instantiations (needed so code that references these types links)
template void MPICommunicator::broadcast<int>(int *, int, int);
template void MPICommunicator::broadcast<float>(float *, int, int);
template void MPICommunicator::broadcast<double>(double *, int, int);
template void MPICommunicator::allgather<int>(const int *, int, int *, int);
template void MPICommunicator::allgather<float>(const float *, int, float *, int);
template void MPICommunicator::allgather<double>(const double *, int, double *, int);
template MPIRequest MPICommunicator::isend<int>(const int *, int, int, int);
template MPIRequest MPICommunicator::isend<float>(const float *, int, int, int);
template MPIRequest MPICommunicator::irecv<int>(int *, int, int, int);
template MPIRequest MPICommunicator::irecv<float>(float *, int, int, int);
template MPIRequest MPICommunicator::iallgather<int>(const int *, int, int *, int);
template MPIRequest MPICommunicator::ibroadcast<int>(int *, int, int);

#endif // HAS_MPI

} // namespace nerve::distributed
