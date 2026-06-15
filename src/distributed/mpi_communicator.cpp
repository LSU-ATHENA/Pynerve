#include "nerve/config.hpp"
#include "nerve/distributed/mpi_persistence.hpp"
#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nerve::distributed
{

// broadcast instantiations
template void MPICommunicator::broadcast<int>(int *data, int count, int root);
template void MPICommunicator::broadcast<unsigned>(unsigned *data, int count, int root);
template void MPICommunicator::broadcast<long>(long *data, int count, int root);
template void MPICommunicator::broadcast<long long>(long long *data, int count, int root);
template void MPICommunicator::broadcast<float>(float *data, int count, int root);
template void MPICommunicator::broadcast<double>(double *data, int count, int root);
template void MPICommunicator::broadcast<char>(char *data, int count, int root);
template void MPICommunicator::broadcast<std::uint64_t>(std::uint64_t *data, int count, int root);

template nerve::errors::ErrorResult<void> MPICommunicator::try_broadcast<int>(int *data, int count,
                                                                              int root);
template nerve::errors::ErrorResult<void>
MPICommunicator::try_broadcast<float>(float *data, int count, int root);
template nerve::errors::ErrorResult<void>
MPICommunicator::try_broadcast<double>(double *data, int count, int root);

// allgather instantiations
template void MPICommunicator::allgather<int>(const int *send_data, int send_count, int *recv_data,
                                              int recv_count);
template void MPICommunicator::allgather<unsigned>(const unsigned *send_data, int send_count,
                                                   unsigned *recv_data, int recv_count);
template void MPICommunicator::allgather<long>(const long *send_data, int send_count,
                                               long *recv_data, int recv_count);
template void MPICommunicator::allgather<long long>(const long long *send_data, int send_count,
                                                    long long *recv_data, int recv_count);
template void MPICommunicator::allgather<float>(const float *send_data, int send_count,
                                                float *recv_data, int recv_count);
template void MPICommunicator::allgather<double>(const double *send_data, int send_count,
                                                 double *recv_data, int recv_count);
template void MPICommunicator::allgather<char>(const char *send_data, int send_count,
                                               char *recv_data, int recv_count);

template nerve::errors::ErrorResult<void> MPICommunicator::try_allgather<int>(const int *send_data,
                                                                              int send_count,
                                                                              int *recv_data,
                                                                              int recv_count);
template nerve::errors::ErrorResult<void>
MPICommunicator::try_allgather<float>(const float *send_data, int send_count, float *recv_data,
                                      int recv_count);
template nerve::errors::ErrorResult<void>
MPICommunicator::try_allgather<double>(const double *send_data, int send_count, double *recv_data,
                                       int recv_count);

// isend/irecv instantiations
template MPIRequest MPICommunicator::isend<int>(const int *data, int count, int dest, int tag);
template MPIRequest MPICommunicator::isend<float>(const float *data, int count, int dest, int tag);
template MPIRequest MPICommunicator::isend<double>(const double *data, int count, int dest,
                                                   int tag);
template MPIRequest MPICommunicator::isend<char>(const char *data, int count, int dest, int tag);

template MPIRequest MPICommunicator::irecv<int>(int *data, int count, int source, int tag);
template MPIRequest MPICommunicator::irecv<float>(float *data, int count, int source, int tag);
template MPIRequest MPICommunicator::irecv<double>(double *data, int count, int source, int tag);
template MPIRequest MPICommunicator::irecv<char>(char *data, int count, int source, int tag);

template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_isend<int>(const int *data, int count, int dest, int tag);
template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_isend<float>(const float *data, int count, int dest, int tag);
template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_isend<double>(const double *data, int count, int dest, int tag);

template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_irecv<int>(int *data, int count, int source, int tag);
template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_irecv<float>(float *data, int count, int source, int tag);
template nerve::errors::ErrorResult<MPIRequest>
MPICommunicator::try_irecv<double>(double *data, int count, int source, int tag);

} // namespace nerve::distributed
