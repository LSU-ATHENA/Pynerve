#include "nerve/determinism.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

// Forward declaration from determinism_ops.cpp
namespace nerve::determinism
{
extern bool cross_count;
}

namespace nerve::determinism
{

void deterministic_reduce(const double *send, double *recv, int n, int root, MPI_Comm comm)
{
    if (!cross_count)
    {
        int mpi_err =
            MPI_Reduce(const_cast<double *>(send), recv, n, MPI_DOUBLE, MPI_SUM, root, comm);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Reduce failed in deterministic_reduce" << std::endl;
        }
        return;
    }
    int num_bins = 3;
    std::vector<double> binned_send(n * num_bins, 0.0);
    std::vector<double> binned_recv(n * num_bins, 0.0);
    int rank;
    int mpi_err = MPI_Comm_rank(comm, &rank);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_rank failed in deterministic_reduce" << std::endl;
        return;
    }
    for (int i = 0; i < n; ++i)
    {
        double v = send[i];
        if (v == 0.0)
        {
            binned_send[i * num_bins] = 0.0;
            continue;
        }
        int exp_;
        frexp(v, &exp_);
        int bin = (exp_ <= 0) ? 0 : (exp_ <= 8) ? 1 : 2;
        binned_send[i * num_bins + bin] = v;
    }
    MPI_Datatype binned_t;
    mpi_err = MPI_Type_contiguous(num_bins, MPI_DOUBLE, &binned_t);
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_reduce (MPI_Type_contiguous): %s\n", buf);
        return;
    }
    mpi_err = MPI_Type_commit(&binned_t);
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_reduce (MPI_Type_commit): %s\n", buf);
        MPI_Type_free(&binned_t);
        return;
    }
    mpi_err = MPI_Reduce(binned_send.data(), binned_recv.data(), n, binned_t, MPI_SUM, root, comm);
    {
        int free_rc = MPI_Type_free(&binned_t);
        if (free_rc != MPI_SUCCESS)
        {
            char buf[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(free_rc, buf, &len);
            fprintf(stderr, "MPI error in deterministic_reduce (MPI_Type_free): %s\n", buf);
        }
    }
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_reduce (cross-count MPI_Reduce): %s\n", buf);
        return;
    }
    if (rank == root)
        for (int i = 0; i < n; ++i)
            recv[i] = binned_recv[i * num_bins] + binned_recv[i * num_bins + 1] +
                      binned_recv[i * num_bins + 2];
}

void deterministic_allreduce(const double *send, double *recv, int n, MPI_Comm comm)
{
    if (!cross_count)
    {
        int mpi_err = MPI_Allreduce(const_cast<double *>(send), recv, n, MPI_DOUBLE, MPI_SUM, comm);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Allreduce failed in deterministic_allreduce" << std::endl;
        }
        return;
    }
    int num_bins = 3;
    std::vector<double> binned_send(n * num_bins, 0.0);
    std::vector<double> binned_recv(n * num_bins, 0.0);
    for (int i = 0; i < n; ++i)
    {
        double v = send[i];
        if (v == 0.0)
        {
            binned_send[i * num_bins] = 0.0;
            continue;
        }
        int exp_;
        frexp(v, &exp_);
        int bin = (exp_ <= 0) ? 0 : (exp_ <= 8) ? 1 : 2;
        binned_send[i * num_bins + bin] = v;
    }
    MPI_Datatype binned_t;
    int mpi_err = MPI_Type_contiguous(num_bins, MPI_DOUBLE, &binned_t);
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_allreduce (MPI_Type_contiguous): %s\n", buf);
        return;
    }
    mpi_err = MPI_Type_commit(&binned_t);
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_allreduce (MPI_Type_commit): %s\n", buf);
        MPI_Type_free(&binned_t);
        return;
    }
    mpi_err = MPI_Allreduce(binned_send.data(), binned_recv.data(), n, binned_t, MPI_SUM, comm);
    {
        int free_rc = MPI_Type_free(&binned_t);
        if (free_rc != MPI_SUCCESS)
        {
            char buf[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(free_rc, buf, &len);
            fprintf(stderr, "MPI error in deterministic_allreduce (MPI_Type_free): %s\n", buf);
        }
    }
    if (mpi_err != MPI_SUCCESS)
    {
        char buf[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err, buf, &len);
        fprintf(stderr, "MPI error in deterministic_allreduce (cross-count MPI_Allreduce): %s\n",
                buf);
        return;
    }
    for (int i = 0; i < n; ++i)
        recv[i] = binned_recv[i * num_bins] + binned_recv[i * num_bins + 1] +
                  binned_recv[i * num_bins + 2];
}

} // namespace nerve::determinism
