#pragma once
#include "nerve/types.hpp"

#include <cstddef>
#include <cstdint>
#if defined(NERVE_HAS_MPI) && NERVE_HAS_MPI
#include <mpi.h>
#else
using MPI_Comm = int;
#endif

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>

namespace nerve::algorithms::multi_gpu
{
cudaError_t distributeDistanceWork(Size n, int *d_counts, int num_gpus);
cudaError_t gatherDistanceResults(const double *const *d_partials, const int *counts, int num_gpus,
                                  double *d_result, cudaStream_t s = 0);
} // namespace nerve::algorithms::multi_gpu

namespace nerve::algorithms::mpi_cuda
{
cudaError_t allGatherDeviceVectors(const double *d_local, Size nlocal, Size stride, double *d_all,
                                   cudaStream_t s = 0);
}

namespace nerve::autodiff::multi_gpu
{
cudaError_t distributeAutodiffWork(Size n_params, int *d_offsets, int num_gpus);
cudaError_t reduceGradients(const double *const *d_grads, const int *sizes, int num_gpus,
                            double *d_result);
} // namespace nerve::autodiff::multi_gpu

namespace nerve::compression::multi_gpu
{
cudaError_t distributeCompression(Size data_size, int num_gpus);
}

namespace nerve::core::multi_gpu
{
cudaError_t scatterInput(const double *d_input, Size n, int num_gpus, double **d_partitions);
cudaError_t gatherOutput(double **d_partitions, const int *sizes, int num_gpus, double *d_out);
} // namespace nerve::core::multi_gpu

namespace nerve::dmt::multi_gpu
{
cudaError_t scatterCells(const int *d_cells, Size n, int ngpus, int **d_partitions);
cudaError_t gatherGradients(const int *const *d_grads, const int *sizes, int ngpus, int *d_result);
} // namespace nerve::dmt::multi_gpu

namespace nerve::encoders::multi_gpu
{
cudaError_t scatterBatches(const double *d_data, Size total, int ngpus, double **d_batches,
                           Size *batch_sizes);
cudaError_t gatherEncoded(double **d_encoded, const Size *sizes, int ngpus, double *d_out);
} // namespace nerve::encoders::multi_gpu

namespace nerve::filtration::multi_gpu
{
cudaError_t distributeFiltration(const int *d_edges, Size nedges, int ngpus, int **d_partitions);
cudaError_t gatherPairs(const Pair *const *d_partials, const int *counts, int ngpus,
                        Pair *d_result);
} // namespace nerve::filtration::multi_gpu

namespace nerve::filtration::mpi_cuda
{
cudaError_t allGatherFiltrationGPU(const double *d_local, Size nlocal, double *d_all,
                                   cudaStream_t s = 0);
}

namespace nerve::graphs::mpi_cuda
{
cudaError_t allGatherGraphGPU(const int *d_local_edges, Size nlocal, int *d_counts,
                              int *d_all_edges, cudaStream_t s = 0);
}

namespace nerve::metrics::multi_gpu
{
cudaError_t distributeDiagramPairs(const double *d_dgm, Size npairs, int ngpus,
                                   double **d_partitions);
cudaError_t gatherDistances(double **d_partials, const int *sizes, int ngpus, double *d_out);
} // namespace nerve::metrics::multi_gpu

namespace nerve::metrics::mpi_cuda
{
cudaError_t bottleneckDistMPI(const double *d_dgm, Size nlocal, int root, double *d_result,
                              cudaStream_t s = 0);
}

namespace nerve::ml::multi_gpu
{
cudaError_t distributeTrainingData(const double *d_data, Size n, int ngpus, double **d_partitions);
cudaError_t syncModelParams(double *d_params, Size nparams, int ngpus);
} // namespace nerve::ml::multi_gpu

namespace nerve::nn::multi_gpu
{
cudaError_t distributeLayers(const double *d_weights, Size nweights, int ngpus,
                             double **d_partitions);
cudaError_t reduceGradients(const double *const *d_grads, const int *sizes, int ngpus,
                            double *d_out);
} // namespace nerve::nn::multi_gpu

namespace nerve::optimization::multi_gpu
{
cudaError_t distributeParams(double *d_params, Size n, int ngpus, double **d_partitions);
cudaError_t reduceParamGrads(const double *const *d_grads, const int *sizes, int ngpus,
                             double *d_out);
} // namespace nerve::optimization::multi_gpu

namespace nerve::persistence::mpi_cuda
{
cudaError_t distributeColumnsGPU(const int *d_columns, Size ncols, int ngpus, int **d_partitions,
                                 Size *counts);
cudaError_t exchangePivotsGPU(const int *d_local_pivots, Size nlocal, int *d_all_pivots,
                              MPI_Comm comm, cudaStream_t s = 0);
cudaError_t allGatherPairsGPU(const Pair *d_local, Size nlocal, Pair *d_all, MPI_Comm comm);
} // namespace nerve::persistence::mpi_cuda

namespace nerve::probabilistic::multi_gpu
{
cudaError_t distributeSamples(const double *d_samples, Size n, int ngpus, double **d_partitions);
cudaError_t gatherStats(double **d_partials, const int *sizes, int ngpus, double *d_out);
} // namespace nerve::probabilistic::multi_gpu

namespace nerve::sheaf::multi_gpu
{
cudaError_t distributeStalks(const double *d_stalks, Size total_stalk_dim, int ngpus,
                             double **d_partitions);
cudaError_t gatherCocycles(const double *const *d_cocycles, const int *sizes, int ngpus,
                           double *d_result);
} // namespace nerve::sheaf::multi_gpu

namespace nerve::spectral::multi_gpu
{
cudaError_t distributeMatrices(const double *d_matrix, Size n, int ngpus, double **d_partitions);
cudaError_t gatherEigenvectors(const double *const *d_vectors, const int *sizes, int ngpus,
                               double *d_result);
} // namespace nerve::spectral::multi_gpu

namespace nerve::streaming::mpi_cuda
{
cudaError_t allGatherPointsGPU(const double *d_local, Size nlocal, Size dim, double *d_all,
                               MPI_Comm comm, cudaStream_t s = 0);
cudaError_t reduceStabilityGPU(const double *d_local, Size nlocal, double *d_global, MPI_Comm comm);
} // namespace nerve::streaming::mpi_cuda

#endif // NERVE_HAS_CUDA

// Pure MPI forward declarations — no CUDA dependency

namespace nerve::algorithms::mpi
{
void initAlgorithmsMPI(int *argc, char ***argv);
int algoRank();
int algoSize();
void allGatherDistances(const double *local, Size nlocal, Size dim, double *all);
} // namespace nerve::algorithms::mpi

namespace nerve::autodiff::mpi
{
void syncGradients(double *grads, Size n);
}

namespace nerve::compression::mpi
{
void allGatherCompressed(const uint8_t *local, Size nlocal, uint8_t *all);
}

namespace nerve::core::mpi
{
void broadcastConfig(void *buf, Size n, int root);
void reduceMetrics(const double *local, double *global, Size n);
} // namespace nerve::core::mpi

namespace nerve::dmt::mpi
{
void allGatherGradientField(const int *local, Size nlocal, int *all);
}

namespace nerve::encoders::mpi
{
void gatherEncodedVectors(const double *local, Size nlocal, Size code_dim, double *all);
}

namespace nerve::ml::mpi
{
void allReduceGradients(double *grads, Size n);
}

namespace nerve::nn::mpi
{
void syncNNParams(double *params, Size n);
}

namespace nerve::optimization::mpi
{
void allReduceOptimizerState(double *state, Size n);
}

namespace nerve::probabilistic::mpi
{
void syncRandomSeeds(unsigned long long *seeds, Size n);
}

namespace nerve::runtime::mpi
{
void broadcastCalibration(double *params, Size n, int root);
}

namespace nerve::sheaf::mpi
{
void allGatherCocycles(const double *local, Size nlocal, double *all);
}

namespace nerve::spectral::mpi
{
void allReduceEigenvalues(const double *local, double *global, Size n);
}
