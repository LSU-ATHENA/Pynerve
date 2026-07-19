#pragma once

#include "nerve/core_types.hpp"
#include "nerve/types.hpp"

#include <span>
#include <vector>

#ifdef NERVE_HAS_CUDA

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace nerve::persistence::gpu::cohomology
{

inline constexpr int COHOMOLOGY_BLOCK_SIZE = 256;
inline constexpr int MAX_DIM_GPU_COHOMOLOGY = 6;
inline constexpr int COBOUNDARY_MAX_SIZE = 1024;

struct SimplexGPU
{
    int vertices[8];
    int num_vertices;
    double filtration_value;
    int dimension;
    int index;
    int birth_partner;
    int death_partner;
    bool reduced = false;
    bool cleared = false;
};

class GPUCohomologyComputer
{
public:
    explicit GPUCohomologyComputer();
    ~GPUCohomologyComputer();

    [[nodiscard]] bool initialize(size_t max_simplices, size_t max_dim);

    [[nodiscard]] bool computePersistenceDiagram(std::span<const SimplexGPU> simplices,
                                                 std::vector<nerve::Pair> &persistence_pairs);

    [[nodiscard]] double getLastComputeTime() const noexcept;
    [[nodiscard]] double getSpeedupVsCPU() const noexcept;
    void setCPUBaseline(double cpu_ms) noexcept;

    /** Copy the `cleared` field from the device SimplexGPU array to host. */
    [[nodiscard]] bool getClearedStates(std::vector<bool> &out_cleared) const;

    /** Copy the reduced coboundary sizes from device to host.
     *  Columns with cow_sizes == -1 were cleared by the clearing kernel. */
    [[nodiscard]] bool getReducedColumnSizes(std::vector<int> &out_sizes) const;

private:
    bool initialized_ = false;

    thrust::device_vector<SimplexGPU> d_simplices_;
    thrust::device_vector<int> d_coboundary_buffer_;
    thrust::device_vector<int> d_coboundary_sizes_;
    thrust::device_vector<int> d_coboundary_offsets_;
    thrust::device_vector<int> d_pivot_table_;
    thrust::device_vector<int> d_red_sizes_;

    double last_compute_time_ms_ = 0.0;
    double cpu_baseline_ms_ = 0.0;

    [[nodiscard]] bool computeCoboundaries(int num_simplices);
    [[nodiscard]] bool performClearing(int num_simplices);
    [[nodiscard]] bool detectApparentPairs(int num_simplices);
    [[nodiscard]] bool reduceCoboundaries(int num_simplices);
};

/** High-level convenience: run full GPU persistence pipeline on a vector of SimplexGPU.
 *
 *  Creates a temporary GPUCohomologyComputer, configures it for the given
 *  data, executes the pipeline, and returns a PersistenceDiagram with
 *  finite and essential pairs.
 *
 *  @param simplices      Filtered simplices sorted by (filtration, dimension).
 *  @param max_dimension  Highest simplex dimension to consider.
 *  @return               Populated PersistenceDiagram on success; empty on failure.
 */
[[nodiscard]] PersistenceDiagram computeGPUCohomology(const std::vector<SimplexGPU> &simplices,
                                                      int max_dimension);

} // namespace nerve::persistence::gpu::cohomology

#endif // NERVE_HAS_CUDA
