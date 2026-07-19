#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/types.hpp"

#include <memory>
#include <vector>

namespace nerve::persistence
{

// Per-phase timing breakdown for HyphaReducer profiling.
// Filled when a non-null pointer is passed to compute().
struct HyphaPhaseTimings
{
    // CPU: CSC array construction (boundary_matrix.buildCSC)
    double csc_build_ms = 0.0;
    // CPU: clearing compression pass
    double clearing_ms = 0.0;
    // CPU: extracting filtration metadata from boundary matrix
    double submatrix_build_ms = 0.0;
    // GPU: CSC upload (col_ptr + row_indices) + memset + pack kernel
    double gpu_pack_ms = 0.0;
    // GPU: warp-level packed-column reduction (KernelDispatcher::computeMatrixReduction)
    double gpu_reduction_ms = 0.0;
    // GPU: pivot table D2H download + pair extraction (incl. filtration-order verify)
    double gpu_download_ms = 0.0;
    // Total wall time of compute() minus the sum of the phases above
    double overhead_ms = 0.0;
};

class HyphaReducer
{
public:
    struct Config
    {};

    HyphaReducer();
    explicit HyphaReducer(const Config &cfg);
    ~HyphaReducer();

    // If timings is non-null, per-phase timings are recorded for profiling.
    std::vector<Pair> compute(const algebra::BoundaryMatrix &matrix,
                              HyphaPhaseTimings *timings = nullptr);

    void setConfig(const Config &cfg) { config_ = cfg; }
    const Config &config() const { return config_; }

private:
    std::vector<Pair> gpuSubmatrixReduction(const int *col_ptr, const int *row_indices, int nnz,
                                            int n_cols, int n_rows,
                                            const std::vector<double> &col_filtration_values,
                                            const std::vector<double> &row_filtration_values,
                                            const std::vector<Dimension> &dimensions,
                                            HyphaPhaseTimings *timings);

    Config config_;
    struct GpuPool;
    std::unique_ptr<GpuPool> gpu_pool_;
};

} // namespace nerve::persistence
