#pragma once

#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::formats
{

struct PipelineConfig
{
    Size chunk_size = 10000;
    Size max_dim = 10;
    bool enable_gpu_scan = true;
    bool enable_clearing = true;
    bool enable_streaming = false;
    bool enable_ptx_kernels = true;
    double gpu_work_ratio = 0.999;
    Size gpu_columns_threshold = 1000;
    int device_id = 0;
};

struct PipelineResult
{
    std::vector<Pair> persistence_pairs;
    double total_time_ms = 0.0;
    double scan_time_ms = 0.0;
    double reduction_time_ms = 0.0;
    Size total_columns = 0;
    Size stable_columns = 0;
    Size unstable_columns = 0;
    Size nnz = 0;
    double compression_ratio = 0.0;
};

class PersistencePipeline
{
public:
    explicit PersistencePipeline(const PipelineConfig &config);
    ~PersistencePipeline();

    PersistencePipeline(const PersistencePipeline &) = delete;
    PersistencePipeline &operator=(const PersistencePipeline &) = delete;

    errors::ErrorResult<PipelineResult> compute(const PackedBoundaryMatrix &boundary_matrix,
                                                const std::vector<Field> &filtration_values);

    errors::ErrorResult<PipelineResult>
    computeFromColumns(Size n_rows, const std::vector<std::vector<Index>> &boundary_columns,
                       const std::vector<Field> &filtration_values);

private:
    PipelineConfig config_;

    PipelineResult runGpuScanAndReduce(const PackedBoundaryMatrix &matrix,
                                       const std::vector<Field> &filtration_values);
};

errors::ErrorResult<std::vector<Pair>>
computePackedPersistence(const std::vector<std::vector<Index>> &boundary_columns, Size n_rows,
                         const std::vector<Field> &filtration_values, int device_id = 0);

} // namespace nerve::formats
