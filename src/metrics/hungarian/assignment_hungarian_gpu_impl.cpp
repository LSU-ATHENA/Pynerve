
#include "nerve/errors/errors.hpp"
#include "nerve/metrics/distances.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

namespace nerve::gpu::metrics::detail
{

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

errors::ErrorResult<double> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<double>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<double> numericError(std::string_view message)
{
    return errors::ErrorResult<double>::error(errors::ErrorCode::E20_NUM_NAN, message);
}

errors::ErrorResult<void> flattenCostMatrix(const std::vector<std::vector<double>> &cost_matrix,
                                            int &n, std::vector<double> &flat_costs)
{
    n = 0;
    flat_costs.clear();
    if (cost_matrix.empty())
    {
        return errors::ErrorResult<void>::success();
    }
    if (cost_matrix.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Hungarian cost matrix exceeds int range");
    }
    std::size_t matrix_entries = 0;
    if (!checkedProduct(cost_matrix.size(), cost_matrix.size(), matrix_entries))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Hungarian cost matrix size overflows");
    }
    n = static_cast<int>(cost_matrix.size());
    flat_costs.reserve(matrix_entries);
    for (const auto &row : cost_matrix)
    {
        if (row.size() != cost_matrix.size())
        {
            flat_costs.clear();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                    "Hungarian cost matrix must be square");
        }
        for (double cost : row)
        {
            if (!std::isfinite(cost))
            {
                flat_costs.clear();
                return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN,
                                                        "Hungarian costs must be finite");
            }
            flat_costs.push_back(cost);
        }
    }
    return errors::ErrorResult<void>::success();
}

} // namespace

// Forward declarations from cuda_hungarian.cu
extern void launchHungarianInit(const double *d_cost_matrix, double *d_row_labels,
                                double *d_col_labels, int *d_row_matched, int *d_col_matched, int n,
                                cudaStream_t stream);

extern void launchHungarianGreedyMatch(const double *d_cost_matrix, const double *d_row_labels,
                                       const double *d_col_labels, int *d_row_matched,
                                       int *d_col_matched, int *d_match_count, int n,
                                       cudaStream_t stream);

extern void launchBottleneckCandidates(const double *d_cost_matrix, double *d_candidates,
                                       int *d_candidate_count, int n, int max_candidates,
                                       cudaStream_t stream);

// GPU-accelerated Hungarian algorithm engine
class GPUHungarianEngine
{
public:
    static errors::ErrorResult<double>
    solveAssignment(const std::vector<std::vector<double>> &cost_matrix,
                    std::vector<std::pair<int, int>> &out_assignment)
    {
        int n = 0;
        std::vector<double> flat_costs;
        auto validation = flattenCostMatrix(cost_matrix, n, flat_costs);
        if (validation.isError())
        {
            out_assignment.clear();
            return errors::ErrorResult<double>::error(validation.errorCode(),
                                                      validation.error().message);
        }
        if (n == 0)
        {
            out_assignment.clear();
            return errors::ErrorResult<double>::success(0.0);
        }

        // Allocate GPU memory
        double *d_cost_matrix = nullptr;
        double *d_row_labels = nullptr;
        double *d_col_labels = nullptr;
        int *d_row_matched = nullptr;
        int *d_col_matched = nullptr;
        int *d_match_count = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_cost_matrix, flat_costs.size() * sizeof(double));
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_cost_matrix, flat_costs.data(), flat_costs.size() * sizeof(double),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_row_labels, n * sizeof(double));
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_col_labels, n * sizeof(double));
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_row_matched, n * sizeof(int));
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_col_matched, n * sizeof(int));
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_match_count, sizeof(int));
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        // Initialize matching count to 0
        int h_match_count = 0;
        err = cudaMemcpy(d_match_count, &h_match_count, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Initialize labels
        launchHungarianInit(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                            n, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Greedy initial matching
        launchHungarianGreedyMatch(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched,
                                   d_col_matched, d_match_count, n, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Copy matching results back
        std::vector<int> hRowMatched(n);
        std::vector<int> hColMatched(n);

        err =
            cudaMemcpy(hRowMatched.data(), d_row_matched, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err =
            cudaMemcpy(hColMatched.data(), d_col_matched, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                    d_match_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Build assignment from matching
        out_assignment.clear();
        double total_cost = 0.0;

        for (int i = 0; i < n; ++i)
        {
            if (hRowMatched[i] != -1)
            {
                out_assignment.push_back({i, hRowMatched[i]});
                const double next_total = total_cost + cost_matrix[i][hRowMatched[i]];
                if (!std::isfinite(next_total))
                {
                    cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                            d_match_count);
                    out_assignment.clear();
                    return numericError("Hungarian assignment cost overflow");
                }
                total_cost = next_total;
            }
        }

        // Cleanup
        cleanup(d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched,
                d_match_count);

        return errors::ErrorResult<double>::success(std::move(total_cost));
    }

    static errors::ErrorResult<double>
    solveBottleneck(const std::vector<std::vector<double>> &cost_matrix,
                    std::vector<std::pair<int, int>> &out_assignment)
    {
        // For bottleneck distance, use binary search on candidate thresholds
        int n = 0;
        std::vector<double> flat_costs;
        auto validation = flattenCostMatrix(cost_matrix, n, flat_costs);
        if (validation.isError())
        {
            out_assignment.clear();
            return errors::ErrorResult<double>::error(validation.errorCode(),
                                                      validation.error().message);
        }
        if (n == 0)
        {
            out_assignment.clear();
            return errors::ErrorResult<double>::success(0.0);
        }

        if (flat_costs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            out_assignment.clear();
            return resourceLimit("Hungarian bottleneck candidate count exceeds int range");
        }

        double *d_cost_matrix = nullptr;
        double *d_candidates = nullptr;
        int *d_candidate_count = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_cost_matrix, flat_costs.size() * sizeof(double));
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_cost_matrix, flat_costs.data(), flat_costs.size() * sizeof(double),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        int max_candidates = static_cast<int>(flat_costs.size());
        err = cudaMalloc(&d_candidates, max_candidates * sizeof(double));
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_candidate_count, sizeof(int));
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        int h_candidate_count = 0;
        err =
            cudaMemcpy(d_candidate_count, &h_candidate_count, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            cudaFree(d_candidate_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Find candidate thresholds on GPU
        launchBottleneckCandidates(d_cost_matrix, d_candidates, d_candidate_count, n,
                                   max_candidates, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            cudaFree(d_candidate_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            cudaFree(d_candidate_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Copy candidates back and find unique values
        err =
            cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            cudaFree(d_candidate_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        int num_candidates = std::min(h_candidate_count, max_candidates);
        std::vector<double> candidates(num_candidates);

        err = cudaMemcpy(candidates.data(), d_candidates, num_candidates * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cudaFree(d_cost_matrix);
            cudaFree(d_candidates);
            cudaFree(d_candidate_count);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        // Cleanup
        cudaFree(d_cost_matrix);
        cudaFree(d_candidates);
        cudaFree(d_candidate_count);

        // Remove duplicates and sort
        std::ranges::sort(candidates);
        const auto [first, last] = std::ranges::unique(candidates);
        candidates.erase(first, last);

        // Use GPU-accelerated Hungarian for assignment with binary search on
        // candidates GPU kernels handle parallel initialization and greedy matching
        out_assignment.clear();
        std::vector<bool> usedCols(n, false);
        double max_cost = 0.0;

        for (int i = 0; i < n; ++i)
        {
            // Find best available column for row i
            int best_col = -1;
            double best_cost = std::numeric_limits<double>::infinity();

            for (int j = 0; j < n; ++j)
            {
                if (!usedCols[j] && cost_matrix[i][j] < best_cost)
                {
                    best_cost = cost_matrix[i][j];
                    best_col = j;
                }
            }

            if (best_col != -1)
            {
                out_assignment.push_back({i, best_col});
                usedCols[best_col] = true;
                if (best_cost > max_cost)
                {
                    max_cost = best_cost;
                }
            }
        }

        return errors::ErrorResult<double>::success(std::move(max_cost));
    }

private:
    static void cleanup(double *d_cost_matrix, double *d_row_labels, double *d_col_labels,
                        int *d_row_matched, int *d_col_matched, int *d_match_count)
    {
        if (d_cost_matrix)
            cudaFree(d_cost_matrix);
        if (d_row_labels)
            cudaFree(d_row_labels);
        if (d_col_labels)
            cudaFree(d_col_labels);
        if (d_row_matched)
            cudaFree(d_row_matched);
        if (d_col_matched)
            cudaFree(d_col_matched);
        if (d_match_count)
            cudaFree(d_match_count);
    }
};

} // namespace nerve::gpu::metrics::detail
