
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/accelerated/gpu_apparent_pairs.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <concepts>
#include <span>

namespace nerve::persistence::accelerated
{

constinit const int MAX_APPARENT_PAIRS_BLOCK_SIZE = 256;
constinit const int MAX_VERTEX_COUNT = 11;
constinit const int MAX_FACE_VERTICES = 10;

template <typename T>
concept ValidSimplexDim = std::integral<T> && requires(T d) {
    { d } -> std::convertible_to<int>;
    requires d >= 0 && d <= 10;
};

__device__ bool isApparentPairRippser(const SimplexGPU *filtration, const int *vertex_data,
                                      const int *vertex_counts, const int *vertex_offsets,
                                      const double *filtration_values, int simplex_idx,
                                      int n_simplices, int dimension)
{
    int vertex_count = vertex_counts[simplex_idx];
    int vertex_offset = vertex_offsets[simplex_idx];
    double simplex_value = filtration_values[simplex_idx];

    constexpr int min_dim_for_apparent = 1;

    if (dimension < min_dim_for_apparent)
    {
        return false;
    }

    if (vertex_count > MAX_VERTEX_COUNT)
    {
        return false;
    }

    for (int face_idx = 0; face_idx < vertex_count; ++face_idx)
    {
        // Build face by removing one vertex
        int face_vertex_count = vertex_count - 1;

        // Create face vertex list
        int face_vertices[10]; // Maximum dimension support
        int face_vertex_idx = 0;

        // Bounds check: ensure we don't overflow face_vertices array
        for (int v = 0; v < vertex_count && face_vertex_idx < 10; ++v)
        {
            if (v != face_idx)
            {
                face_vertices[face_vertex_idx++] = vertex_data[vertex_offset + v];
            }
        }

        // Sort face vertices for consistent comparison
        for (int i = 0; i < face_vertex_count - 1; ++i)
        {
            for (int j = i + 1; j < face_vertex_count; ++j)
            {
                if (face_vertices[i] > face_vertices[j])
                {
                    int temp = face_vertices[i];
                    face_vertices[i] = face_vertices[j];
                    face_vertices[j] = temp;
                }
            }
        }

        // Search for face in filtration
        for (int other_idx = 0; other_idx < n_simplices; ++other_idx)
        {
            if (other_idx == simplex_idx)
                continue;

            int other_vertex_count = vertex_counts[other_idx];
            if (other_vertex_count != face_vertex_count)
                continue;

            int other_vertex_offset = vertex_offsets[other_idx];

            // Check if this simplex represents the face
            bool is_face = true;
            for (int v = 0; v < face_vertex_count; ++v)
            {
                if (face_vertices[v] != vertex_data[other_vertex_offset + v])
                {
                    is_face = false;
                    break;
                }
            }

            if (is_face)
            {
                double face_value = filtration_values[other_idx];

                // Apparent pair condition: face has same filtration value
                // and no other simplex with same face has smaller value
                if (face_value == simplex_value)
                {
                    return true;
                }

                // Check if this is the smallest simplex with this face
                bool is_smallest = true;
                for (int check_idx = 0; check_idx < n_simplices; ++check_idx)
                {
                    if (check_idx == simplex_idx || check_idx == other_idx)
                        continue;

                    int check_vertex_count = vertex_counts[check_idx];
                    if (check_vertex_count != face_vertex_count)
                        continue;

                    int check_vertex_offset = vertex_offsets[check_idx];

                    bool check_is_face = true;
                    for (int v = 0; v < face_vertex_count; ++v)
                    {
                        if (face_vertices[v] != vertex_data[check_vertex_offset + v])
                        {
                            check_is_face = false;
                            break;
                        }
                    }

                    if (check_is_face && filtration_values[check_idx] < simplex_value)
                    {
                        is_smallest = false;
                        break;
                    }
                }

                if (is_smallest)
                {
                    return true;
                }
            }
        }
    }

    return false;
}

// Advanced kernel for apparent pair detection with fast++ optimization
__global__ void __launch_bounds__(256)
    detectApparentPairsAdvancedKernel(const int *vertex_data, const int *vertex_counts,
                                      const int *vertex_offsets, const double *filtration_values,
                                      int *apparent_pairs, int *pair_types, int n_simplices,
                                      int max_dimension, int determinism_level)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    int vertex_count = vertex_counts[idx];
    int dimension = vertex_count - 1;

    // Check if this simplex is an apparent pair
    bool is_apparent = isApparentPairRippser(nullptr, vertex_data, vertex_counts, vertex_offsets,
                                             filtration_values, idx, n_simplices, dimension);

    if (is_apparent)
    {
        apparent_pairs[idx] = idx;   // Mark as apparent pair
        pair_types[idx] = dimension; // Store dimension
    }
    else
    {
        apparent_pairs[idx] = -1;
        pair_types[idx] = -1;
    }
}

// Kernel for matrix reduction with clearing optimization
__global__ void __launch_bounds__(256)
    gpuMatrixReductionKernel(const int *columns_data, const int *column_offsets,
                             const int *column_sizes, int *low_row_to_col, int *col_pivot,
                             int *clear_flags, int n_columns, int n_rows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns || clear_flags[col])
        return;

    int col_start = column_offsets[col];
    int col_size = column_sizes[col];

    // Process column reduction (based on existing fast_vr.cpp patterns)
    while (col_size > 0)
    {
        int pivot = columns_data[col_start + col_size - 1];
        int existing = low_row_to_col[pivot];

        if (existing == -1)
        {
            // Found new pivot - use atomic operation for thread safety
            // Only one thread should succeed in claiming this pivot row
            int old_val = atomicCAS(&low_row_to_col[pivot], -1, col);
            if (old_val == -1)
            {
                // This thread successfully claimed the pivot
                col_pivot[col] = pivot;
                break;
            }
            // Another thread claimed it first, continue elimination
        }

        // Approximate elimination in sorted-column representation:
        // remove all trailing entries equal to the collided pivot.
        while (col_size > 0 && columns_data[col_start + col_size - 1] == pivot)
        {
            --col_size;
        }
    }

    if (col_size == 0)
    {
        col_pivot[col] = -1;
    }
}

// Kernel for clearing optimization
__global__ void __launch_bounds__(256)
    gpuClearingKernel(const int *col_pivot, int *clear_flags, int n_columns, int target_dimension)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
        return;

    int pivot = col_pivot[col];
    if (pivot != -1)
    {
        // Mark corresponding row for clearing (based on clear_cols pattern)
        clear_flags[pivot] = 1;
    }
}

// Advanced kernel for work distribution optimization
__global__ void __launch_bounds__(256)
    computeWorkDistributionKernel(const int *apparent_pairs, const int *pair_types,
                                  int *gpu_work_count, int *cpu_work_count, int n_simplices,
                                  double gpu_ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    if (apparent_pairs[idx] != -1)
    {
        // Apparent pairs go to GPU (99.9% of work)
        atomicAdd(gpu_work_count, 1);
    }
    else
    {
        // Non-apparent pairs go to CPU (0.1% of work)
        atomicAdd(cpu_work_count, 1);
    }
}



} // namespace nerve::persistence::accelerated
