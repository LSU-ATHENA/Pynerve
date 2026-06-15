
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/accelerated/gpu_apparent_pairs.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

namespace nerve::persistence::accelerated
{

namespace
{

// GPU Apparent Pairs Kernel Constants
constexpr int GPU_APPARENT_PAIRS_BLOCK_SIZE = 256; // Threads per block

constexpr double kApparentPairEpsilon = 1e-12;

__device__ bool faceMatchesSimplexWithoutVertex(const int *simplex_vertices,
                                                int simplex_vertex_count, int omitted_vertex_idx,
                                                const int *candidate_face_vertices,
                                                int candidate_face_count)
{
    if (candidate_face_count != simplex_vertex_count - 1)
    {
        return false;
    }

    int simplex_it = 0;
    int face_it = 0;
    while (simplex_it < simplex_vertex_count && face_it < candidate_face_count)
    {
        if (simplex_it == omitted_vertex_idx)
        {
            ++simplex_it;
            continue;
        }
        if (simplex_vertices[simplex_it] != candidate_face_vertices[face_it])
        {
            return false;
        }
        ++simplex_it;
        ++face_it;
    }
    return face_it == candidate_face_count;
}

__device__ bool simplexContainsFaceVertices(const int *simplex_vertices, int simplex_vertex_count,
                                            const int *face_vertices, int face_vertex_count)
{
    int simplex_it = 0;
    int face_it = 0;
    while (simplex_it < simplex_vertex_count && face_it < face_vertex_count)
    {
        if (simplex_vertices[simplex_it] < face_vertices[face_it])
        {
            ++simplex_it;
            continue;
        }
        if (simplex_vertices[simplex_it] > face_vertices[face_it])
        {
            return false;
        }
        ++simplex_it;
        ++face_it;
    }
    return face_it == face_vertex_count;
}

} // namespace

__device__ bool checkApparentPairCondition(const int *vertex_data, const int *vertex_counts,
                                           const int *vertex_offsets,
                                           const double *filtration_values, int simplex_idx,
                                           int n_simplices)
{
    const int simplex_vertex_count = vertex_counts[simplex_idx];
    if (simplex_vertex_count <= 1)
    {
        return false;
    }

    const int simplex_vertex_offset = vertex_offsets[simplex_idx];
    const int *simplex_vertices = vertex_data + simplex_vertex_offset;
    const double simplex_filtration = filtration_values[simplex_idx];

    for (int omitted = 0; omitted < simplex_vertex_count; ++omitted)
    {
        const int face_vertex_count = simplex_vertex_count - 1;
        int matched_face_index = -1;

        for (int face_idx = 0; face_idx < n_simplices; ++face_idx)
        {
            if (vertex_counts[face_idx] != face_vertex_count)
            {
                continue;
            }
            if (filtration_values[face_idx] > simplex_filtration + kApparentPairEpsilon)
            {
                continue;
            }
            const int *face_vertices = vertex_data + vertex_offsets[face_idx];
            if (faceMatchesSimplexWithoutVertex(simplex_vertices, simplex_vertex_count, omitted,
                                                face_vertices, face_vertex_count))
            {
                matched_face_index = face_idx;
                break;
            }
        }

        if (matched_face_index < 0)
        {
            continue;
        }

        if (fabs(filtration_values[matched_face_index] - simplex_filtration) > kApparentPairEpsilon)
        {
            continue;
        }

        const int *matched_face_vertices = vertex_data + vertex_offsets[matched_face_index];
        int equal_filtration_cofaces = 0;
        for (int coface_idx = 0; coface_idx < n_simplices; ++coface_idx)
        {
            if (vertex_counts[coface_idx] != simplex_vertex_count)
            {
                continue;
            }
            if (fabs(filtration_values[coface_idx] - simplex_filtration) > kApparentPairEpsilon)
            {
                continue;
            }
            const int *coface_vertices = vertex_data + vertex_offsets[coface_idx];
            if (simplexContainsFaceVertices(coface_vertices, simplex_vertex_count,
                                            matched_face_vertices, face_vertex_count))
            {
                ++equal_filtration_cofaces;
                if (equal_filtration_cofaces > 1)
                {
                    break;
                }
            }
        }

        if (equal_filtration_cofaces == 1)
        {
            return true;
        }
    }

    return false;
}

__global__ __launch_bounds__(GPU_APPARENT_PAIRS_BLOCK_SIZE) void detectApparentPairsKernel(
    const int *vertex_data, const int *vertex_counts, const int *vertex_offsets,
    const double *filtration_values, int *apparent_pairs, int n_simplices, int determinism_level)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    // Check if this simplex is an apparent pair
    bool is_apparent = checkApparentPairCondition(vertex_data, vertex_counts, vertex_offsets,
                                                  filtration_values, idx, n_simplices);

    if (is_apparent)
    {
        apparent_pairs[idx] = idx; // Mark as apparent pair
    }
    else
    {
        apparent_pairs[idx] = -1;
    }
}

void GPUApparentPairs::detectApparentPairsKernel(const int *vertex_data, const int *vertex_counts,
                                                 const int *vertex_offsets,
                                                 const double *filtration_values,
                                                 int *apparent_pairs, int n_simplices,
                                                 int determinism_level)
{
    dim3 blocks_per_grid;
    dim3 threads_per_block;

    // Configure grid and block dimensions
    threads_per_block = dim3(GPU_APPARENT_PAIRS_BLOCK_SIZE);
    blocks_per_grid = dim3((n_simplices + threads_per_block.x - 1) / threads_per_block.x);

    // Launch kernel
    ::nerve::persistence::accelerated::
        detectApparentPairsKernel<<<blocks_per_grid, threads_per_block>>>(
            vertex_data, vertex_counts, vertex_offsets, filtration_values, apparent_pairs,
            n_simplices, determinism_level);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace nerve::persistence::accelerated
