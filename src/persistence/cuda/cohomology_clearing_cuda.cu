// GPU-accelerated persistent cohomology with proper column reduction and
// dimension-cascade clearing.  The pipeline follows the standard persistent
// cohomology algorithm (de Silva-Morozov, Z2 coefficients):
//
//   Compute coboundary matrix (all dims at once)
//   Dimension-cascade clearing + column reduction
//   Extract pairs from pivot table

#include "nerve/core_types.hpp"
#include "nerve/persistence/cohomology/gpu_cohomology.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <concepts>
#include <limits>
#include <span>
#include <vector>

namespace nerve::persistence::gpu::cohomology
{

template <typename T>
concept CoboundaryComputable = std::integral<T> || std::floating_point<T>;

// True if all vertices of a are in b (both sorted).
__device__ bool isFaceOf(const int *va, int na, const int *vb, int nb)
{
    int ai = 0, bi = 0;
    while (ai < na && bi < nb)
    {
        if (va[ai] == vb[bi])
        {
            ++ai;
            ++bi;
        }
        else if (va[ai] > vb[bi])
        {
            ++bi;
        }
        else
        {
            return false;
        }
    }
    return ai == na;
}

// Symmetric difference of two sorted int arrays into out[0..out_capacity).
// Returns written count.  O(a_size + b_size).
__device__ int symmetricDifference(const int *a, int a_size, const int *b, int b_size, int *out,
                                   int out_capacity)
{
    int oi = 0, ai = 0, bi = 0;
    while (ai < a_size && bi < b_size && oi < out_capacity)
    {
        if (a[ai] < b[bi])
        {
            out[oi++] = a[ai++];
        }
        else if (a[ai] > b[bi])
        {
            out[oi++] = b[bi++];
        }
        else
        {
            ++ai;
            ++bi;
        } // Z2 cancel
    }
    while (ai < a_size && oi < out_capacity)
    {
        out[oi++] = a[ai++];
    }
    while (bi < b_size && oi < out_capacity)
    {
        out[oi++] = b[bi++];
    }
    return oi;
}

/* Kernel: Parallel coboundary computation
 *
 * For each simplex, find all cofaces of dimension+1.
 * Result in flat CSR buffer: cow_buffer[offset .. offset+size-1]. */
__global__ void __launch_bounds__(256)
    computeCoboundaryKernel(const SimplexGPU *__restrict__ simplices, int num_simplices,
                            int *__restrict__ cow_buffer, int *__restrict__ cow_sizes,
                            int *__restrict__ cow_offsets, int max_cow_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_simplices)
        return;

    const SimplexGPU s = simplices[idx];
    const int target_dim = s.dimension + 1;
    if (target_dim > MAX_DIM_GPU_COHOMOLOGY)
    {
        cow_sizes[idx] = 0;
        cow_offsets[idx] = idx * max_cow_size;
        return;
    }

    int base = idx * max_cow_size;
    int count = 0;

    // Forward scan (dim+1 simplices after this one in sorted filtration)
    for (int j = idx + 1; j < num_simplices && count < max_cow_size; ++j)
    {
        const SimplexGPU t = simplices[j];
        if (t.dimension > target_dim)
            break;
        if (t.dimension < target_dim)
            continue;
        if (isFaceOf(s.vertices, s.num_vertices, t.vertices, t.num_vertices))
            cow_buffer[base + count++] = j;
    }
    // Backward scan (tie-broken cofaces with same filtration)
    for (int j = idx - 1; j >= 0 && count < max_cow_size; --j)
    {
        const SimplexGPU t = simplices[j];
        if (t.dimension > target_dim)
            continue;
        if (t.dimension < target_dim)
            break;
        if (t.filtration_value != s.filtration_value)
            break;
        if (isFaceOf(s.vertices, s.num_vertices, t.vertices, t.num_vertices))
            cow_buffer[base + count++] = j;
    }

    cow_sizes[idx] = count;
    cow_offsets[idx] = base;
}

/* Kernel: Clearing optimization
 *
 * Mark columns whose birth simplex was paired in a previous dimension.
 * Uses the pivot table: pivot_table[pivot] = column that claimed pivot.
 * For dimension d, any d-simplex that appears as a pivot in the table
 * has already been paired and its column should be skipped. */
__global__ void __launch_bounds__(256)
    clearingKernel(SimplexGPU *__restrict__ simplices, int num_simplices,
                   const int *__restrict__ pivot_table, int *__restrict__ cow_sizes,
                   int current_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_simplices)
        return;

    SimplexGPU s = simplices[idx];
    if (s.dimension != current_dim)
        return;

    // If this simplex's index appears as a pivot in the table,
    // it was claimed as a death in dimension current_dim-1.
    // Its coboundary column in dimension current_dim is cleared.
    if (idx < num_simplices && pivot_table[idx] != -1)
    {
        cow_sizes[idx] = -1;
        simplices[idx].cleared = true;
        simplices[idx].reduced = true;
    }
}

/* Kernel: Column reduction
 *
 * Each thread reduces one coboundary column using atomicCAS pivot claiming.
 * XOR uses per-thread local memory for symmetric difference.
 * After reduction, writes the updated column back.
 *
 * When current_dim >= 0, only processes columns whose simplex has that
 * dimension (dimension-cascade clearing mode).  Pass -1 to process all
 * columns (legacy single-pass mode, used by reduceCoboundaries()). */
__global__ void __launch_bounds__(256)
    coboundaryReductionKernel(int *__restrict__ column_data, int *__restrict__ column_sizes,
                              int *__restrict__ pivot_table, int num_columns, int max_cow_size,
                              int num_simplices, const SimplexGPU *__restrict__ simplices,
                              int current_dim)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= num_columns)
        return;

    // In dimension-cascade mode, only process columns of this dimension
    if (current_dim >= 0 && simplices[col_idx].dimension != current_dim)
        return;

    int size = column_sizes[col_idx];
    if (size <= 0)
        return;

    // Per-thread local memory (not shared), so 1024-element arrays are safe.
    constexpr int SCRATCH = 1024;
    int col[SCRATCH];
    for (int i = 0; i < size && i < SCRATCH; ++i)
        col[i] = column_data[col_idx * max_cow_size + i];
    int local_size = size;

    constexpr int MAX_ITER = 2000;
    for (int iter = 0; iter < MAX_ITER && local_size > 0; ++iter)
    {
        int pivot = col[local_size - 1];
        if (pivot < 0 || pivot >= num_simplices)
            break;

        // Write reduced column to global memory AND fence BEFORE attempting
        // to claim the pivot.  This fixes a write-after-publish race: if
        // atomicCAS succeeded first, other threads could read our column
        // data before it was fully written.  With write-then-fence-then-CAS,
        // any thread that sees us in the pivot table is guaranteed to see
        // our fully-written column data (release-acquire via atomicCAS).
        int *dst = column_data + col_idx * max_cow_size;
        for (int i = 0; i < local_size; ++i)
            dst[i] = col[i];
        __threadfence();

        // atomicCAS returns the *previous* value at pivot_table[pivot]
        int old_val = atomicCAS(&pivot_table[pivot], -1, col_idx);
        if (old_val == -1)
        {
            // Claimed -- column data and fence already done above.
            // Only now publish the reduced size since we own this pivot.
            column_sizes[col_idx] = local_size;
            return;
        }

        // Unclaimed -- XOR with old_val's column
        const int *other = column_data + old_val * max_cow_size;
        int other_size = column_sizes[old_val];
        if (other_size <= 0)
            break;

        int tmp[SCRATCH];
        int ns = symmetricDifference(col, local_size, other, other_size, tmp, SCRATCH);
        for (int i = 0; i < ns && i < SCRATCH; ++i)
            col[i] = tmp[i];
        local_size = ns;
    }
}

/* Kernel: Apparent pair marking
 *
 * An apparent pair (sigma, tau) in dimension d occurs when:
 *   - sigma is a d-simplex, tau is a (d+1)-simplex
 *   - sigma is a face of tau
 *   - tau is the YOUNGEST cofacet of sigma (highest filtration among cofaces)
 *   - sigma is the OLDEST facet of tau (lowest filtration among facets)
 *
 * This kernel pre-marks such pairs so reduction can skip them. */
__global__ void __launch_bounds__(256)
    apparentPairKernel(const SimplexGPU *__restrict__ simplices, int num_simplices,
                       const int *__restrict__ cow_sizes, const int *__restrict__ cow_offsets,
                       const int *__restrict__ cow_buffer, int *__restrict__ pivot_table,
                       int max_cow_size)
{
    int sigma_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sigma_idx >= num_simplices)
        return;

    const SimplexGPU sigma = simplices[sigma_idx];
    const int d = sigma.dimension;
    if (d < 0 || d + 1 > MAX_DIM_GPU_COHOMOLOGY)
        return;

    int base = cow_offsets[sigma_idx];
    int nc = cow_sizes[sigma_idx];
    if (nc <= 0 || nc > max_cow_size)
        return;

    // Find youngest cofacet: highest filtration (then highest index)
    int youngest_idx = cow_buffer[base];
    double youngest_filt = simplices[youngest_idx].filtration_value;
    for (int i = 1; i < nc; ++i)
    {
        int ci = cow_buffer[base + i];
        double cf = simplices[ci].filtration_value;
        if (cf > youngest_filt)
        {
            youngest_idx = ci;
            youngest_filt = cf;
        }
        // Tie: higher index = later in sorted order = "younger"
        else if (cf == youngest_filt && ci > youngest_idx)
        {
            youngest_idx = ci;
        }
    }

    bool sigma_is_oldest = true;
    const SimplexGPU tau = simplices[youngest_idx];
    for (int other_idx = 0; other_idx < num_simplices; ++other_idx)
    {
        if (other_idx == sigma_idx)
            continue;
        const SimplexGPU other = simplices[other_idx];
        if (other.dimension != d)
            continue;
        if (!isFaceOf(other.vertices, other.num_vertices, tau.vertices, tau.num_vertices))
            continue;
        if (other.filtration_value < sigma.filtration_value ||
            (other.filtration_value == sigma.filtration_value && other.index < sigma.index))
        {
            sigma_is_oldest = false;
            break;
        }
    }

    if (sigma_is_oldest)
    {
        pivot_table[youngest_idx] = sigma_idx;
    }
}

/* Host helper: Build persistence pairs from pivot table
 *
 * The reduction kernel stores pivot_table[pivot_simplex] = column_simplex,
 * i.e. the column (lower-dim) claimed the pivot (higher-dim).
 * This means: birth = column_simplex, death = pivot_simplex.
 *
 * Essential (infinite) pairs: columns that never successfully claimed
 * any pivot (their coboundary reduced to empty). */
static void extractPairs(const SimplexGPU *simplices, int num_simplices, const int *pivot_table,
                         const int *original_sizes, std::vector<Pair> &out_pairs)
{
    out_pairs.clear();

    // Track which simplices are births (claimed a pivot) and which are
    // deaths (claimed as a pivot by some column).
    std::vector<bool> is_birth(num_simplices, false);
    std::vector<bool> is_death(num_simplices, false);
    std::vector<int> birth_col(num_simplices, -1); // birth_col[p] = column that claimed pivot p

    for (int p = 0; p < num_simplices; ++p)
    {
        int c = pivot_table[p];
        if (c >= 0 && c < num_simplices)
        {
            is_death[p] = true;
            is_birth[c] = true;
            birth_col[p] = c;
        }
    }

    // Finite pairs: (birth = column c, death = pivot p)
    for (int p = 0; p < num_simplices; ++p)
    {
        int c = birth_col[p];
        if (c < 0)
            continue;

        Pair pair{};
        pair.birth_index = c;
        pair.death_index = p;
        pair.dimension = simplices[c].dimension;
        pair.birth = simplices[c].filtration_value;
        pair.death = simplices[p].filtration_value;
        if (pair.death <= pair.birth)
            continue;
        out_pairs.push_back(pair);
    }

    // Essential classes: columns that had a non-empty coboundary (original_sizes > 0)
    // but reduced to empty (no pivot claimed, not a death partner).
    // Simplices with original_sizes == 0 (naturally empty coboundary) are
    // not essential -- they're unpaired top-dimension simplices.
    for (int i = 0; i < num_simplices; ++i)
    {
        if (original_sizes[i] <= 0)
            continue;
        if (is_birth[i])
            continue;
        if (is_death[i])
            continue;

        // Had a coboundary but reduced to empty -- this column is essential
        Pair pair{};
        pair.birth_index = i;
        pair.death_index = -1;
        pair.dimension = simplices[i].dimension;
        pair.birth = simplices[i].filtration_value;
        pair.death = std::numeric_limits<double>::infinity();
        out_pairs.push_back(pair);
    }
}

GPUCohomologyComputer::GPUCohomologyComputer()
    : initialized_(false)
    , last_compute_time_ms_(0.0)
    , cpu_baseline_ms_(0.0)
{}

GPUCohomologyComputer::~GPUCohomologyComputer() = default;

bool GPUCohomologyComputer::initialize(size_t max_simplices, size_t max_dim)
{
    if (max_simplices == 0 || max_dim == 0 || max_dim > MAX_DIM_GPU_COHOMOLOGY)
        return initialized_ = false;
    try
    {
        d_simplices_.resize(max_simplices);
        d_coboundary_buffer_.resize(max_simplices * COBOUNDARY_MAX_SIZE);
        d_coboundary_sizes_.resize(max_simplices);
        d_coboundary_offsets_.resize(max_simplices);
        d_pivot_table_.resize(max_simplices);
        d_red_sizes_.resize(max_simplices);
        initialized_ = true;
        return true;
    }
    catch (...)
    {
        return initialized_ = false;
    }
}

bool GPUCohomologyComputer::computePersistenceDiagram(std::span<const SimplexGPU> simplices,
                                                      std::vector<Pair> &persistence_pairs)
{
    if (!initialized_)
        return false;
    int N = static_cast<int>(simplices.size());
    if (N == 0)
        return false;
    if (static_cast<size_t>(N) > d_simplices_.size())
        return false;

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::copy(simplices.begin(), simplices.end(), d_simplices_.begin());

    const int BS = COHOMOLOGY_BLOCK_SIZE;
    const int GS = (N + BS - 1) / BS;

    thrust::fill(d_coboundary_sizes_.begin(), d_coboundary_sizes_.end(), 0);
    thrust::fill(d_pivot_table_.begin(), d_pivot_table_.end(), -1);

    computeCoboundaryKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_simplices_.data()), N,
                                        thrust::raw_pointer_cast(d_coboundary_buffer_.data()),
                                        thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
                                        thrust::raw_pointer_cast(d_coboundary_offsets_.data()),
                                        COBOUNDARY_MAX_SIZE);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    // Working copies of coboundary data (modified during reduction,
    // while originals stay intact for the essential-class size check).
    d_red_sizes_ = d_coboundary_sizes_;
    thrust::device_vector<int> d_red_data = d_coboundary_buffer_;

    // Apparent pair detection -- pre-populates pivot table so that
    // reduction naturally finds these pairs without iteration.
    apparentPairKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_simplices_.data()), N,
                                   thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
                                   thrust::raw_pointer_cast(d_coboundary_offsets_.data()),
                                   thrust::raw_pointer_cast(d_coboundary_buffer_.data()),
                                   thrust::raw_pointer_cast(d_pivot_table_.data()),
                                   COBOUNDARY_MAX_SIZE);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    int max_dim = 0;
    for (const auto &s : simplices)
    {
        if (s.dimension > max_dim)
            max_dim = s.dimension;
        if (max_dim >= MAX_DIM_GPU_COHOMOLOGY)
            break;
    }

    // Dimension-cascade reduction: process one dimension at a time
    // so clearing can skip columns whose simplices were already paired
    // as deaths in the previous dimension.
    for (int d = 0; d <= max_dim; ++d)
    {
        // Clearing: mark any dim-d column whose simplex index appears
        // as a pivot in the table (paired as death in dimension d-1).
        clearingKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_simplices_.data()), N,
                                   thrust::raw_pointer_cast(d_pivot_table_.data()),
                                   thrust::raw_pointer_cast(d_red_sizes_.data()), d);
        if (cudaGetLastError() != cudaSuccess)
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }

        coboundaryReductionKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_red_data.data()),
                                              thrust::raw_pointer_cast(d_red_sizes_.data()),
                                              thrust::raw_pointer_cast(d_pivot_table_.data()), N,
                                              COBOUNDARY_MAX_SIZE, N,
                                              thrust::raw_pointer_cast(d_simplices_.data()), d);
        if (cudaGetLastError() != cudaSuccess)
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }
    }

    cudaDeviceSynchronize();

    std::vector<int> h_pivot(static_cast<size_t>(N), -1);
    if (cudaMemcpy(h_pivot.data(), thrust::raw_pointer_cast(d_pivot_table_.data()),
                   static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    std::vector<int> h_orig_sizes(static_cast<size_t>(N), 0);
    if (cudaMemcpy(h_orig_sizes.data(), thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
                   static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    thrust::host_vector<SimplexGPU> h_simplices = d_simplices_;
    extractPairs(h_simplices.data(), N, h_pivot.data(), h_orig_sizes.data(), persistence_pairs);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    last_compute_time_ms_ = static_cast<double>(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return true;
}

double GPUCohomologyComputer::getLastComputeTime() const noexcept
{
    return last_compute_time_ms_;
}

double GPUCohomologyComputer::getSpeedupVsCPU() const noexcept
{
    if (cpu_baseline_ms_ <= 0.0 || last_compute_time_ms_ <= 0.0)
        return 0.0;
    return cpu_baseline_ms_ / last_compute_time_ms_;
}

void GPUCohomologyComputer::setCPUBaseline(double cpu_ms) noexcept
{
    cpu_baseline_ms_ = cpu_ms;
}

bool GPUCohomologyComputer::getClearedStates(std::vector<bool> &out_cleared) const
{
    if (!initialized_)
        return false;
    std::size_t N = static_cast<std::size_t>(d_simplices_.size());
    if (N == 0)
        return false;

    thrust::host_vector<SimplexGPU> h_simplices = d_simplices_;
    out_cleared.resize(N);
    for (std::size_t i = 0; i < N; ++i)
        out_cleared[i] = h_simplices[i].cleared;
    return true;
}

bool GPUCohomologyComputer::getReducedColumnSizes(std::vector<int> &out_sizes) const
{
    if (!initialized_)
        return false;
    std::size_t N = static_cast<std::size_t>(d_red_sizes_.size());
    if (N == 0)
        return false;

    out_sizes.resize(N);
    cudaError_t err = cudaMemcpy(out_sizes.data(), thrust::raw_pointer_cast(d_red_sizes_.data()),
                                 N * sizeof(int), cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
}

bool GPUCohomologyComputer::computeCoboundaries(int num_simplices)
{
    int BS = COHOMOLOGY_BLOCK_SIZE;
    int GS = (num_simplices + BS - 1) / BS;
    computeCoboundaryKernel<<<GS, BS>>>(
        thrust::raw_pointer_cast(d_simplices_.data()), num_simplices,
        thrust::raw_pointer_cast(d_coboundary_buffer_.data()),
        thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
        thrust::raw_pointer_cast(d_coboundary_offsets_.data()), COBOUNDARY_MAX_SIZE);
    return cudaGetLastError() == cudaSuccess;
}

bool GPUCohomologyComputer::performClearing(int num_simplices)
{
    int BS = COHOMOLOGY_BLOCK_SIZE;
    int GS = (num_simplices + BS - 1) / BS;
    clearingKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_simplices_.data()), num_simplices,
                               thrust::raw_pointer_cast(d_pivot_table_.data()),
                               thrust::raw_pointer_cast(d_coboundary_sizes_.data()), 0);
    return cudaGetLastError() == cudaSuccess;
}

bool GPUCohomologyComputer::detectApparentPairs(int num_simplices)
{
    int BS = COHOMOLOGY_BLOCK_SIZE;
    int GS = (num_simplices + BS - 1) / BS;
    apparentPairKernel<<<GS, BS>>>(thrust::raw_pointer_cast(d_simplices_.data()), num_simplices,
                                   thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
                                   thrust::raw_pointer_cast(d_coboundary_offsets_.data()),
                                   thrust::raw_pointer_cast(d_coboundary_buffer_.data()),
                                   thrust::raw_pointer_cast(d_pivot_table_.data()),
                                   COBOUNDARY_MAX_SIZE);
    return cudaGetLastError() == cudaSuccess;
}

bool GPUCohomologyComputer::reduceCoboundaries(int num_simplices)
{
    int BS = COHOMOLOGY_BLOCK_SIZE;
    int GS = (num_simplices + BS - 1) / BS;
    coboundaryReductionKernel<<<GS, BS>>>(
        thrust::raw_pointer_cast(d_coboundary_buffer_.data()),
        thrust::raw_pointer_cast(d_coboundary_sizes_.data()),
        thrust::raw_pointer_cast(d_pivot_table_.data()), num_simplices, COBOUNDARY_MAX_SIZE,
        num_simplices, thrust::raw_pointer_cast(d_simplices_.data()), -1); // -1 = process all dims
    return cudaGetLastError() == cudaSuccess;
}

/* High-level API: run full GPU cohomology pipeline on SimplexGPU data */
PersistenceDiagram computeGPUCohomology(const std::vector<SimplexGPU> &simplices, int max_dimension)
{
    GPUCohomologyComputer computer;
    if (!computer.initialize(simplices.size(), static_cast<size_t>(max_dimension)))
    {
        return PersistenceDiagram{};
    }

    std::vector<Pair> pairs;
    if (!computer.computePersistenceDiagram(simplices, pairs))
    {
        return PersistenceDiagram{};
    }

    PersistenceDiagram diagram;
    diagram.pairs.reserve(pairs.size());
    for (const auto &p : pairs)
    {
        diagram.pairs.emplace_back(p.birth, p.death, p.dimension);
    }
    return diagram;
}

} // namespace nerve::persistence::gpu::cohomology
