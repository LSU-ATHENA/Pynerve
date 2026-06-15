
#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace nerve
{
namespace gpu
{
namespace algebra
{
namespace kernels
{
constexpr int CECH_BLOCK_SIZE = 256;
constexpr int CECH_MAX_INT_PAIR_POINTS = 65536;
constexpr double CECH_DIAMETER_FACTOR =
    2.0; // Balls intersect when distance < 2 * radius (diameter)

namespace
{

int checkedPairCount(int n_points, const char *context)
{
    if (n_points < 0 || n_points > CECH_MAX_INT_PAIR_POINTS)
    {
        throw std::length_error(context);
    }
    return n_points * (n_points - 1) / 2;
}

int checkedGridSize(int count, int block_size, const char *context)
{
    if (count < 0 || block_size <= 0)
    {
        throw std::invalid_argument(context);
    }
    const int grid_size = (count + block_size - 1) / block_size;
    if (grid_size < 0)
    {
        throw std::length_error(context);
    }
    return grid_size;
}

} // namespace

__global__ void __launch_bounds__(256)
    cechBallIntersectionKernel(const double *__restrict__ points,    // [n_points x dim]
                               int *__restrict__ intersection_graph, // [n_points x n_points] output
                               int n_points, int point_dim, double radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_points * (n_points - 1) / 2;

    if (idx >= total_pairs)
        return;
    int i = 0, j = 0, count = 0;
    for (int row = 0; row < n_points; ++row)
    {
        int row_pairs = n_points - row - 1;
        if (idx < count + row_pairs)
        {
            i = row;
            j = row + 1 + (idx - count);
            break;
        }
        count += row_pairs;
    }

    if (i >= j)
        return;
    double dist_sq = 0.0;
    for (int d = 0; d < point_dim; ++d)
    {
        double diff = points[i * point_dim + d] - points[j * point_dim + d];
        double contribution = diff * diff;
        double next_dist_sq = dist_sq + contribution;
        if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
        {
            dist_sq = INFINITY;
            break;
        }
        dist_sq = next_dist_sq;
    }
    double dist = sqrt(dist_sq);
    if (dist < CECH_DIAMETER_FACTOR * radius)
    {
        intersection_graph[i * n_points + j] = 1;
        intersection_graph[j * n_points + i] = 1;
    }
}
__global__ void __launch_bounds__(256)
    cechSimplexValidationKernel(const double *__restrict__ points,
                                const int *__restrict__ simplex_vertices, // [n_simplices x max_dim]
                                const int *__restrict__ simplex_sizes,    // [n_simplices]
                                int *__restrict__ isValid,                // [n_simplices] output
                                double *__restrict__ filtration_values,   // [n_simplices] output
                                int n_simplices, int max_dim, int point_dim, double radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    int size = simplex_sizes[idx];
    if (size == 0)
    {
        isValid[idx] = 0;
        return;
    }

    double max_pairwise_dist = 0.0;

    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            int v_i = simplex_vertices[idx * max_dim + i];
            int v_j = simplex_vertices[idx * max_dim + j];

            double dist_sq = 0.0;
            for (int d = 0; d < point_dim; ++d)
            {
                double diff = points[v_i * point_dim + d] - points[v_j * point_dim + d];
                double contribution = diff * diff;
                double next_dist_sq = dist_sq + contribution;
                if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                {
                    dist_sq = INFINITY;
                    break;
                }
                dist_sq = next_dist_sq;
            }
            double dist = sqrt(dist_sq);

            if (dist > max_pairwise_dist)
            {
                max_pairwise_dist = dist;
            }
        }
    }
    double cech_radius = max_pairwise_dist / 2.0;
    isValid[idx] = (cech_radius <= radius) ? 1 : 0;
    filtration_values[idx] = cech_radius;
}
__global__ void __launch_bounds__(256)
    alphaFiltrationKernel(const double *__restrict__ points,
                          const int *__restrict__ simplex_vertices,
                          const int *__restrict__ simplex_sizes, double *__restrict__ alpha_values,
                          int n_simplices, int max_dim, int point_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    int size = simplex_sizes[idx];
    if (size == 0)
    {
        alpha_values[idx] = 0.0;
        return;
    }

    if (size == 2)
    {
        int v0 = simplex_vertices[idx * max_dim + 0];
        int v1 = simplex_vertices[idx * max_dim + 1];
        double dist_sq = 0.0;
        for (int d = 0; d < point_dim; ++d)
        {
            double diff = points[v0 * point_dim + d] - points[v1 * point_dim + d];
            double contribution = diff * diff;
            double next_dist_sq = dist_sq + contribution;
            if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
            {
                dist_sq = INFINITY;
                break;
            }
            dist_sq = next_dist_sq;
        }
        alpha_values[idx] = sqrt(dist_sq) * 0.5;
    }
    else if (size == 3 && point_dim >= 2)
    {
        int v0 = simplex_vertices[idx * max_dim + 0];
        int v1 = simplex_vertices[idx * max_dim + 1];
        int v2 = simplex_vertices[idx * max_dim + 2];

        double a_sq = 0.0, b_sq = 0.0, c_sq = 0.0;
        for (int d = 0; d < point_dim; ++d)
        {
            double p0 = points[v0 * point_dim + d];
            double p1 = points[v1 * point_dim + d];
            double p2 = points[v2 * point_dim + d];
            double da = p1 - p2;
            double db = p0 - p2;
            double dc = p0 - p1;
            double next_a = a_sq + da * da;
            double next_b = b_sq + db * db;
            double next_c = c_sq + dc * dc;
            if (!isfinite(da) || !isfinite(db) || !isfinite(dc) || !isfinite(next_a) ||
                !isfinite(next_b) || !isfinite(next_c))
            {
                alpha_values[idx] = INFINITY;
                return;
            }
            a_sq = next_a;
            b_sq = next_b;
            c_sq = next_c;
        }
        double a = sqrt(a_sq), b = sqrt(b_sq), c = sqrt(c_sq);
        double s = (a + b + c) * 0.5; // semi-perimeter
        double area = sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)));

        if (area > 1e-10)
        {
            alpha_values[idx] = (a * b * c) / (4.0 * area);
        }
        else
        {
            alpha_values[idx] = max(a, max(b, c)) * 0.5;
        }
        if (!isfinite(alpha_values[idx]))
        {
            alpha_values[idx] = INFINITY;
        }
    }
    else if (size == 4 && point_dim >= 3)
    {
        int v[4];
        double edges[6]; // d01, d02, d03, d12, d13, d23
        int edge_idx = 0;

        for (int i = 0; i < 4; ++i)
        {
            v[i] = simplex_vertices[idx * max_dim + i];
        }
        for (int i = 0; i < 4; ++i)
        {
            for (int j = i + 1; j < 4; ++j)
            {
                double dist_sq = 0.0;
                for (int d = 0; d < point_dim; ++d)
                {
                    double diff = points[v[i] * point_dim + d] - points[v[j] * point_dim + d];
                    double contribution = diff * diff;
                    double next_dist_sq = dist_sq + contribution;
                    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                    {
                        dist_sq = INFINITY;
                        break;
                    }
                    dist_sq = next_dist_sq;
                }
                edges[edge_idx++] = sqrt(dist_sq);
            }
        }

        double d01 = edges[0], d02 = edges[1], d03 = edges[2];
        double d12 = edges[3], d13 = edges[4], d23 = edges[5];

        double cm[5][5];
        cm[0][0] = 0.0;
        cm[0][1] = 1.0;
        cm[0][2] = 1.0;
        cm[0][3] = 1.0;
        cm[0][4] = 1.0;
        cm[1][0] = 1.0;
        cm[1][1] = 0.0;
        cm[1][2] = d01 * d01;
        cm[1][3] = d02 * d02;
        cm[1][4] = d03 * d03;
        cm[2][0] = 1.0;
        cm[2][1] = d01 * d01;
        cm[2][2] = 0.0;
        cm[2][3] = d12 * d12;
        cm[2][4] = d13 * d13;
        cm[3][0] = 1.0;
        cm[3][1] = d02 * d02;
        cm[3][2] = d12 * d12;
        cm[3][3] = 0.0;
        cm[3][4] = d23 * d23;
        cm[4][0] = 1.0;
        cm[4][1] = d03 * d03;
        cm[4][2] = d13 * d13;
        cm[4][3] = d23 * d23;
        cm[4][4] = 0.0;
        double det = 0.0;
        for (int col = 0; col < 5; ++col)
        {
            double minor[4][4];
            int minor_row = 0;
            for (int r = 1; r < 5; ++r)
            {
                int minor_col = 0;
                for (int c = 0; c < 5; ++c)
                {
                    if (c != col)
                    {
                        minor[minor_row][minor_col++] = cm[r][c];
                    }
                }
                minor_row++;
            }
            double minor_det =
                minor[0][0] *
                    (minor[1][1] * (minor[2][2] * minor[3][3] - minor[2][3] * minor[3][2]) -
                     minor[1][2] * (minor[2][1] * minor[3][3] - minor[2][3] * minor[3][1]) +
                     minor[1][3] * (minor[2][1] * minor[3][2] - minor[2][2] * minor[3][1])) -
                minor[0][1] *
                    (minor[1][0] * (minor[2][2] * minor[3][3] - minor[2][3] * minor[3][2]) -
                     minor[1][2] * (minor[2][0] * minor[3][3] - minor[2][3] * minor[3][0]) +
                     minor[1][3] * (minor[2][0] * minor[3][2] - minor[2][2] * minor[3][0])) +
                minor[0][2] *
                    (minor[1][0] * (minor[2][1] * minor[3][3] - minor[2][3] * minor[3][1]) -
                     minor[1][1] * (minor[2][0] * minor[3][3] - minor[2][3] * minor[3][0]) +
                     minor[1][3] * (minor[2][0] * minor[3][1] - minor[2][1] * minor[3][0])) -
                minor[0][3] *
                    (minor[1][0] * (minor[2][1] * minor[3][2] - minor[2][2] * minor[3][1]) -
                     minor[1][1] * (minor[2][0] * minor[3][2] - minor[2][2] * minor[3][0]) +
                     minor[1][2] * (minor[2][0] * minor[3][1] - minor[2][1] * minor[3][0]));

            int sign = (col % 2 == 0) ? 1 : -1;
            det += sign * cm[0][col] * minor_det;
        }

        double volume_squared = det / 288.0;
        if (volume_squared > 1e-15)
        {
            double volume = sqrt(volume_squared);
            double circumradius = sqrt(det) / (12.0 * sqrt(2.0) * volume);
            alpha_values[idx] = circumradius;
        }
        else
        {
            double max_edge = 0.0;
            for (int i = 0; i < 6; ++i)
            {
                max_edge = max(max_edge, edges[i]);
            }
            alpha_values[idx] = max_edge * 0.5;
        }
        if (!isfinite(alpha_values[idx]))
        {
            alpha_values[idx] = INFINITY;
        }
    }
    else
    {
        double max_edge = 0.0;
        for (int i = 0; i < size; ++i)
        {
            for (int j = i + 1; j < size; ++j)
            {
                int v_i = simplex_vertices[idx * max_dim + i];
                int v_j = simplex_vertices[idx * max_dim + j];
                double dist_sq = 0.0;
                for (int d = 0; d < point_dim; ++d)
                {
                    double diff = points[v_i * point_dim + d] - points[v_j * point_dim + d];
                    double contribution = diff * diff;
                    double next_dist_sq = dist_sq + contribution;
                    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                    {
                        dist_sq = INFINITY;
                        break;
                    }
                    dist_sq = next_dist_sq;
                }
                max_edge = max(max_edge, sqrt(dist_sq));
            }
        }
        alpha_values[idx] = max_edge * 0.5;
    }
}
__global__ void __launch_bounds__(256)
    cechFilterByRadiusKernel(const double *__restrict__ filtration_values,
                             int *__restrict__ isValid, double max_radius, int n_simplices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    isValid[idx] = (filtration_values[idx] <= max_radius) ? 1 : 0;
}
__global__ void __launch_bounds__(256)
    minimalEnclosingBallKernel(const double *__restrict__ points,
                               const int *__restrict__ simplex_vertices,
                               const int *__restrict__ simplex_sizes,
                               double *__restrict__ meb_radii, int n_simplices, int max_dim,
                               int point_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    int size = simplex_sizes[idx];
    if (size == 0)
    {
        meb_radii[idx] = 0.0;
        return;
    }

    double max_dist = 0.0;

    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            int v_i = simplex_vertices[idx * max_dim + i];
            int v_j = simplex_vertices[idx * max_dim + j];

            double dist_sq = 0.0;
            for (int d = 0; d < point_dim; ++d)
            {
                double diff = points[v_i * point_dim + d] - points[v_j * point_dim + d];
                double contribution = diff * diff;
                double next_dist_sq = dist_sq + contribution;
                if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                {
                    dist_sq = INFINITY;
                    break;
                }
                dist_sq = next_dist_sq;
            }
            double dist = sqrt(dist_sq);

            if (dist > max_dist)
            {
                max_dist = dist;
            }
        }
    }
    meb_radii[idx] = max_dist / 2.0;
}
void launchCechBallIntersection(const double *d_points, int *d_intersection_graph, int n_points,
                                int point_dim, double radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_intersection_graph == nullptr || n_points <= 1 || point_dim <= 0 ||
        !std::isfinite(radius) || radius < 0.0 || !std::isfinite(radius * CECH_DIAMETER_FACTOR))
    {
        return;
    }
    int total_pairs = checkedPairCount(n_points, "Cech edge pair count exceeds int range");
    int block_size = CECH_BLOCK_SIZE;
    int grid_size = checkedGridSize(total_pairs, block_size, "Cech edge grid exceeds CUDA limits");

    cechBallIntersectionKernel<<<grid_size, block_size, 0, stream>>>(d_points, d_intersection_graph,
                                                                     n_points, point_dim, radius);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchCechSimplexValidation(const double *d_points, const int *d_simplex_vertices,
                                 const int *d_simplex_sizes, int *d_is_valid,
                                 double *d_filtration_values, int n_simplices, int max_dim,
                                 int point_dim, double radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_simplex_vertices == nullptr || d_simplex_sizes == nullptr ||
        d_is_valid == nullptr || d_filtration_values == nullptr || n_simplices <= 0 ||
        max_dim <= 0 || point_dim <= 0 || !std::isfinite(radius) || radius < 0.0)
    {
        return;
    }
    int block_size = CECH_BLOCK_SIZE;
    int grid_size =
        checkedGridSize(n_simplices, block_size, "Cech validation grid exceeds CUDA limits");

    cechSimplexValidationKernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_simplex_vertices, d_simplex_sizes, d_is_valid, d_filtration_values, n_simplices,
        max_dim, point_dim, radius);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchAlphaFiltration(const double *d_points, const int *d_simplex_vertices,
                           const int *d_simplex_sizes, double *d_alpha_values, int n_simplices,
                           int max_dim, int point_dim, cudaStream_t stream)
{
    if (d_points == nullptr || d_simplex_vertices == nullptr || d_simplex_sizes == nullptr ||
        d_alpha_values == nullptr || n_simplices <= 0 || max_dim <= 0 || point_dim <= 0)
    {
        return;
    }
    int block_size = CECH_BLOCK_SIZE;
    int grid_size = checkedGridSize(n_simplices, block_size, "Cech alpha grid exceeds CUDA limits");

    alphaFiltrationKernel<<<grid_size, block_size, 0, stream>>>(d_points, d_simplex_vertices,
                                                                d_simplex_sizes, d_alpha_values,
                                                                n_simplices, max_dim, point_dim);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchMinimalEnclosingBall(const double *d_points, const int *d_simplex_vertices,
                                const int *d_simplex_sizes, double *d_meb_radii, int n_simplices,
                                int max_dim, int point_dim, cudaStream_t stream)
{
    if (d_points == nullptr || d_simplex_vertices == nullptr || d_simplex_sizes == nullptr ||
        d_meb_radii == nullptr || n_simplices <= 0 || max_dim <= 0 || point_dim <= 0)
    {
        return;
    }
    int block_size = CECH_BLOCK_SIZE;
    int grid_size = checkedGridSize(n_simplices, block_size,
                                    "Cech minimal enclosing ball grid exceeds CUDA limits");

    minimalEnclosingBallKernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_simplex_vertices, d_simplex_sizes, d_meb_radii, n_simplices, max_dim,
        point_dim);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace algebra
} // namespace gpu
} // namespace nerve
