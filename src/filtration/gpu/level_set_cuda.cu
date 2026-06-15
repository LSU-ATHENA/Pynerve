#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::filtration::levelset::gpu
{

constexpr int BLOCK_DIM_X = 16, BLOCK_DIM_Y = 16, BLOCK_DIM_Z = 4;
constexpr int MARCHING_CUBES_BLOCK_DIM = 8, MAX_TRIANGLES_PER_CELL = 5;
constexpr int FLOATS_PER_TRIANGLE = 9, INDICES_PER_TRIANGLE = 3;

namespace
{

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs, const char *message)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

std::size_t checkedElementCount(int nx, int ny, int nz)
{
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        throw std::invalid_argument("GPU level set dimensions must be positive");
    }
    const auto xy = checkedProduct(static_cast<std::size_t>(nx), static_cast<std::size_t>(ny),
                                   "GPU level set grid size exceeds host limits");
    return checkedProduct(xy, static_cast<std::size_t>(nz),
                          "GPU level set grid size exceeds host limits");
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

void validatePhi(const std::vector<float> &phi, int nx, int ny, int nz)
{
    const std::size_t expected = checkedElementCount(nx, ny, nz);
    if (phi.size() != expected)
    {
        throw std::invalid_argument("GPU level set field size does not match grid dimensions");
    }
    if (!valuesAreFinite(phi))
    {
        throw std::invalid_argument("GPU level set field must contain only finite values");
    }
}

void requireFiniteOutput(const std::vector<float> &values, const char *message)
{
    if (!valuesAreFinite(values))
    {
        throw std::runtime_error(message);
    }
}

} // namespace

__global__ void __launch_bounds__(1024)
    levelSetReinitializeKernel(const float *__restrict__ phi_old, float *__restrict__ phi_new,
                               int nx, int ny, int nz, float dx, float dy, float dz, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz)
        return;

    int idx = x + nx * (y + ny * z);
    float phi0 = phi_old[idx];

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
    {
        phi_new[idx] = phi0;
        return;
    }

    float phi_x_plus = (phi_old[idx + 1] - phi0) / dx;
    float phi_x_minus = (phi0 - phi_old[idx - 1]) / dx;
    float phi_y_plus = (phi_old[idx + nx] - phi0) / dy;
    float phi_y_minus = (phi0 - phi_old[idx - nx]) / dy;
    float phi_z_plus = (phi_old[idx + nx * ny] - phi0) / dz;
    float phi_z_minus = (phi0 - phi_old[idx - nx * ny]) / dz;

    float phi_x = (phi0 > 0) ? phi_x_minus : phi_x_plus;
    float phi_y = (phi0 > 0) ? phi_y_minus : phi_y_plus;
    float phi_z = (phi0 > 0) ? phi_z_minus : phi_z_plus;

    float grad_mag = sqrtf(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z);

    float sign_phi0 = phi0 / sqrtf(phi0 * phi0 + 1e-6f);
    phi_new[idx] = phi0 - dt * sign_phi0 * (grad_mag - 1.0f);
}

__global__ void __launch_bounds__(1024)
    fastMarchingKernel(const float *__restrict__ speed, float *__restrict__ phi, int nx, int ny,
                       int nz, float dx, float dy, float dz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz)
        return;
    if (x == 0 || y == 0 || z == 0 || x + 1 >= nx || y + 1 >= ny || z + 1 >= nz)
        return;

    int idx = x + nx * (y + ny * z);
    float s = speed[idx];
    if (s <= 1e-8f)
        return;

    float val = phi[idx];

    const float ax = fminf(phi[idx - 1], phi[idx + 1]);
    const float ay = fminf(phi[idx - nx], phi[idx + nx]);
    const float az = fminf(phi[idx - nx * ny], phi[idx + nx * ny]);

    float mins[3] = {ax, ay, az};
    if (mins[0] > mins[1])
    {
        float tmp = mins[0];
        mins[0] = mins[1];
        mins[1] = tmp;
    }
    if (mins[1] > mins[2])
    {
        float tmp = mins[1];
        mins[1] = mins[2];
        mins[2] = tmp;
    }
    if (mins[0] > mins[1])
    {
        float tmp = mins[0];
        mins[0] = mins[1];
        mins[1] = tmp;
    }

    const float h = fminf(dx, fminf(dy, dz));
    const float rhs = h / s;

    float t = mins[0] + rhs;
    if (t > mins[1])
    {
        const float sum2 = mins[0] + mins[1];
        const float disc2 =
            fmaxf(0.0f, 2.0f * rhs * rhs - (mins[0] - mins[1]) * (mins[0] - mins[1]));
        t = 0.5f * (sum2 + sqrtf(disc2));
    }
    if (t > mins[2])
    {
        const float a = 3.0f;
        const float b = -2.0f * (mins[0] + mins[1] + mins[2]);
        const float c = mins[0] * mins[0] + mins[1] * mins[1] + mins[2] * mins[2] - rhs * rhs;
        const float disc3 = fmaxf(0.0f, b * b - 4.0f * a * c);
        t = (-b + sqrtf(disc3)) / (2.0f * a);
    }

    phi[idx] = fminf(val, t);
}

__global__ void __launch_bounds__(512)
    marchingCubesKernel(const float *__restrict__ phi, float *__restrict__ vertices,
                        int *__restrict__ indices, int *__restrict__ num_triangles, int nx, int ny,
                        int nz, float isovalue, int max_triangles)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1)
        return;

    float cube_vals[8];
    int base = x + nx * (y + ny * z);
    cube_vals[0] = phi[base];
    cube_vals[1] = phi[base + 1];
    cube_vals[2] = phi[base + 1 + nx];
    cube_vals[3] = phi[base + nx];
    cube_vals[4] = phi[base + nx * ny];
    cube_vals[5] = phi[base + 1 + nx * ny];
    cube_vals[6] = phi[base + 1 + nx + nx * ny];
    cube_vals[7] = phi[base + nx + nx * ny];

    int case_index = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (cube_vals[i] < isovalue)
        {
            case_index |= (1 << i);
        }
    }

    if (case_index == 0 || case_index == 255)
        return;

    constexpr int edge_to_vertex[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                           {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    constexpr int vertex_offsets[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                          {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

    float edge_points[12][3];
    int point_count = 0;
    for (int e = 0; e < 12; ++e)
    {
        const int v0 = edge_to_vertex[e][0];
        const int v1 = edge_to_vertex[e][1];
        const bool s0 = cube_vals[v0] < isovalue;
        const bool s1 = cube_vals[v1] < isovalue;
        if (s0 == s1)
        {
            continue;
        }
        const float denom = cube_vals[v1] - cube_vals[v0];
        const float t = fabsf(denom) > 1e-12f ? (isovalue - cube_vals[v0]) / denom : 0.5f;
        edge_points[point_count][0] =
            static_cast<float>(x + vertex_offsets[v0][0]) +
            t * static_cast<float>(vertex_offsets[v1][0] - vertex_offsets[v0][0]);
        edge_points[point_count][1] =
            static_cast<float>(y + vertex_offsets[v0][1]) +
            t * static_cast<float>(vertex_offsets[v1][1] - vertex_offsets[v0][1]);
        edge_points[point_count][2] =
            static_cast<float>(z + vertex_offsets[v0][2]) +
            t * static_cast<float>(vertex_offsets[v1][2] - vertex_offsets[v0][2]);
        ++point_count;
    }

    if (point_count < 3)
    {
        return;
    }

    const int tri_count = point_count - 2;
    const int tri_base = atomicAdd(num_triangles, tri_count);
    if (tri_base >= max_triangles)
    {
        return;
    }
    const int writable_triangles = min(tri_count, max_triangles - tri_base);
    for (int t = 0; t < writable_triangles; ++t)
    {
        const int tri_idx = tri_base + t;
        const int vertex_base = tri_idx * FLOATS_PER_TRIANGLE;
        const int index_base = tri_idx * INDICES_PER_TRIANGLE;

        vertices[vertex_base + 0] = edge_points[0][0];
        vertices[vertex_base + 1] = edge_points[0][1];
        vertices[vertex_base + 2] = edge_points[0][2];
        vertices[vertex_base + 3] = edge_points[t + 1][0];
        vertices[vertex_base + 4] = edge_points[t + 1][1];
        vertices[vertex_base + 5] = edge_points[t + 1][2];
        vertices[vertex_base + 6] = edge_points[t + 2][0];
        vertices[vertex_base + 7] = edge_points[t + 2][1];
        vertices[vertex_base + 8] = edge_points[t + 2][2];

        indices[index_base + 0] = tri_idx * 3;
        indices[index_base + 1] = tri_idx * 3 + 1;
        indices[index_base + 2] = tri_idx * 3 + 2;
    }
}

class GPULevelSetSolver
{
public:
    GPULevelSetSolver(int nx, int ny, int nz)
        : nx_(nx)
        , ny_(ny)
        , nz_(nz)
    {
        const std::size_t size = checkedProduct(checkedElementCount(nx, ny, nz), sizeof(float),
                                                "GPU level set allocation exceeds host limits");
        try
        {
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_phi1_), size), "cudaMalloc d_phi1");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_phi2_), size), "cudaMalloc d_phi2");
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    ~GPULevelSetSolver() { release(); }

    void reinitialize(const std::vector<float> &phi, int num_iterations = 10)
    {
        validatePhi(phi, nx_, ny_, nz_);
        if (num_iterations < 0)
        {
            throw std::invalid_argument("GPU level set iteration count must be nonnegative");
        }
        const std::size_t phi_size =
            checkedProduct(phi.size(), sizeof(float), "GPU level set transfer exceeds host limits");
        checkCuda(cudaMemcpy(d_phi1_, phi.data(), phi_size, cudaMemcpyHostToDevice),
                  "cudaMemcpy phi host-to-device");

        dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
        dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y,
                  (nz_ + block.z - 1) / block.z);

        float dx = 1.0f / nx_;
        float dt = 0.5f * dx;

        for (int iter = 0; iter < num_iterations; ++iter)
        {
            levelSetReinitializeKernel<<<grid, block>>>(d_phi1_, d_phi2_, nx_, ny_, nz_, dx, dx, dx,
                                                        dt);
            checkCuda(cudaGetLastError(), "levelSetReinitializeKernel launch");
            checkCuda(cudaDeviceSynchronize(), "levelSetReinitializeKernel execution");
            std::swap(d_phi1_, d_phi2_);
        }
    }

    std::vector<float> getPhi()
    {
        std::vector<float> result(checkedElementCount(nx_, ny_, nz_));
        const std::size_t phi_size = checkedProduct(result.size(), sizeof(float),
                                                    "GPU level set transfer exceeds host limits");
        checkCuda(cudaMemcpy(result.data(), d_phi1_, phi_size, cudaMemcpyDeviceToHost),
                  "cudaMemcpy phi device-to-host");
        requireFiniteOutput(result, "GPU level set produced non-finite field values");
        return result;
    }

private:
    void release()
    {
        if (d_phi1_)
            cudaFree(d_phi1_);
        if (d_phi2_)
            cudaFree(d_phi2_);
        d_phi1_ = nullptr;
        d_phi2_ = nullptr;
    }

    int nx_, ny_, nz_;
    float *d_phi1_ = nullptr;
    float *d_phi2_ = nullptr;
};

struct Isosurface
{
    std::vector<float> vertices;
    std::vector<int> indices;
};

Isosurface extractIsosurfaceGPU(const std::vector<float> &phi, int nx, int ny, int nz,
                                float isovalue = 0.0f)
{
    Isosurface surface;
    validatePhi(phi, nx, ny, nz);
    if (!std::isfinite(isovalue))
    {
        throw std::invalid_argument("GPU isosurface value must be finite");
    }
    if (nx < 2 || ny < 2 || nz < 2)
    {
        return surface;
    }

    float *d_phi = nullptr;
    float *d_vertices = nullptr;
    int *d_indices = nullptr;
    int *d_num_triangles = nullptr;

    auto release = [&]() {
        if (d_phi)
            cudaFree(d_phi);
        if (d_vertices)
            cudaFree(d_vertices);
        if (d_indices)
            cudaFree(d_indices);
        if (d_num_triangles)
            cudaFree(d_num_triangles);
    };

    const std::size_t phi_size =
        checkedProduct(phi.size(), sizeof(float), "GPU level set allocation exceeds host limits");
    const std::size_t cell_xy =
        checkedProduct(static_cast<std::size_t>(nx - 1), static_cast<std::size_t>(ny - 1),
                       "GPU isosurface cell count exceeds host limits");
    const std::size_t cell_count = checkedProduct(cell_xy, static_cast<std::size_t>(nz - 1),
                                                  "GPU isosurface cell count exceeds host limits");
    const std::size_t max_triangles_size =
        checkedProduct(cell_count, static_cast<std::size_t>(MAX_TRIANGLES_PER_CELL),
                       "GPU isosurface triangle allocation exceeds host limits");
    if (max_triangles_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("GPU isosurface triangle allocation exceeds kernel limits");
    }
    const int max_triangles = static_cast<int>(max_triangles_size);
    const std::size_t vertex_count =
        checkedProduct(max_triangles_size, static_cast<std::size_t>(FLOATS_PER_TRIANGLE),
                       "GPU isosurface vertex allocation exceeds host limits");
    const std::size_t index_count =
        checkedProduct(max_triangles_size, static_cast<std::size_t>(INDICES_PER_TRIANGLE),
                       "GPU isosurface index allocation exceeds host limits");
    const std::size_t vertex_bytes = checkedProduct(
        vertex_count, sizeof(float), "GPU isosurface vertex allocation exceeds host limits");
    const std::size_t index_bytes = checkedProduct(
        index_count, sizeof(int), "GPU isosurface index allocation exceeds host limits");

    try
    {
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_phi), phi_size),
                  "cudaMalloc marching cubes phi");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertex_bytes),
                  "cudaMalloc marching cubes vertices");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_indices), index_bytes),
                  "cudaMalloc marching cubes indices");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_num_triangles), sizeof(int)),
                  "cudaMalloc marching cubes triangle counter");

        checkCuda(cudaMemcpy(d_phi, phi.data(), phi_size, cudaMemcpyHostToDevice),
                  "cudaMemcpy marching cubes phi host-to-device");
        checkCuda(cudaMemset(d_num_triangles, 0, sizeof(int)),
                  "cudaMemset marching cubes triangle counter");

        dim3 block(MARCHING_CUBES_BLOCK_DIM, MARCHING_CUBES_BLOCK_DIM, MARCHING_CUBES_BLOCK_DIM);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                  (nz + block.z - 1) / block.z);

        marchingCubesKernel<<<grid, block>>>(d_phi, d_vertices, d_indices, d_num_triangles, nx, ny,
                                             nz, isovalue, max_triangles);
        checkCuda(cudaGetLastError(), "marchingCubesKernel launch");
        checkCuda(cudaDeviceSynchronize(), "marchingCubesKernel execution");

        int num_tri = 0;
        checkCuda(cudaMemcpy(&num_tri, d_num_triangles, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy marching cubes triangle counter device-to-host");

        if (num_tri < 0)
        {
            throw std::runtime_error("GPU isosurface returned invalid triangle count");
        }
        num_tri = std::min(num_tri, max_triangles);
        if (num_tri > 0)
        {
            surface.vertices.resize(checkedProduct(
                static_cast<std::size_t>(num_tri), static_cast<std::size_t>(FLOATS_PER_TRIANGLE),
                "GPU isosurface vertex transfer exceeds host limits"));
            surface.indices.resize(checkedProduct(
                static_cast<std::size_t>(num_tri), static_cast<std::size_t>(INDICES_PER_TRIANGLE),
                "GPU isosurface index transfer exceeds host limits"));
            const std::size_t surface_vertex_bytes =
                checkedProduct(surface.vertices.size(), sizeof(float),
                               "GPU isosurface vertex transfer exceeds host limits");
            const std::size_t surface_index_bytes =
                checkedProduct(surface.indices.size(), sizeof(int),
                               "GPU isosurface index transfer exceeds host limits");
            checkCuda(cudaMemcpy(surface.vertices.data(), d_vertices, surface_vertex_bytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy marching cubes vertices device-to-host");
            checkCuda(cudaMemcpy(surface.indices.data(), d_indices, surface_index_bytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy marching cubes indices device-to-host");
            requireFiniteOutput(surface.vertices,
                                "GPU isosurface produced non-finite vertex values");
        }

        release();
    }
    catch (...)
    {
        release();
        throw;
    }

    return surface;
}

} // namespace nerve::filtration::levelset::gpu
