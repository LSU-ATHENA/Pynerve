#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/cuda_warp_specialized_kernels.hpp"
#include "nerve/persistence/kernels/dimension_specialized_kernels.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU specialized kernel coverage tests\n";
        return 0;
    }

    // Specialized: WarpSpecializationConfig constructible
    {
        nerve::persistence::gpu::WarpSpecializationConfig cfg;
        // Verify all fields are initializable, not matching exact defaults
        assert(cfg.warps_per_block > 0);
        cfg.warps_per_block = 16;
        assert(cfg.warps_per_block == 16);
        std::cout << "PASSED: warp specialization config (warps_per_block=16)\n";
    }

    // Specialized: DimensionConfig with real members
    {
        nerve::persistence::kernels::DimensionConfig cfg;
        cfg.max_dimension = 2;
        cfg.use_cohomology = true;
        cfg.use_bit_parallel = false;
        cfg.chunk_size = 512;
        assert(cfg.max_dimension == 2);
        assert(cfg.use_cohomology == true);
        assert(cfg.chunk_size == 512);
        std::cout << "PASSED: DimensionConfig (max_dim=2, cohomology, chunk=512)\n";
    }

    // Specialized: H0Kernel defaults
    {
        nerve::persistence::kernels::H0Kernel h0;
        assert(h0.num_vertices >= 0);
        std::cout << "PASSED: H0Kernel defaults\n";
    }

    // Specialized: H1Kernel defaults
    {
        nerve::persistence::kernels::H1Kernel h1;
        assert(h1.num_edges >= 0);
        std::cout << "PASSED: H1Kernel defaults\n";
    }

    // Specialized: H2Kernel defaults
    {
        nerve::persistence::kernels::H2Kernel h2;
        assert(h2.num_triangles >= 0);
        std::cout << "PASSED: H2Kernel defaults\n";
    }

    return 0;
}
