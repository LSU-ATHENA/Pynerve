#include "gpu_test_helpers.cuh"
#include "nerve/sheaf/gpu_sheaf.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU sheaf kernel coverage tests\n";
        return 0;
    }

    // Sheaf: detectSheafHardware (real GPU detection)
    {
        auto hw = nerve::sheaf::detectSheafHardware();
        assert(hw.has_gpu);
        assert(hw.num_cores > 0);
        std::cout << "PASSED: detectSheafHardware (gpu=" << hw.has_gpu
                  << " cores=" << hw.num_cores << " avx512=" << hw.has_avx512 << ")\n";
    }

    // Sheaf: benchmarkSheafGPU (GPU code path)
    {
        auto bench = nerve::sheaf::gpu::benchmarkSheafGPU(32, 8);
        assert(bench.num_stalks == 32);
        assert(bench.stalk_dim == 8);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkSheafGPU (32 stalks, dim=8)\n";
    }

    // Sheaf: benchmarkParallelSheaf
    {
        auto bench = nerve::sheaf::parallel::benchmarkParallelSheaf(16, 4, 2);
        assert(bench.num_stalks == 16);
        assert(bench.stalk_dim == 4);
        assert(bench.sequential_time_ms >= 0.0);
        assert(bench.parallel_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkParallelSheaf (16 stalks, dim=4, threads=2)\n";
    }

    // Sheaf: StalkData
    {
        nerve::sheaf::parallel::StalkData sd(1, 4);
        assert(sd.id == 1);
        assert(sd.dimension == 4);
        assert(sd.data.size() == 4);
        std::cout << "PASSED: StalkData constructor\n";
    }

    // Sheaf: SIMDStalkOperations
    {
        nerve::sheaf::parallel::StalkData a(0, 4);
        nerve::sheaf::parallel::StalkData b(1, 4);
        nerve::sheaf::parallel::StalkData result(2, 4);
        a.data = {1.0f, 0.0f, 0.0f, 0.0f};
        b.data = {0.0f, 1.0f, 0.0f, 0.0f};
        nerve::sheaf::parallel::SIMDStalkOperations::addStalks(a, b, result);
        assert(result.data[0] == 1.0f);
        assert(result.data[1] == 1.0f);
        std::cout << "PASSED: SIMDStalkOperations addStalks\n";
    }

    // Sheaf: SheafConfig + SheafResult defaults
    {
        nerve::sheaf::SheafConfig cfg;
        assert(cfg.num_stalks == 0);
        assert(cfg.use_parallel == true);
        assert(cfg.use_simd == true);
        assert(cfg.gpu_batch_size == 256);
        nerve::sheaf::SheafResult result;
        assert(result.success == false);
        std::cout << "PASSED: SheafConfig + SheafResult defaults\n";
    }

    return 0;
}
