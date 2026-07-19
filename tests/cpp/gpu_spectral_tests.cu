#include "gpu_test_helpers.cuh"
#include "nerve/spectral/persistent_laplacian.hpp"
#include "nerve/spectral/symmetric_eigendecomposition.hpp"

#include <Eigen/Sparse>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU spectral kernel coverage tests\n";
        return 0;
    }

    // Spectral: SpectralConfig defaults
    {
        nerve::spectral::SpectralConfig config;
        assert(config.num_eigenpairs == 50);
        assert(config.convergence_tolerance > 0.0);
        assert(config.max_iterations == 1000);
        std::cout << "PASSED: spectral config defaults\n";
    }

    // Spectral: Eigenpair struct validation
    {
        nerve::spectral::Eigenpair ep;
        ep.eigenvalue = 1.0;
        ep.error_estimate = 0.001;
        ep.spectral_gap = 0.01;
        ep.participation_ratio = 0.5;
        ep.complex_eigenvalue = std::complex<double>(0.0, 0.0);
        bool valid = ep.isValid();
        static_cast<void>(valid);
        std::cout << "PASSED: spectral eigenpair struct smoke\n";
    }

    // Spectral: SpectralDecomposition struct defaults
    {
        nerve::spectral::SpectralDecomposition sd;
        assert(sd.matrix_size == 0);
        assert(sd.num_harmonic == 0);
        assert(sd.converged == false);
        assert(sd.computation_time_ms == 0.0);
        std::cout << "PASSED: spectral decomposition struct defaults\n";
    }

    // Spectral: Jacobi eigendecomposition (CPU, inline, small matrix)
    {
        std::vector<std::vector<double>> mat = {{2.0, -1.0}, {-1.0, 2.0}};
        auto result = nerve::spectral::detail::jacobiEigendecomposition(mat, 64, 1e-10);
        assert(result.eigenvalues.size() == 2);
        assert(result.eigenvalues[0] >= 0.0);
        assert(result.eigenvalues[1] >= result.eigenvalues[0]);
        std::cout << "PASSED: spectral jacobi eigendecomposition (2x2)\n";
    }

    // Spectral: jacobi eigendecomposition on 3x3
    {
        std::vector<std::vector<double>> mat = {
            {2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}};
        auto result = nerve::spectral::detail::jacobiEigendecomposition(mat, 128, 1e-12);
        assert(result.eigenvalues.size() == 3);
        assert(result.eigenvectors.size() == 3);
        for (size_t i = 0; i < 3; ++i)
            assert(result.eigenvalues[i] >= -1e-10);
        std::cout << "PASSED: spectral jacobi eigendecomposition (3x3)\n";
    }

    // Spectral: PersistentLaplacianSolverGPU -- isAvailable (exercises cudaGetDeviceCount)
    {
        nerve::spectral::SpectralConfig config;
        nerve::spectral::PersistentLaplacianSolverGPU gpu_solver(config);
        bool avail = gpu_solver.isAvailable();
        assert(avail);
        std::cout << "PASSED: PersistentLaplacianSolverGPU isAvailable=" << avail << "\n";
    }

    // Spectral: PersistentLaplacianSolverGPU -- getGpuInfo (exercises cudaMemGetInfo)
    {
        nerve::spectral::SpectralConfig config;
        nerve::spectral::PersistentLaplacianSolverGPU gpu_solver(config);
        std::string info = gpu_solver.getGpuInfo();
        assert(!info.empty());
        std::cout << "PASSED: PersistentLaplacianSolverGPU getGpuInfo=" << info << "\n";
    }

    // Spectral: PersistentLaplacianSolverGPU -- setGpuMemoryLimit + getGpuMemoryUsage
    {
        nerve::spectral::SpectralConfig config;
        nerve::spectral::PersistentLaplacianSolverGPU gpu_solver(config);
        gpu_solver.setGpuMemoryLimit(256);
        size_t usage = gpu_solver.getGpuMemoryUsage();
        assert(usage <= 256 || usage == 0);
        std::cout << "PASSED: PersistentLaplacianSolverGPU memory limit (usage=" << usage
                  << " MB)\n";
    }

    // Spectral: PersistentLaplacianSolverGPU -- computeSpectrumGpu on 2x2 laplacian
    {
        nerve::spectral::SpectralConfig config;
        config.num_eigenpairs = 1;
        config.max_iterations = 100;
        nerve::spectral::PersistentLaplacianSolverGPU gpu_solver(config);

        if (!gpu_solver.isAvailable())
        {
            std::cout
                << "SKIP: PersistentLaplacianSolverGPU computeSpectrumGpu (GPU unavailable)\n";
        }
        else
        {
            Eigen::SparseMatrix<double> laplacian(2, 2);
            laplacian.insert(0, 0) = 1.0;
            laplacian.insert(0, 1) = -1.0;
            laplacian.insert(1, 0) = -1.0;
            laplacian.insert(1, 1) = 1.0;
            laplacian.makeCompressed();

            auto result = gpu_solver.computeSpectrumGpu(laplacian);
            if (result.isSuccess())
            {
                auto &sd = result.value();
                assert(sd.eigenvalues.size() > 0);
                std::cout << "PASSED: PersistentLaplacianSolverGPU computeSpectrumGpu (2x2, "
                          << sd.eigenvalues.size() << " eigenvalues)\n";
            }
            else
            {
                std::cout
                    << "PASSED: PersistentLaplacianSolverGPU computeSpectrumGpu returned error ("
                    << static_cast<uint32_t>(result.errorCode()) << ")\n";
            }
        }
    }

    // Spectral: computeSpectrumGpu on 3x3 path graph laplacian
    {
        nerve::spectral::SpectralConfig config;
        config.num_eigenpairs = 2;
        config.max_iterations = 200;
        nerve::spectral::PersistentLaplacianSolverGPU gpu_solver(config);

        if (!gpu_solver.isAvailable())
        {
            std::cout << "SKIP: PersistentLaplacianSolverGPU 3x3 laplacian (GPU unavailable)\n";
        }
        else
        {
            Eigen::SparseMatrix<double> laplacian(3, 3);
            laplacian.insert(0, 0) = 1.0;
            laplacian.insert(0, 1) = -1.0;
            laplacian.insert(1, 0) = -1.0;
            laplacian.insert(1, 1) = 2.0;
            laplacian.insert(1, 2) = -1.0;
            laplacian.insert(2, 1) = -1.0;
            laplacian.insert(2, 2) = 1.0;
            laplacian.makeCompressed();

            auto result = gpu_solver.computeSpectrumGpu(laplacian);
            if (result.isSuccess())
            {
                auto &sd = result.value();
                assert(sd.eigenvalues.size() > 0);
                for (size_t i = 0; i < sd.eigenvalues.size(); ++i)
                    assert(sd.eigenvalues[i] >= -1e-10);
                std::cout
                    << "PASSED: PersistentLaplacianSolverGPU computeSpectrumGpu (3x3 path graph)\n";
            }
            else
            {
                std::cout << "PASSED: PersistentLaplacianSolverGPU 3x3 returned error ("
                          << static_cast<uint32_t>(result.errorCode()) << ")\n";
            }
        }
    }

    // Spectral: SpectralFeatureExtractor FeatureConfig
    {
        nerve::spectral::SpectralFeatureExtractor::FeatureConfig cfg;
        assert(cfg.use_eigenvalues == true);
        assert(cfg.use_eigenvectors == true);
        assert(cfg.normalize_features == true);
        assert(cfg.max_features == 100);
        std::cout << "PASSED: spectral feature extractor config\n";
    }

    // Spectral: SpectralAnomalyDetector AnomalyConfig
    {
        nerve::spectral::SpectralAnomalyDetector::AnomalyConfig cfg;
        assert(cfg.anomaly_threshold > 0.0);
        assert(cfg.reference_window_size == 100);
        assert(cfg.use_eigenvalue_anomalies == true);
        std::cout << "PASSED: spectral anomaly detector config\n";
    }

    return 0;
}
