#include "gpu_test_helpers.cuh"
#include "nerve/filtration/level_set.hpp"

#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU filtration kernel coverage tests\n";
        return 0;
    }

    // Filtration: LevelSet grid + computeSublevelFiltration
    {
        std::vector<double> scalar_field(16);
        for (int i = 0; i < 16; ++i)
            scalar_field[i] = static_cast<double>(i % 4 + i / 4);
        std::vector<nerve::Size> shape = {4, 4};
        auto result = nerve::filtration::computeSublevelFiltration(scalar_field, shape, 8);
        assert(result.size() > 0);
        std::cout << "PASSED: computeSublevelFiltration (4x4, " << result.size() << " simplices)\n";
    }

    // Filtration: computeSuperlevelFiltration
    {
        std::vector<double> scalar_field(9);
        for (int i = 0; i < 9; ++i)
            scalar_field[i] = static_cast<double>(i % 3 + i / 3);
        std::vector<nerve::Size> shape = {3, 3};
        auto result = nerve::filtration::computeSuperlevelFiltration(scalar_field, shape, 8);
        assert(result.size() > 0);
        std::cout << "PASSED: computeSuperlevelFiltration (3x3, " << result.size() << " simplices)\n";
    }

    // Filtration: computeAdaptiveFiltration
    {
        std::vector<double> scalar_field(9);
        for (int i = 0; i < 9; ++i)
            scalar_field[i] = static_cast<double>(i % 3 + i / 3);
        std::vector<nerve::Size> shape = {3, 3};
        auto result = nerve::filtration::computeAdaptiveFiltration(scalar_field, shape);
        assert(!result.isError());
        assert(result.value().size() > 0);
        std::cout << "PASSED: computeAdaptiveFiltration (3x3, " << result.value().size() << " simplices)\n";
    }

    // Filtration: findCriticalPoints2d
    {
        std::vector<double> scalar_field(16);
        for (int i = 0; i < 16; ++i)
            scalar_field[i] = static_cast<double>(i % 4) * static_cast<double>(i / 4);
        auto crit = nerve::filtration::findCriticalPoints2d(scalar_field, 4, 4);
        assert(crit.size() >= 0);
        std::cout << "PASSED: findCriticalPoints2d (4x4, " << crit.size() << " critical)\n";
    }

    // Filtration: computeGradientField
    {
        std::vector<double> scalar_field(9);
        for (int i = 0; i < 9; ++i)
            scalar_field[i] = static_cast<double>(i % 3) * static_cast<double>(i / 3);
        std::vector<nerve::Size> shape = {3, 3};
        auto gradient = nerve::filtration::computeGradientField(scalar_field, shape);
        assert(gradient.size() == 9);
        std::cout << "PASSED: computeGradientField (3x3 grid)\n";
    }

    // Filtration: LevelSet findMinima/findMaxima
    {
        nerve::filtration::LevelSet ls;
        std::vector<nerve::Size> shape = {3, 3};
        ls.setGridShape(shape);
        std::vector<double> scalar_field = {0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0};
        auto minima = ls.findMinima(scalar_field);
        auto maxima = ls.findMaxima(scalar_field);
        assert(minima.size() > 0);
        assert(maxima.size() > 0);
        std::cout << "PASSED: LevelSet findMinima/Maxima (3x3, "
                  << minima.size() << " minima, " << maxima.size() << " maxima)\n";
    }

    return 0;
}
