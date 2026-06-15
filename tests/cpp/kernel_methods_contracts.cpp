#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace nerve::algorithms
{
extern "C" void nerve_gaussian_kernel_matrix_f64(const double **diagrams, const size_t *sizes,
                                                 size_t n_diagrams, double sigma, double *output);
}

namespace
{

template <typename Fn>
bool throws_invalid_argument(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

template <typename Fn>
bool does_not_throw(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool check_kernel_c_api_boundary_contract()
{
    const size_t huge = std::numeric_limits<size_t>::max();
    const double diagram[2] = {0.0, 1.0};
    const double *diagrams[1] = {diagram};
    const double *empty_diagrams[1] = {nullptr};
    const double *null_diagrams[1] = {nullptr};
    size_t sizes[1] = {1};
    size_t empty_sizes[1] = {0};
    double output[1] = {};

    if (!throws_invalid_argument([&]() {
            nerve::algorithms::nerve_gaussian_kernel_matrix_f64(nullptr, sizes, 1, 1.0, output);
        }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() {
            nerve::algorithms::nerve_gaussian_kernel_matrix_f64(diagrams, nullptr, 1, 1.0, output);
        }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() {
            nerve::algorithms::nerve_gaussian_kernel_matrix_f64(diagrams, sizes, 1, 1.0, nullptr);
        }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() {
            nerve::algorithms::nerve_gaussian_kernel_matrix_f64(null_diagrams, sizes, 1, 1.0,
                                                                output);
        }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() {
            nerve::algorithms::nerve_gaussian_kernel_matrix_f64(nullptr, nullptr, huge, 1.0,
                                                                nullptr);
        }))
    {
        return false;
    }
    return does_not_throw([&]() {
        nerve::algorithms::nerve_gaussian_kernel_matrix_f64(nullptr, nullptr, 0, 1.0, nullptr);
        nerve::algorithms::nerve_gaussian_kernel_matrix_f64(empty_diagrams, empty_sizes, 1, 1.0,
                                                            output);
    });
}

} // namespace

int main()
{
    if (!check_kernel_c_api_boundary_contract())
    {
        std::cerr << "kernel C API boundary contract failed\n";
        return 1;
    }
    return 0;
}
