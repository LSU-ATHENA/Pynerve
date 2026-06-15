#include "nerve/autodiff/autodiff.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

int main()
{
    using nerve::autodiff::Shape;
    using nerve::autodiff::Tensor;

    Tensor valid(std::vector<double>(24, 0.0), Shape{2, 3, 4});
    assert(valid.size() == 24);

    Tensor empty_with_zero_dim(std::vector<double>{}, Shape{2, 0, 4});
    assert(empty_with_zero_dim.size() == 0);

    bool rejected_constructor_overflow = false;
    try
    {
        Tensor tensor(std::vector<double>{},
                      Shape{std::numeric_limits<nerve::Size>::max() / 2 + 1, 3});
        (void)tensor;
    }
    catch (const std::length_error &)
    {
        rejected_constructor_overflow = true;
    }
    assert(rejected_constructor_overflow);

    bool rejected_zeros_overflow = false;
    try
    {
        (void)Tensor::zeros(Shape{std::numeric_limits<nerve::Size>::max() / 2 + 1, 3});
    }
    catch (const std::length_error &)
    {
        rejected_zeros_overflow = true;
    }
    assert(rejected_zeros_overflow);

    return 0;
}
