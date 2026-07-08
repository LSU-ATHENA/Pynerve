#include "nerve/nn/diagram_conv.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

template <typename Exception, typename Func>
void assertThrows(Func &&func)
{
    bool thrown = false;
    try
    {
        func();
    }
    catch (const Exception &)
    {
        thrown = true;
    }
    assert(thrown);
}

template <typename T>
std::span<const T> asSpan(const std::vector<T> &values)
{
    return std::span<const T>(values.data(), values.size());
}

template <typename T>
bool allFinite(const std::vector<T> &values)
{
    return std::all_of(values.begin(), values.end(), [](T value) { return std::isfinite(value); });
}

} // namespace

int main()
{
    {
        nerve::nn::DiagramConv1D<double>::Config config;
        config.in_channels = 2;
        config.out_channels = 3;
        config.kernel_size = 3;
        nerve::nn::DiagramConv1D<double> conv(config);

        const std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 1.0};
        const std::vector<double> features{1.0, 0.5, 0.2, 0.4};
        const auto output = conv.forward(asSpan(diagram), asSpan(features), 1, 2);
        assert(output.size() == 6);
        assert(allFinite(output));

        auto overflow_config = config;
        overflow_config.in_channels = std::numeric_limits<int>::max();
        overflow_config.out_channels = std::numeric_limits<int>::max();
        overflow_config.kernel_size = std::numeric_limits<int>::max();
        assertThrows<std::length_error>(
            [&] { nerve::nn::DiagramConv1D<double> invalid(overflow_config); });

        const std::vector<double> empty;
        const size_t overflowing_batch = std::numeric_limits<size_t>::max() / 3 + 1;
        assertThrows<std::length_error>([&] {
            const auto result = conv.forward(asSpan(empty), asSpan(empty), overflowing_batch, 3);
            (void)result;
        });

        std::vector<double> bad_diagram = diagram;
        bad_diagram[1] = std::numeric_limits<double>::infinity();
        assertThrows<std::invalid_argument>([&] {
            const auto result = conv.forward(asSpan(bad_diagram), asSpan(features), 1, 2);
            (void)result;
        });

        std::vector<double> bad_features = features;
        bad_features[0] = std::numeric_limits<double>::quiet_NaN();
        assertThrows<std::invalid_argument>([&] {
            const auto result = conv.forward(asSpan(diagram), asSpan(bad_features), 1, 2);
            (void)result;
        });

        nerve::nn::DiagramConv1D<double>::Config small_config;
        small_config.kernel_size = 1;
        nerve::nn::DiagramConv1D<double> small(small_config);
        const std::vector<double> bad_kernel{1.0, std::numeric_limits<double>::infinity()};
        const std::vector<double> bias{0.0};
        assertThrows<std::invalid_argument>(
            [&] { small.set_weights(asSpan(bad_kernel), asSpan(bias)); });
    }

    {
        nerve::nn::DiagramConv2D<double>::Config config;
        config.in_channels = 1;
        config.out_channels = 1;
        config.kernel_h = 2;
        config.kernel_w = 2;
        nerve::nn::DiagramConv2D<double> conv(config);

        const std::vector<double> input{
            1.0, 0.0, 0.5, 0.2, 0.4, 0.6, 0.3, 0.7, 0.9,
        };
        const auto output = conv.forward(asSpan(input), 1, 3, 3);
        assert(output.size() == 4);
        assert(allFinite(output));

        auto overflow_config = config;
        overflow_config.in_channels = std::numeric_limits<int>::max();
        overflow_config.out_channels = std::numeric_limits<int>::max();
        overflow_config.kernel_h = std::numeric_limits<int>::max();
        overflow_config.kernel_w = std::numeric_limits<int>::max();
        assertThrows<std::length_error>(
            [&] { nerve::nn::DiagramConv2D<double> invalid(overflow_config); });

        const std::vector<double> empty;
        const size_t overflowing_batch = std::numeric_limits<size_t>::max() / 3 + 1;
        assertThrows<std::length_error>([&] {
            const auto result = conv.forward(asSpan(empty), overflowing_batch, 3, 3);
            (void)result;
        });

        std::vector<double> bad_input = input;
        bad_input[3] = std::numeric_limits<double>::quiet_NaN();
        assertThrows<std::invalid_argument>([&] {
            const auto result = conv.forward(asSpan(bad_input), 1, 3, 3);
            (void)result;
        });
    }

    {
        nerve::nn::PersistenceImageLayer<double>::Config config;
        config.resolution_h = 4;
        config.resolution_w = 4;
        config.sigma = 0.1;
        nerve::nn::PersistenceImageLayer<double> layer(config);

        const std::vector<double> diagram{0.0, 1.0, 0.0, 0.2, 0.7, 1.0};
        const auto image = layer.forward(asSpan(diagram), 1, 2);
        assert(image.size() == 16);
        assert(allFinite(image));

        auto bad_config = config;
        bad_config.sigma = std::numeric_limits<double>::quiet_NaN();
        assertThrows<std::invalid_argument>(
            [&] { nerve::nn::PersistenceImageLayer<double> invalid(bad_config); });

        auto huge_config = config;
        huge_config.resolution_h = std::numeric_limits<int>::max();
        huge_config.resolution_w = std::numeric_limits<int>::max();
        nerve::nn::PersistenceImageLayer<double> huge_layer(huge_config);
        const std::vector<double> empty;
        const size_t overflowing_batch = std::numeric_limits<size_t>::max() /
                                             static_cast<size_t>(std::numeric_limits<int>::max()) +
                                         1;
        assertThrows<std::length_error>([&] {
            const auto result = huge_layer.forward(asSpan(empty), overflowing_batch, 0);
            (void)result;
        });
    }

    return 0;
}
