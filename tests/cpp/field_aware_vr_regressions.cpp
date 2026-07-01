#include "persistence/field_aware_vietoris_rips.hpp"

#include <cassert>
#include <exception>
#include <limits>
#include <vector>

int main()
{
    try
    {
        std::vector<double> points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        };
        nerve::core::BufferView<const double> view(points);

        nerve::common::VRConfig config;
        config.max_radius = 2.0;
        config.max_dim = 2;

        nerve::persistence::FieldAwareVietorisRips vr(2);
        assert(vr.getNumThreads() >= 1);
        const auto field_result = vr.setFieldCharacteristic(3);
        assert(field_result.isOk());

        const auto result = vr.computeVrPersistenceFast(view, 2, config);
        assert(result.isOk());

        const auto invalid_dimension = vr.computeVrPersistenceFast(view, 0, config);
        assert(invalid_dimension.isErr());

        std::vector<double> ragged_points{0.0, 0.0, 1.0};
        nerve::core::BufferView<const double> ragged_view(ragged_points);
        const auto ragged = vr.computeVrPersistenceFast(ragged_view, 2, config);
        assert(ragged.isErr());

        std::vector<double> infinite_points{0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0};
        nerve::core::BufferView<const double> infinite_view(infinite_points);
        const auto infinite_coordinates = vr.computeVrPersistenceFast(infinite_view, 2, config);
        assert(infinite_coordinates.isErr());

        std::vector<double> overflowing_points{0.0, 0.0, std::numeric_limits<double>::max(), 0.0};
        nerve::core::BufferView<const double> overflowing_view(overflowing_points);
        const auto overflowing_coordinates =
            vr.computeVrPersistenceFast(overflowing_view, 2, config);
        assert(overflowing_coordinates.isErr());

        nerve::common::VRConfig invalid_config = config;
        invalid_config.max_radius = std::numeric_limits<double>::quiet_NaN();
        const auto invalid_radius = vr.computeVrPersistenceFast(view, 2, invalid_config);
        assert(invalid_radius.isErr());

        invalid_config.max_radius = -1.0;
        const auto negative_radius = vr.computeVrPersistenceFast(view, 2, invalid_config);
        assert(negative_radius.isErr());

        invalid_config = config;
        invalid_config.max_radius = std::numeric_limits<double>::max();
        const auto overflowing_radius = vr.computeVrPersistenceFast(view, 2, invalid_config);
        assert(overflowing_radius.isErr());

        std::vector<double> wrapped_distance_points{
            0.0,
            0.0,
            3.0,
            0.0,
        };
        nerve::core::BufferView<const double> wrapped_view(wrapped_distance_points);
        nerve::common::VRConfig wrapped_config;
        wrapped_config.max_radius = 3.0;
        wrapped_config.max_dim = 1;
        const auto wrapped_result = vr.computeVrPersistenceFast(wrapped_view, 2, wrapped_config);
        assert(wrapped_result.isOk());
        assert(!wrapped_result.value().empty());

        return 0;
    }
    catch (const std::exception &)
    {
        return 1;
    }
}
