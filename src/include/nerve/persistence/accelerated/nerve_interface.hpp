
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/accelerated/nerve_data_wrapper.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

class NerveVRInterface
{
public:
    static errors::ErrorResult<std::unique_ptr<NerveVRInterface>>
    create(const VRConfig &config = {})
    {
        try
        {
            auto interface = std::unique_ptr<NerveVRInterface>(new NerveVRInterface(config));
            return errors::ErrorResult<std::unique_ptr<NerveVRInterface>>::success(
                std::move(interface));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<NerveVRInterface>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(const core::BufferView<const double> &points, size_t point_dim,
                         const core::DeterminismContract &contract = {})
    {
        if (points.size() == 0 || point_dim == 0 || points.size() % point_dim != 0)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                                 "invalid point buffer");
        }
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!std::isfinite(points[i]))
            {
                return errors::ErrorResult<std::vector<Pair>>::error(
                    errors::ErrorCode::E50_PH_ABORT, "Point coordinates must be finite");
            }
        }

        // Validate input according to determinism level
        if (contract.level == core::DeterminismLevel::STRICT)
        {
            const bool validation = validateDeterministicInput(points, point_dim, contract);
            if (!validation)
            {
                return errors::ErrorResult<std::vector<Pair>>::error(
                    errors::ErrorCode::E30_DET_MISMATCH);
            }
        }

        // Call existing implementation
        return computeVrPersistenceNerve(points, point_dim, contract);
    }

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(const std::vector<double> &points, size_t point_dim,
                         const core::DeterminismContract &contract = {})
    {
        auto buffer_view = NerveDataWrapper::createBufferView(points);
        return computeVrPersistence(buffer_view, point_dim, contract);
    }

    bool validateDeterministicInput(const core::BufferView<const double> &points, size_t point_dim,
                                    const core::DeterminismContract &contract)
    {
        if (points.size() == 0 || point_dim == 0)
        {
            return false;
        }

        if (points.size() % point_dim != 0)
        {
            return false;
        }

        if (contract.level == core::DeterminismLevel::STRICT)
        {
            for (size_t i = 0; i < points.size(); ++i)
            {
                if (!std::isfinite(points[i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

private:
    explicit NerveVRInterface(const VRConfig &config)
        : config_(config)
    {}

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistenceNerve(const core::BufferView<const double> &points, size_t point_dim,
                              const core::DeterminismContract &contract);

    VRConfig config_;
};

} // namespace nerve::persistence::accelerated
