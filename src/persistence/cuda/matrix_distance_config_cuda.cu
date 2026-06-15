#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"

#include <algorithm>
#include <limits>

namespace nerve::persistence::accelerated::cuda_host
{

CUDADistanceMatrixConfig getOptimalConfig(Size n_points, Size point_dim,
                                          const CUDADistanceMatrixConfig &base_config)
{
    CUDADistanceMatrixConfig config = base_config;
    if (n_points == 0 || point_dim == 0)
    {
        return config;
    }

    if (n_points < 2048)
    {
        config.max_block_size = std::min<Size>(config.max_block_size, 128);
    }
    else if (n_points < 16384)
    {
        config.max_block_size = std::min<Size>(config.max_block_size, 256);
    }
    else
    {
        config.max_block_size = std::min<Size>(config.max_block_size, 512);
    }
    config.max_block_size = std::max<Size>(32, config.max_block_size);

    if (n_points > std::numeric_limits<Size>::max() / n_points)
    {
        return config;
    }
    const Size total_elements = n_points * n_points;
    const Size block = std::max<Size>(1, config.max_block_size);
    const Size grid = (total_elements / block) + ((total_elements % block) == 0 ? 0 : 1);
    config.max_grid_size = std::max<Size>(1, grid);
    return config;
}

errors::ErrorResult<void> validateLaunchParams(Size n_points, Size point_dim,
                                               const CUDADistanceMatrixConfig &config)
{
    auto cfg_valid = config.validate();
    if (cfg_valid.isError())
    {
        return cfg_valid;
    }
    if (n_points == 0 || point_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Launch dimensions must be positive");
    }
    if (n_points > std::numeric_limits<Size>::max() / n_points)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Distance matrix dimension overflow");
    }
    if (n_points > std::numeric_limits<Size>::max() / point_dim)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Point coordinate buffer size overflow");
    }
    return errors::ErrorResult<void>::ok();
}

} // namespace nerve::persistence::accelerated::cuda_host
