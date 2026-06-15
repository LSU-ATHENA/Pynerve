#pragma once

#include "nerve/gpu/consumer_config.hpp"

namespace nerve::gpu::consumer
{

template <typename T = double>
using ConsumerGPUConfig = ConsumerConfig;

template <typename T = double>
[[nodiscard]] inline ConsumerGPUConfig<T> autoConfigure()
{
    auto config = detectGPU();
    if (config.isOptimized())
    {
        (void)autoTune(config);
    }
    return config;
}

} // namespace nerve::gpu::consumer
