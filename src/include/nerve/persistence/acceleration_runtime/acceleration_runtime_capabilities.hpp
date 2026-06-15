#pragma once

#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"

namespace nerve::persistence::acceleration_runtime
{

using GPUInfo = adaptive_acceleration::GPUInfo;
using CPUInfo = adaptive_acceleration::CPUInfo;
using NUMAConfig = adaptive_acceleration::NUMAConfig;
using SystemCapabilities = adaptive_acceleration::SystemCapabilities;
using ThreadAffinityManager = adaptive_acceleration::ThreadAffinityManager;

} // namespace nerve::persistence::acceleration_runtime
