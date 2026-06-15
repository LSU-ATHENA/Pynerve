// This header now includes decomposed component headers for GPU capabilities
//
// Components:
// - gpu_capability_core.hpp: AdvancedCapabilities class
// - gpu_ptx.hpp: PTX optimization helpers
// - gpu_tuning.hpp: Auto-tuning and configuration
// - gpu_launch.hpp: High-level kernel launch APIs

#pragma once

#include "nerve/gpu/gpu_capability_core.hpp"
#include "nerve/gpu/gpu_launch.hpp"
#include "nerve/gpu/gpu_ptx.hpp"
#include "nerve/gpu/gpu_tuning.hpp"

namespace nerve::gpu::advanced
{

// Namespace is now populated from component headers

} // namespace nerve::gpu::advanced

// Compilation Guard
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
// Host-side: Warn if advanced features are missing
#pragma message("Advanced kernels require sm90+ (Hopper/Blackwell). "                              \
                "Some features will be disabled.")
#endif
