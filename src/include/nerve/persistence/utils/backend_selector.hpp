#pragma once

namespace nerve::persistence
{

// Canonical selector for modern persistence backends exposed to runtime policy.
enum class BackendSelection
{
    AUTO,
    FLOOD_COMPLEX,
    TENSOR_CORE,
    SKETCHING,
    ACCELERATED,
    EXACT_STANDARD,
    ANN_EDGE_DETECTION,
    BLACKWELL_GPU,
    MULTI_GPU
};

} // namespace nerve::persistence
