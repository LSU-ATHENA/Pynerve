
#include "nerve/config.hpp"
#include "nerve/persistence/differentiable/differentiable_persistence.hpp"

#if HAS_PYTORCH

namespace nerve::persistence
{

namespace
{
[[maybe_unused]] constexpr bool kDifferentiableBackendEnabled = true;
}

} // namespace nerve::persistence

#else // !HAS_PYTORCH

namespace nerve::persistence
{

namespace
{
[[maybe_unused]] constexpr bool kDifferentiableBackendEnabled = false;
}

} // namespace nerve::persistence

#endif // HAS_PYTORCH
