
#pragma once

#include "nerve/errors/detail/base.hpp"
#include "nerve/errors/detail/macros.hpp"
#include "nerve/errors/types/runtime.hpp"
#include "nerve/errors/types/type.hpp"
#include "nerve/errors/types/value.hpp"

// Re-export error types into nerve namespace for convenience.
namespace nerve
{

using errors::AllocationError;
using errors::BettiError;
using errors::BudgetExceededError;
using errors::ConvergenceError;
using errors::DeterminismError;
using errors::DimensionError;
using errors::GPUError;
using errors::GPULaunchError;
using errors::GPUMemoryError;
using errors::InvalidArgumentError;
using errors::InvalidSimplexError;
using errors::IOError;
using errors::MatrixStructureError;
using errors::MemoryError;
using errors::NerveError;
using errors::NUMAError;
using errors::NumericalError;
using errors::NumericalInstabilityError;
using errors::OutOfMemoryError;
using errors::PersistenceError;
using errors::PrecisionError;
using errors::ShapeMismatchError;
using errors::TypeError;

using errors::formatBytes;
using errors::formatDurationMs;
using errors::formatShape;

} // namespace nerve
