// Internal helpers for incremental persistence recomputation.
// Kept in a private header to split implementation while preserving a small
// public translation unit footprint.

#pragma once

#include "nerve/core/persistence.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

namespace nerve::persistence
{

::nerve::ErrorResult<algebra::SimplicialComplex> buildIncrementalComplex(const PointCloud &points,
                                                                         size_t max_dimension);

void overwriteDiagram(const ExactPersistenceResult &exact, ::nerve::Diagram *diagram);

} // namespace nerve::persistence
