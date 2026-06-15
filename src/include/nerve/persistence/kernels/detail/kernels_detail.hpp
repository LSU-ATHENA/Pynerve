#pragma once
#include "nerve/core/budget.hpp"
#include "nerve/core/persistence.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <vector>

namespace nerve::persistence
{
bool hasValidPointCloud(const ::nerve::PointCloud &points);
double euclideanDistance(const std::vector<double> &a, const std::vector<double> &b);
void overwriteDiagramFromExact(const ::nerve::persistence::ExactPersistenceResult &exact,
                               ::nerve::Diagram *diagram);

class PH6HighDimensional
{
public:
    explicit PH6HighDimensional(const ::nerve::PersistenceBudget &budget);
};
} // namespace nerve::persistence
