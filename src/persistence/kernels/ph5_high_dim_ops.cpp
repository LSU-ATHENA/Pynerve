
#include "nerve/algebra/simplex.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/kernels/ph5_high_dim_ops.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <tuple>

namespace nerve::persistence
{

namespace
{

constexpr double PERSISTENCE_PAIR_TOLERANCE = 1e-12;

void overwriteDiagramFromExact(const ExactPersistenceResult &exact_result,
                               ::nerve::Diagram *diagram)
{
    if (diagram == nullptr)
    {
        return;
    }
    diagram->clear();
    for (const auto &pair : exact_result.pairs)
    {
        if (pair.dimension < 0)
        {
            continue;
        }
        diagram->addPair(pair.birth, pair.death, static_cast<size_t>(pair.dimension));
    }
}

} // namespace

PH5HighDimensional::PH5HighDimensional(const PersistenceBudget &budget)
    : budget_(budget)
    , diagram_(std::make_unique<::nerve::Diagram>())
    , summary_(std::make_unique<::nerve::CompactSummary>())
    , certificate_(StabilityCertificate::createPh5Ph6Certificate(0, 0, budget.memory_limit_mb,
                                                                 budget.time_limit_ms))
    , ordering_strategy_(CohomologyOrdering::STANDARD)
    , optimization_enabled_(true)
    , budget_exceeded_(false)
{}

ResultType PH5HighDimensional::computePersistenceCohomology(const SimplicialComplex &complex,
                                                            size_t max_dimension)
{
    if (!checkBudget())
    {
        handleBudgetExceeded();
        return ResultType::error(::nerve::ErrorCode::E11_PH5_OVERFLOW, "budget exceeded");
    }
    budget_exceeded_ = false;
    return computeCohomologyReduction(complex, max_dimension);
}

ResultType PH5HighDimensional::computeCohomologyReduction(const SimplicialComplex &complex,
                                                          size_t max_dimension)
{
    buildCohomologyComplex(complex, max_dimension);
    applyCleverOrdering();
    reduceBoundaryMatrix();
    extractPersistencePairs();
    certificate_ = StabilityCertificate::createPh5Ph6Certificate(
        complex.size(), max_dimension, budget_.memory_limit_mb, budget_.time_limit_ms);
    return ResultType::success(*diagram_);
}

void PH5HighDimensional::buildCohomologyComplex(const SimplicialComplex &complex,
                                                size_t max_dimension)
{
    if (!diagram_)
    {
        diagram_ = std::make_unique<::nerve::Diagram>();
    }
    const auto exact = computeExactPersistenceZ2(complex, max_dimension);
    overwriteDiagramFromExact(exact, diagram_.get());
}

void PH5HighDimensional::applyCleverOrdering()
{
    if (!diagram_ || ordering_strategy_ == CohomologyOrdering::STANDARD)
    {
        return;
    }
    // Deterministic stabilization pass: sort by dimension then lifetime.
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    std::sort(pairs.begin(), pairs.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.dimension != rhs.dimension)
        {
            return lhs.dimension < rhs.dimension;
        }
        const double lhs_lifetime = lhs.death - lhs.birth;
        const double rhs_lifetime = rhs.death - rhs.birth;
        return lhs_lifetime > rhs_lifetime;
    });
    diagram_->clear();
    for (const auto &pair : pairs)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

void PH5HighDimensional::reduceBoundaryMatrix()
{
    if (optimization_enabled_)
    {
        optimizeCohomologyComputation();
    }
    applyColumnReductions();
    eliminatePersistentPairs();
}

void PH5HighDimensional::extractPersistencePairs()
{
    if (!diagram_)
    {
        diagram_ = std::make_unique<::nerve::Diagram>();
    }
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        return std::tie(a.dimension, a.birth, a.death) < std::tie(b.dimension, b.birth, b.death);
    });
    diagram_->clear();
    for (const auto &pair : pairs)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

bool PH5HighDimensional::checkBudget()
{
    return budget_.memory_limit_mb > 0 && budget_.time_limit_ms > 0;
}

void PH5HighDimensional::handleBudgetExceeded()
{
    budget_exceeded_ = true;
    if (!summary_)
    {
        summary_ = std::make_unique<::nerve::CompactSummary>();
    }
}

void PH5HighDimensional::optimizeCohomologyComputation()
{
    if (!diagram_)
    {
        return;
    }
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        const double a_lifetime = a.death - a.birth;
        const double b_lifetime = b.death - b.birth;
        if (a.dimension != b.dimension)
            return a.dimension < b.dimension;
        if (a_lifetime != b_lifetime)
            return a_lifetime > b_lifetime; // Descending lifetime
        return a.birth < b.birth;
    });
    diagram_->clear();
    for (const auto &pair : pairs)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

void PH5HighDimensional::applyColumnReductions()
{
    if (!diagram_)
    {
        return;
    }
    std::vector<PersistencePairRecord> reduced;
    reduced.reserve(diagram_->size());
    for (const auto &pair : diagram_->pairs())
    {
        if (!std::isfinite(pair.birth))
        {
            continue;
        }
        if (std::isfinite(pair.death) && pair.death + PERSISTENCE_PAIR_TOLERANCE < pair.birth)
        {
            continue;
        }
        reduced.push_back(pair);
    }
    diagram_->clear();
    for (const auto &pair : reduced)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

void PH5HighDimensional::eliminatePersistentPairs()
{
    if (!diagram_)
    {
        return;
    }
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        return std::tie(a.dimension, a.birth, a.death) < std::tie(b.dimension, b.birth, b.death);
    });

    if (pairs.empty())
        return;

    // Keep dimension-wise significant features and all essential classes.
    std::map<size_t, double> max_persistence;
    for (const auto &pair : pairs)
    {
        if (!std::isfinite(pair.death))
        {
            continue;
        }
        max_persistence[pair.dimension] =
            std::max(max_persistence[pair.dimension], std::max(0.0, pair.death - pair.birth));
    }

    std::vector<PersistencePairRecord> filtered;
    filtered.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        if (!std::isfinite(pair.death))
        {
            filtered.push_back(pair);
            continue;
        }
        const double persistence = std::max(0.0, pair.death - pair.birth);
        const double max_dim_persistence = max_persistence[pair.dimension];
        const double cutoff = max_dim_persistence * 0.01;
        if (persistence >= cutoff)
        {
            filtered.push_back(pair);
        }
    }

    diagram_->clear();
    for (const auto &pair : filtered)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

void PH5HighDimensional::setBudget(const PersistenceBudget &budget)
{
    budget_ = budget;
}

const PersistenceBudget &PH5HighDimensional::getBudget() const
{
    return budget_;
}

void PH5HighDimensional::setCohomologyOrderingStrategy(CohomologyOrdering strategy)
{
    ordering_strategy_ = strategy;
}

void PH5HighDimensional::enableOptimization(bool enable)
{
    optimization_enabled_ = enable;
}

const ::nerve::Diagram &PH5HighDimensional::getDiagram() const
{
    return *diagram_;
}

const ::nerve::CompactSummary &PH5HighDimensional::getSummary() const
{
    return *summary_;
}

const StabilityCertificate &PH5HighDimensional::getCertificate() const
{
    return certificate_;
}

bool PH5HighDimensional::hasDiagram() const
{
    return diagram_ != nullptr;
}

bool PH5HighDimensional::hasSummary() const
{
    return summary_ != nullptr;
}

bool PH5HighDimensional::budgetExceeded() const
{
    return budget_exceeded_;
}

} // namespace nerve::persistence
