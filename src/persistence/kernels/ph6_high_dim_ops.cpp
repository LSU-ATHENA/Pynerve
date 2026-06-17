
#include "nerve/algebra/simplex.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/kernels/ph5_high_dim_ops.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::persistence
{

double euclideanDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    const size_t dims = lhs.size();
    double sum = 0.0;
    for (size_t d = 0; d < dims; ++d)
    {
        if (!std::isfinite(lhs[d]) || !std::isfinite(rhs[d]))
        {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = lhs[d] - rhs[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

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

bool hasValidPointCloud(const PointCloud &points)
{
    if (points.empty())
    {
        return true;
    }
    const size_t point_dim = points.front().size();
    if (point_dim == 0)
    {
        return false;
    }
    for (const auto &point : points)
    {
        if (point.size() != point_dim)
        {
            return false;
        }
        for (double value : point)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
    }
    return true;
}

PH6HighDimensional::PH6HighDimensional(const PersistenceBudget &budget)
    : budget_(budget)
    , diagram_(std::make_unique<::nerve::Diagram>())
    , summary_(std::make_unique<::nerve::CompactSummary>())
    , certificate_(StabilityCertificate::createPh5Ph6Certificate(0, 0, budget.memory_limit_mb,
                                                                 budget.time_limit_ms))
    , sampling_strategy_(WitnessSampling::HIERARCHICAL)
    , adaptive_truncation_enabled_(true)
    , landmark_ratio_(0.1)
    , budget_exceeded_(false)
{}

ResultType PH6HighDimensional::computePersistenceWitness(const PointCloud &points,
                                                         size_t max_dimension,
                                                         double landmark_ratio)
{
    if (!std::isfinite(landmark_ratio) || !hasValidPointCloud(points))
    {
        return ResultType::error(::nerve::ErrorCode::E54_PH4_INVALID_INPUT,
                                 "invalid witness point cloud");
    }
    landmark_ratio_ = std::clamp(landmark_ratio, 0.01, 1.0);
    if (!checkBudget())
    {
        handleBudgetExceeded();
        return ResultType::error(::nerve::ErrorCode::E12_PH6_OVERFLOW, "budget exceeded");
    }
    budget_exceeded_ = false;
    return computeHierarchicalWitness(points, max_dimension);
}

ResultType PH6HighDimensional::computeHierarchicalWitness(const PointCloud &points,
                                                          size_t max_dimension)
{
    const size_t requested_landmarks = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(static_cast<double>(points.size()) * landmark_ratio_)));
    const std::vector<size_t> landmarks = selectLandmarks(points, requested_landmarks);
    buildWitnessComplex(points, landmarks, max_dimension);
    if (adaptive_truncation_enabled_)
    {
        applyAdaptiveTruncation();
    }
    computePartialDiagram();
    certificate_ = StabilityCertificate::createPh5Ph6Certificate(
        points.size(), max_dimension, budget_.memory_limit_mb, budget_.time_limit_ms);
    return ResultType::success(*diagram_);
}

std::vector<size_t> PH6HighDimensional::selectLandmarks(const PointCloud &points,
                                                        size_t num_landmarks)
{
    std::vector<size_t> landmarks;
    if (points.empty())
    {
        return landmarks;
    }
    const size_t clamped_count = std::min(points.size(), num_landmarks);
    landmarks.reserve(clamped_count);
    if (sampling_strategy_ == WitnessSampling::RANDOM)
    {
        std::vector<size_t> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 generator(42);
        std::shuffle(indices.begin(), indices.end(), generator);
        indices.resize(clamped_count);
        std::sort(indices.begin(), indices.end());
        return indices;
    }

    landmarks.push_back(0);
    std::vector<double> minDist(points.size(), std::numeric_limits<double>::infinity());
    while (landmarks.size() < clamped_count)
    {
        const size_t anchor = landmarks.back();
        for (size_t i = 0; i < points.size(); ++i)
        {
            minDist[i] = std::min(minDist[i], euclideanDistance(points[i], points[anchor]));
        }
        size_t candidate = 0;
        double farthest = -1.0;
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (std::ranges::find(landmarks, i) != landmarks.end())
            {
                continue;
            }
            if (minDist[i] > farthest)
            {
                farthest = minDist[i];
                candidate = i;
            }
        }
        landmarks.push_back(candidate);
    }
    return landmarks;
}

void PH6HighDimensional::buildWitnessComplex(const PointCloud &points,
                                             const std::vector<size_t> &landmarks,
                                             size_t max_dimension)
{
    if (!diagram_)
    {
        diagram_ = std::make_unique<::nerve::Diagram>();
    }
    diagram_->clear();
    if (landmarks.empty() || points.empty())
    {
        return;
    }

    algebra::SimplicialComplex witness_complex;
    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        witness_complex.addSimplexWithFiltration(algebra::Simplex({static_cast<Index>(i)}), 0.0);
    }

    std::vector<std::vector<double>> distance(landmarks.size(),
                                              std::vector<double>(landmarks.size(), 0.0));
    std::vector<double> edge_filtrations;
    edge_filtrations.reserve((landmarks.size() * (landmarks.size() - 1)) / 2);
    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        for (size_t j = i + 1; j < landmarks.size(); ++j)
        {
            const double dist = euclideanDistance(points[landmarks[i]], points[landmarks[j]]);
            distance[i][j] = dist;
            distance[j][i] = dist;
            edge_filtrations.push_back(dist);
        }
    }
    std::sort(edge_filtrations.begin(), edge_filtrations.end());
    const double threshold =
        edge_filtrations.empty() ? 0.0 : edge_filtrations[edge_filtrations.size() / 2] * 1.5;

    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        for (size_t j = i + 1; j < landmarks.size(); ++j)
        {
            if (distance[i][j] <= threshold)
            {
                witness_complex.addSimplexWithFiltration(
                    algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j)}),
                    distance[i][j]);
            }
        }
    }

    if (max_dimension >= 2)
    {
        for (size_t i = 0; i < landmarks.size(); ++i)
        {
            for (size_t j = i + 1; j < landmarks.size(); ++j)
            {
                for (size_t k = j + 1; k < landmarks.size(); ++k)
                {
                    const double edge_max =
                        std::max({distance[i][j], distance[i][k], distance[j][k]});
                    if (edge_max <= threshold)
                    {
                        witness_complex.addSimplexWithFiltration(
                            algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j),
                                              static_cast<Index>(k)}),
                            edge_max);
                    }
                }
            }
        }
    }

    if (max_dimension >= 3)
    {
        for (size_t i = 0; i < landmarks.size(); ++i)
        {
            for (size_t j = i + 1; j < landmarks.size(); ++j)
            {
                for (size_t k = j + 1; k < landmarks.size(); ++k)
                {
                    for (size_t l = k + 1; l < landmarks.size(); ++l)
                    {
                        const double edge_max =
                            std::max({distance[i][j], distance[i][k], distance[i][l],
                                      distance[j][k], distance[j][l], distance[k][l]});
                        if (edge_max <= threshold)
                        {
                            witness_complex.addSimplexWithFiltration(
                                algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j),
                                                  static_cast<Index>(k), static_cast<Index>(l)}),
                                edge_max);
                        }
                    }
                }
            }
        }
    }

    const auto exact = computeExactPersistenceZ2(witness_complex, max_dimension);
    overwriteDiagramFromExact(exact, diagram_.get());
}

void PH6HighDimensional::applyAdaptiveTruncation()
{
    if (!diagram_ || diagram_->empty())
    {
        return;
    }
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        return (a.death - a.birth) > (b.death - b.birth);
    });
    const size_t keep = std::max<size_t>(1, pairs.size() / 2);
    diagram_->clear();
    for (size_t i = 0; i < keep; ++i)
    {
        diagram_->addPair(pairs[i].birth, pairs[i].death, pairs[i].dimension);
    }
}

void PH6HighDimensional::computePartialDiagram()
{
    if (!diagram_)
    {
        diagram_ = std::make_unique<::nerve::Diagram>();
    }
    std::vector<PersistencePairRecord> pairs = diagram_->pairs();
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [](const PersistencePairRecord &pair) {
                                   return std::isfinite(pair.death) &&
                                          pair.death + 1e-12 < pair.birth;
                               }),
                pairs.end());
    std::sort(pairs.begin(), pairs.end(),
              [](const auto &a, const auto &b) { return a.dimension < b.dimension; });
    diagram_->clear();
    for (const auto &pair : pairs)
    {
        diagram_->addPair(pair.birth, pair.death, pair.dimension);
    }
}

bool PH6HighDimensional::checkBudget()
{
    return budget_.memory_limit_mb > 0 && budget_.time_limit_ms > 0;
}

void PH6HighDimensional::handleBudgetExceeded()
{
    budget_exceeded_ = true;
    if (!summary_)
    {
        summary_ = std::make_unique<::nerve::CompactSummary>();
    }
}

void PH6HighDimensional::setSamplingStrategy(WitnessSampling strategy)
{
    sampling_strategy_ = strategy;
}

void PH6HighDimensional::setAdaptiveTruncation(bool enable)
{
    adaptive_truncation_enabled_ = enable;
}

void PH6HighDimensional::setLandmarkRatio(double ratio)
{
    if (!std::isfinite(ratio))
    {
        throw std::invalid_argument("landmark ratio must be finite");
    }
    landmark_ratio_ = std::clamp(ratio, 0.01, 1.0);
}

const ::nerve::Diagram &PH6HighDimensional::getDiagram() const
{
    return *diagram_;
}

const ::nerve::CompactSummary &PH6HighDimensional::getSummary() const
{
    return *summary_;
}

const StabilityCertificate &PH6HighDimensional::getCertificate() const
{
    return certificate_;
}

bool PH6HighDimensional::hasDiagram() const
{
    return diagram_ != nullptr;
}

bool PH6HighDimensional::hasSummary() const
{
    return summary_ != nullptr;
}

bool PH6HighDimensional::budgetExceeded() const
{
    return budget_exceeded_;
}

} // namespace nerve::persistence
