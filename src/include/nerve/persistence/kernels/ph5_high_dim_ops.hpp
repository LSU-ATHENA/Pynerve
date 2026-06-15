
#pragma once
#include "nerve/core/budget.hpp"
#include "nerve/core/persistence.hpp"
#include "nerve/core/stability_certificate.hpp"

#include <memory>
#include <vector>
namespace nerve
{
namespace persistence
{
class PH5HighDimensional
{
public:
    explicit PH5HighDimensional(const PersistenceBudget &budget = PersistenceBudget{});
    ResultType computePersistenceCohomology(const SimplicialComplex &complex, size_t max_dimension);
    void setBudget(const PersistenceBudget &budget);
    const PersistenceBudget &getBudget() const;
    void setCohomologyOrderingStrategy(CohomologyOrdering strategy);
    void enableOptimization(bool enable);
    const ::nerve::Diagram &getDiagram() const;
    const ::nerve::CompactSummary &getSummary() const;
    const StabilityCertificate &getCertificate() const;
    bool hasDiagram() const;
    bool hasSummary() const;
    bool budgetExceeded() const;

private:
    PersistenceBudget budget_;
    std::unique_ptr<::nerve::Diagram> diagram_;
    std::unique_ptr<::nerve::CompactSummary> summary_;
    StabilityCertificate certificate_;
    CohomologyOrdering ordering_strategy_;
    bool optimization_enabled_;
    bool budget_exceeded_ = false;
    ResultType computeCohomologyReduction(const SimplicialComplex &complex, size_t max_dimension);
    void buildCohomologyComplex(const SimplicialComplex &complex, size_t max_dimension);
    void applyCleverOrdering();
    void reduceBoundaryMatrix();
    void extractPersistencePairs();
    bool checkBudget();
    void handleBudgetExceeded();
    void optimizeCohomologyComputation();
    void applyColumnReductions();
    void eliminatePersistentPairs();
};
class PH6HighDimensional
{
public:
    explicit PH6HighDimensional(const PersistenceBudget &budget = PersistenceBudget{});
    ResultType computePersistenceWitness(const PointCloud &points, size_t max_dimension,
                                         double landmark_ratio = 0.1);
    void setSamplingStrategy(WitnessSampling strategy);
    void setAdaptiveTruncation(bool enable);
    void setLandmarkRatio(double ratio);
    const ::nerve::Diagram &getDiagram() const;
    const ::nerve::CompactSummary &getSummary() const;
    const StabilityCertificate &getCertificate() const;
    bool hasDiagram() const;
    bool hasSummary() const;
    bool budgetExceeded() const;

private:
    PersistenceBudget budget_;
    std::unique_ptr<::nerve::Diagram> diagram_;
    std::unique_ptr<::nerve::CompactSummary> summary_;
    StabilityCertificate certificate_;
    WitnessSampling sampling_strategy_;
    bool adaptive_truncation_enabled_;
    double landmark_ratio_;
    bool budget_exceeded_ = false;
    ResultType computeHierarchicalWitness(const PointCloud &points, size_t max_dimension);
    std::vector<size_t> selectLandmarks(const PointCloud &points, size_t num_landmarks);
    void buildWitnessComplex(const PointCloud &points, const std::vector<size_t> &landmarks,
                             size_t max_dimension);
    void applyAdaptiveTruncation();
    void computePartialDiagram();
    bool checkBudget();
    void handleBudgetExceeded();
};
} // namespace persistence
} // namespace nerve
