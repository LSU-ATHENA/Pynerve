
#pragma once
#include "nerve/core/budget.hpp"
#include "nerve/core/persistence.hpp"
#include "nerve/core/stability_certificate.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
namespace nerve
{
namespace persistence
{
class IncrementalPersistence
{
public:
    explicit IncrementalPersistence(const PersistenceBudget &budget = PersistenceBudget{});
    ResultType incrementalAddPoint(const PointCloud &current_points,
                                   const std::vector<double> &new_point, size_t max_dimension,
                                   const std::vector<size_t> &affected_simplices = {});
    ResultType incrementalRemovePoint(const PointCloud &current_points, size_t point_index,
                                      size_t max_dimension,
                                      const std::vector<size_t> &affected_simplices = {});
    ResultType incrementalAddPoints(const PointCloud &current_points,
                                    const std::vector<std::vector<double>> &new_points,
                                    size_t max_dimension,
                                    const std::vector<std::vector<size_t>> &batch_hints = {});
    ResultType incrementalRemovePoints(const PointCloud &current_points,
                                       const std::vector<size_t> &point_indices,
                                       size_t max_dimension,
                                       const std::vector<std::vector<size_t>> &batch_hints = {});
    void enableStreamingMode(bool enable);
    void setStreamingWindowSize(size_t window_size);
    void setStreamingStride(size_t stride);
    void setBudget(const PersistenceBudget &budget);
    const PersistenceBudget &getBudget() const;
    const ::nerve::Diagram &getDiagram() const;
    const ::nerve::CompactSummary &getSummary() const;
    const StabilityCertificate &getCertificate() const;
    bool hasValidDiagram() const;
    bool budgetExceeded() const;
    bool streamingModeActive() const;
    size_t getTotalOperations() const;
    double getAverageUpdateTime() const;
    void setUpdateStrategy(IncrementalStrategy strategy);
    void enableHintsOptimization(bool enable);
    void setSummaryRefreshThreshold(double threshold);

private:
    PersistenceBudget budget_;
    std::unique_ptr<::nerve::Diagram> current_diagram_;
    std::unique_ptr<::nerve::CompactSummary> summary_;
    StabilityCertificate certificate_;
    IncrementalStrategy update_strategy_;
    bool hints_optimization_enabled_;
    double summary_refresh_threshold_;
    bool budget_exceeded_;
    bool streaming_mode_enabled_;
    size_t streaming_window_size_;
    size_t streaming_stride_;
    size_t total_operations_;
    double total_update_time_;
    std::vector<double> update_times_;
    std::unordered_map<size_t, std::vector<size_t>> point_to_simplices_;
    std::vector<std::vector<double>> streaming_buffer_;
    ResultType performIncrementalUpdate(const PointCloud &points,
                                        const IncrementalOperation &operation, size_t max_dimension,
                                        const std::vector<size_t> &hints);
    void updateSimplicialComplex(const PointCloud &points, const IncrementalOperation &operation);
    std::vector<size_t> computeAffectedSimplices(const IncrementalOperation &operation,
                                                 const std::vector<size_t> &hints);
    ResultType updatePersistenceDiagram(const PointCloud &points, size_t max_dimension,
                                        const std::vector<size_t> &affected_simplices);
    bool shouldRefreshSummary();
    void refreshSummary();
    void processStreamingBuffer();
    void updateStreamingWindow(const std::vector<double> &new_point);
    bool checkBudget();
    void handleBudgetExceeded();
    void recordUpdateTime(double time);
    void updatePerformanceMetrics();
};
class StreamingPersistence
{
public:
    explicit StreamingPersistence(const PersistenceBudget &budget = PersistenceBudget{},
                                  size_t window_size = 1000, size_t stride = 100);
    ResultType processStreamPoint(const std::vector<double> &point);
    ResultType processStreamBatch(const std::vector<std::vector<double>> &batch);
    void setWindowSize(size_t size);
    void setStride(size_t stride);
    void enableSlidingWindow(bool enable);
    const ::nerve::Diagram &getCurrentDiagram() const;
    const ::nerve::CompactSummary &getWindowSummary() const;
    size_t getPointsProcessed() const;
    size_t getWindowsProcessed() const;
    double getAverageProcessingTime() const;

private:
    IncrementalPersistence incremental_;
    size_t window_size_;
    size_t stride_;
    bool sliding_window_enabled_;
    std::vector<std::vector<double>> current_window_;
    size_t points_processed_;
    size_t windows_processed_;
    std::vector<double> processing_times_;
    void updateWindow(const std::vector<double> &point);
    void processCurrentWindow();
    void slideWindow();
};
} // namespace persistence
} // namespace nerve
