# Pipeline

```cpp
class CompactSummaryPipeline {
    struct PipelineConfig {
        float max_computation_time_ms = 5.0f;
        size_t max_data_points = 10000;
        float sampling_rate = 0.1f;
        bool enable_approximation = true;
        uint32_t random_seed = 42;
        Size max_persistence_dim = 2;
        double max_filtration_radius = 1.0;
    };

    explicit CompactSummaryPipeline(const PipelineConfig& config);

    ErrorResult<CompactSummary> computeSummary(...) const;
    CompactSummary computeApproximateSummary(...) const;
    void updateSummary(...) const;
    PerformanceMetrics getLastMetrics() const;
};
```


[Back to index](index.md)
