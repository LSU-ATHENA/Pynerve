## Streaming PH optimization

### AcceleratedStreamingPh

Memory-efficient streaming persistence with error-bounded approximations.
Uses witness complexes, sparse filtrations, and coarsening to bound memory
growth.

```cpp
class AcceleratedStreamingPh {
public:
    struct StreamingConfig {
        bool use_witness_complexes = true;
        bool enable_sparse_filtrations = true;
        double error_budget = 0.01;
        bool enable_incrementality = true;
        size_t max_active_simplex_growth = 10000;
        bool enable_coarsening = true;
        size_t coarsening_threshold = 5000;
        bool enable_summary_cap = true;
    };

    struct StreamingResult {
        std::vector<std::pair<float, float>> persistence_diagram;
        double actual_error;
        bool used_approximation;
        double computation_time_ms;
        ErrorCode error_code;
    };

    explicit AcceleratedStreamingPh(const StreamingConfig& config);

    StreamingResult computeStreamingPh(
        const std::vector<std::vector<float>>& points);
    void updateStreamingPh(
        const std::vector<std::vector<float>>& new_points,
        const std::vector<uint32_t>& changed_indices);
    void capSimplexGrowth();
    void applyCoarsening();

    StreamingResult computeSummaryOnly(
        const std::vector<std::vector<float>>& points,
        const CallContract& contract);

    std::vector<std::pair<float, float>> compute(
        const std::vector<std::vector<float>>& window);
    std::vector<std::pair<float, float>> computeExact(
        const std::vector<std::vector<float>>& window);
    std::vector<std::pair<float, float>> computeIncremental(
        const std::vector<std::vector<float>>& window);

    bool isCacheValid() const;
};
```

### StreamingPH (approximate streaming)

The `StreamingPH` class in `src/include/nerve/approximate/streaming_ph.hpp`:

The StreamingPH system is composed of several components: `WitnessComplex` provides landmark-based sparse approximation; `SketchBasedPH` uses Johnson-Lindenstrauss sketching for fast PH; `AdaptiveSampler` performs importance sampling based on error tracking; `ErrorTracker` provides confidence-bounded error estimation; `ThroughputAccuracyOptimizer` auto-tunes the configuration for the throughput-accuracy trade-off; and `StreamingPHManager` acts as a singleton managing multiple named streams.

```cpp
#include <nerve/approximate/streaming_ph.hpp>

using namespace nerve::approximate;

StreamingConfig cfg;
cfg.error_budget = 0.01;
cfg.sketch_size = 1000;

StreamingPH stream(cfg);
stream.addPointsBatch(points, timestamps);
auto pairs = stream.compute();                     // approximate pairs
auto bounded = stream.computePersistenceWithBounds();  // with error bars
auto metrics = stream.getPerformanceMetrics();
// metrics.points_processed_per_second, metrics.current_error_estimate
```

### Named stream management

```cpp
auto& mgr = StreamingPHManager::instance();
mgr.createStream("sensor_1", cfg);
auto stream = mgr.getStream("sensor_1");
stream->addPoint({0.5, 1.2, 3.7}, timestamp_ns);
mgr.computeAllPersistence();
```

### Python API

```python
from pynerve.optimization import StreamingPH

stream = StreamingPH(
    error_budget=0.01,
    sketch_size=1000,
    use_witness_complexes=True,
)

stream.add_points_batch(points, timestamps)
pairs = stream.compute()
print(f"Found {len(pairs)} pairs, error estimate: {stream.current_error_estimate}")
```


## Streaming PH algorithm details

### Witness complex construction

```cpp
// Landmark selection via max-min sampling
std::vector<int> selectLandmarks(
    const std::vector<std::vector<float>>& points,
    int num_landmarks) {

    std::vector<int> landmarks;
    std::vector<float> min_dist(points.size(), INFINITY);

    // First landmark: random
    landmarks.push_back(rand() % points.size());

    for (int l = 1; l < num_landmarks; l++) {
        // Pick point farthest from existing landmarks
        int best = -1;
        float best_dist = -1;
        for (size_t i = 0; i < points.size(); i++) {
            float d = distance(points[i], points[landmarks.back()]);
            min_dist[i] = min(min_dist[i], d);
            if (min_dist[i] > best_dist) {
                best_dist = min_dist[i];
                best = i;
            }
        }
        landmarks.push_back(best);
    }
    return landmarks;
}
```

### Error tracking

```cpp
class ErrorTracker {
    double estimated_error;
    double confidence_interval;
    int exact_checks;

public:
    // Compare approximate vs exact result
    void checkAccuracy(
        const std::vector<std::pair<float,float>>& approx,
        const std::vector<std::pair<float,float>>& exact) {

        double error = computeBottleneckDistance(approx, exact);
        estimated_error = (estimated_error * exact_checks + error)
                          / (exact_checks + 1);
        exact_checks++;

        // Update confidence interval (95% CI)
        double std_error = stddev(error_history);
        confidence_interval = 1.96 * std_error / sqrt(exact_checks);
    }

    bool isErrorWithinBudget(double budget) const {
        return estimated_error + confidence_interval < budget;
    }
};
```

### Throughput-accuracy auto-tuning

```cpp
class ThroughputAccuracyOptimizer {
    struct Trial {
        int sketch_size;
        int num_landmarks;
        double throughput;  // points/second
        double error;
    };

    std::vector<Trial> history;

public:
    Config autoTune(double target_error, double min_throughput) {
        // Bayesian optimization over (sketch_size, num_landmarks)
        // to maximize throughput while keeping error < target

        Config best = {1000, 100};  // defaults
        double best_score = 0;

        for (auto& trial : history) {
            if (trial.error < target_error &&
                trial.throughput > min_throughput) {
                double score = trial.throughput / trial.error;
                if (score > best_score) {
                    best_score = score;
                    best = {trial.sketch_size, trial.num_landmarks};
                }
            }
        }
        return best;
    }
};
```


## Named stream management example

```python
from pynerve.optimization import StreamingPHManager

mgr = StreamingPHManager.instance()

# Create streams for multiple sensors
mgr.createStream("sensor_1", StreamingConfig(error_budget=0.01))
mgr.createStream("sensor_2", StreamingConfig(error_budget=0.05))

# Feed data asynchronously
for timestamp, data in sensor_data_stream:
    mgr.getStream("sensor_1").addPoint(data, timestamp)

# Compute all streams in parallel
results = mgr.computeAllPersistence()
for name, pairs in results.items():
    print(f"{name}: {len(pairs)} pairs")

# Get performance metrics
for name in mgr.getStreamNames():
    metrics = mgr.getStream(name).getPerformanceMetrics()
    print(f"{name}: {metrics.points_processed_per_second:.0f} pts/s, "
          f"error: {metrics.current_error_estimate:.4f}")
```


## FAQ

**Q: When should I use streaming PH vs batch PH?**
A: Use streaming when the data does not fit in memory, when points arrive incrementally, or when you need bounded-latency results. Use batch PH for small datasets where exact results are needed.

**Q: How does the error budget interact with coarsening?**
A: When the error tracker detects that the budget may be exceeded, it triggers coarsening (merging nearby landmarks) to reduce complexity and error. Coarsening temporarily reduces throughput but improves accuracy.

**Q: Can I use streaming PH with non-Euclidean distances?**
A: Yes. The witness complex only requires pairwise distances between points and landmarks. Provide a custom distance function via the `StreamingConfig::distance_function` callback.


### Cross-references

- `pynerve.optimization`: Optimization module overview
- `pynerve.optimization.compact_summary`: Compact summaries for streaming
- `pynerve.specialized`: Zigzag persistence for time-varying data
- `pynerve.approximate`: Approximation algorithms for streaming
