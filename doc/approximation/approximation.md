# Approximation

## Quick start

```python
import pynerve.approximation as approx

# Sliced Wasserstein distance between diagrams
sw = approx.SlicedWasserstein(num_projections=100)
dist = sw.computeDistance(diagram1, diagram2)
# -> dist = 0.037

# LSH-based diagram similarity search
lsh = approx.DiagramLSH(num_hash_tables=10, num_hash_functions=5)
lsh.buildHashTables(database_diagrams)
similar_ids = lsh.findSimilarDiagrams(query_diagram, max_results=5)
# -> similar_ids = [3, 17, 42, 88, 91]
```

Johnson-Lindenstrauss random projection for dimensionality reduction before
persistence computation. Epsilon-net landmark selection (uniform, maxmin
farthest-point sampling) to build witness or sparse Rips complexes.


## API

```cpp
#include <nerve/approximation/distance_approximation.hpp>

namespace nerve::approximation {

struct DiagramPoint {
    float birth, death, persistence;
    uint8_t dimension;
    bool isValid() const;
    float getLifetime() const;
};

struct ApproximationConfig {
    size_t num_projections = 100;
    uint32_t random_seed = 42;
    float approximation_tolerance = 0.01f;
    bool enable_adaptive_projections = true;
};

// Sliced Wasserstein: random-project then 1D Wasserstein
class SlicedWasserstein {
    explicit SlicedWasserstein(const ApproximationConfig& config);
    float computeDistance(const vector<DiagramPoint>& d1,
                          const vector<DiagramPoint>& d2);
    vector<vector<float>> computeDistanceMatrix(
        const vector<vector<DiagramPoint>>& diagrams);
    vector<vector<float>> generateProjections(size_t num_projections);
    DistanceStats getStats() const;
};

// Locality-sensitive hashing for diagram retrieval
class DiagramLSH {
    explicit DiagramLSH(const ApproximationConfig& config);
    void buildHashTables(const vector<vector<DiagramPoint>>& diagrams);
    vector<size_t> findSimilarDiagrams(
        const vector<DiagramPoint>& query, size_t max_results = 10);
    vector<vector<size_t>> clusterDiagrams(
        const vector<vector<DiagramPoint>>& diagrams,
        float similarity_threshold = 0.8f);
};

// Coarse-grained grid-based matching
class CoarseGrainedMatcher {
    explicit CoarseGrainedMatcher(const CoarseConfig& config);
    vector<vector<float>> discretizeDiagram(
        const vector<DiagramPoint>& diagram);
    float computeGridDistance(const vector<vector<float>>& grid1,
                              const vector<vector<float>>& grid2);
};

// Approximate bottleneck with landmark sampling
class ApproximateBottleneck {
    explicit ApproximateBottleneck(const BottleneckConfig& config);
    float computeDistance(const vector<DiagramPoint>& d1,
                          const vector<DiagramPoint>& d2);
    float computeLandmarkDistance(const vector<DiagramPoint>& d1,
                                  const vector<DiagramPoint>& d2);
};

// Manager with fast-dispatch
class DistanceApproximationManager {
    static DistanceApproximationManager& instance();
    float computeFastDistance(
        const vector<DiagramPoint>& d1, const vector<DiagramPoint>& d2,
        const string& method = "sliced_wasserstein");
    PerformanceStats getPerformanceStats() const;
};

}
```

Use `random_projection` when the ambient dimension exceeds 50-100 -- JL lemma
guarantees pairwise distances are preserved within (1 +/- epsilon) after
projecting to O(log n / epsilon^2) dimensions. Use `landmarks` (farthest-point
sampling or epsilon-net) to reduce n before computing persistence on large
point clouds.
