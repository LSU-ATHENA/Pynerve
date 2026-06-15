# `nerve::filtration` -- VR Filtration Construction

```cpp
#include <nerve/filtration/vr_filtration.hpp>

namespace nerve::filtration;

class VRFiltration {
public:
    struct Config {
        double max_radius = 1.0;
        size_t max_dim = 2;
        bool use_hnsw = false;
        size_t hnsw_M = 16;
        size_t hnsw_ef_construction = 200;
    };

    explicit VRFiltration(const Config& config = {});

    // Build sorted simplex list from distance matrix
    std::vector<Simplex> build(
        std::span<const double> distance_matrix,
        size_t n_points);

    // Build from point cloud (computes distance internally)
    std::vector<Simplex> buildFromPoints(
        std::span<const double> points, size_t n, size_t dim);

    // Sparse VR with landmarks
    std::vector<Simplex> buildSparse(
        std::span<const double> points, size_t n, size_t dim,
        double landmark_ratio = 0.1);
};

class EdgeCollapser {
public:
    EdgeCollapser(double tolerance = 0.0);

    // Collapse edges, preserving persistence up to tolerance
    std::vector<Simplex> collapse(
        std::span<const Simplex> simplices,
        size_t n_points);

    // Get collapse statistics
    size_t original_edges() const;
    size_t collapsed_edges() const;
    double compression_ratio() const;
};
```

**Cost (VRFiltration::build):** O(n^2) edge generation + O(m log m) sorting.

**Cost (EdgeCollapser::collapse):** O(m * alpha(m)) near-linear in edges.

<- [C++ API Overview](index.md)
