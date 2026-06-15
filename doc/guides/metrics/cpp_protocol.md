# Distance callable protocol (C++)

For C++ custom metrics, implement the `DistanceMetric` interface:

```cpp
#include <nerve/algorithms/distance.hpp>

class CustomMetric : public nerve::algorithms::DistanceMetric<float> {
public:
    float compute(std::span<const float> a,
                  std::span<const float> b) const override {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    std::vector<float> compute_matrix(
        std::span<const float> points,
        size_t n_points, size_t dim) const override {
        // Optional: optimized batched implementation
        std::vector<float> matrix(n_points * n_points, 0.0f);
        for (size_t i = 0; i < n_points; ++i) {
            for (size_t j = i + 1; j < n_points; ++j) {
                auto a = points.subspan(i * dim, dim);
                auto b = points.subspan(j * dim, dim);
                float d = compute(a, b);
                matrix[i * n_points + j] = d;
                matrix[j * n_points + i] = d;
            }
        }
        return matrix;
    }
};
```

[Back to index](index.md)
