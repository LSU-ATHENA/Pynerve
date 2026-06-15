#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

struct GpuScanResult
{
    std::vector<int> stable_columns;
    std::vector<int> pivot_candidates;
    std::vector<int> zero_addition;
    std::vector<int> claimed_pivots;
    std::vector<int> unstable_columns;
};

struct SubMatrix
{
    std::vector<std::vector<int>> columns;
    std::vector<int> column_map;
    std::vector<double> filtration_values;
};

class HyphaReducer
{
public:
    struct Config
    {
        float scan_ratio = 1.0f;
        int unstable_threshold = 1000;
        bool use_clearing = true;
    };

    HyphaReducer() = default;
    explicit HyphaReducer(const Config &cfg)
        : config_(cfg)
    {}

    std::vector<Pair> compute(const algebra::BoundaryMatrix &matrix);

    void setConfig(const Config &cfg) { config_ = cfg; }
    const Config &config() const { return config_; }

private:
    GpuScanResult gpuBoundaryScan(const algebra::BoundaryMatrix &matrix);
    void cpuClearingCompression(GpuScanResult &scan, const algebra::BoundaryMatrix &matrix);
    std::vector<Pair> cpuSubmatrixReduction(const SubMatrix &sub);

    Config config_;
};

} // namespace nerve::persistence
