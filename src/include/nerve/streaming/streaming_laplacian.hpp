#pragma once
#include "nerve/core_types.hpp"
#include "nerve/streaming/incremental.hpp"

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nerve::streaming
{

struct IncrementalLaplacianConfig
{
    Index max_dimension = 2;
    double smoothing_sigma = 0.1;
    double eigenvalue_tolerance = 1e-10;
};

struct LaplacianSpectrum
{
    std::vector<double> eigenvalues;
    double trace = 0.0;
    double frobenius_norm = 0.0;
    Size rank = 0;
};

struct LaplacianState
{
    Index dimension;
    Size matrix_size;
    double birth_time;
    std::vector<double> diagonal;
    std::vector<double> off_diagonal;
    std::vector<Index> column_start;
    std::vector<Index> row_index;
    std::vector<double> values;
    bool is_valid;
};

class IncrementalGraphLaplacian
{
public:
    explicit IncrementalGraphLaplacian(const IncrementalLaplacianConfig &config);
    IncrementalGraphLaplacian();
    ~IncrementalGraphLaplacian();

    void addEdge(Index u, Index v, double weight);
    void removeEdge(Index u, Index v);
    void addVertex(Index vertex_id);
    void removeVertex(Index vertex_id);

    const LaplacianState &getState() const;
    LaplacianSpectrum computeSpectrum() const;
    double computeSpectralGap() const;
    std::vector<double> computeFiedlerVector() const;
    double computeAlgebraicConnectivity() const;
    double computeTotalEffectiveResistance() const;

    void reset();
    Size getVertexCount() const;
    Size getEdgeCount() const;

private:
    IncrementalLaplacianConfig config_;
    LaplacianState state_;
    std::vector<double> vertex_weights_;
    std::unordered_map<Index, double> degree_map_;
    std::mutex state_mutex_;

    void updateDiagonal(Index idx, double delta);
    void updateOffDiagonal(Index u, Index v, double delta);
    Size edgeIndex(Index u, Index v) const;
    void rebuildMatrix();
};

std::vector<double> lanczosIteration(const LaplacianState &L, Size k, double tolerance);

class StreamingLaplacianProcessor
{
public:
    explicit StreamingLaplacianProcessor(const IncrementalLaplacianConfig &config);

    void ingestPoints(const std::vector<double> &points, Size dim,
                      const std::vector<Index> &indices);
    void ingestPointBatch(const std::vector<double> &points, Size dim, Size batch_size);
    void flushWindow();

    LaplacianSpectrum getLatestSpectrum() const;
    std::vector<LaplacianSpectrum> getSpectrumTimeline() const;
    double getStabilityScore() const;
    Size getProcessedCount() const;

    void setWindowSize(Size window_size);
    void setEdgeThreshold(double radius);

private:
    IncrementalLaplacianConfig config_;
    IncrementalGraphLaplacian laplacian_;
    Size window_size_;
    Size processed_count_;
    double edge_threshold_;
    std::vector<LaplacianSpectrum> spectrum_timeline_;
    std::vector<double> point_buffer_;

    double pairwiseDistance(const double *a, const double *b, Size dim) const;
    void processWindow(const std::vector<double> &points, Size dim,
                       const std::vector<Index> &indices);
};

} // namespace nerve::streaming
