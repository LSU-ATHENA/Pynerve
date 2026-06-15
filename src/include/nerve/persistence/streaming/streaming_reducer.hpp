#pragma once

#include "nerve/algebra/bit_packed.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/persistence.hpp"
#include "nerve/core_types.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::streaming
{

constinit const size_t DEFAULT_CHUNK_SIZE = 10000;
constinit const size_t MAX_PIVOT_CACHE_SIZE = 1000000;

struct Clique
{
    std::vector<int> vertices;
    double filtration_value = 0.0;
    size_t simplex_index = 0;

    bool operator<(const std::vector<int> &other) const { return vertices < other; }
    bool operator<(const Clique &other) const { return vertices < other.vertices; }
    bool operator==(const std::vector<int> &other) const { return vertices == other; }
    bool operator==(const Clique &other) const { return vertices == other.vertices; }
};

class BronKerboschEnumerator
{
public:
    using Clique = struct Clique;

    explicit BronKerboschEnumerator(const std::vector<std::vector<int>> &adjacency);

    [[nodiscard]] std::vector<Clique> enumerateCliques(std::span<const double> edge_weights,
                                                       size_t n_points, int max_dim);

private:
    std::vector<std::vector<int>> adjacency_;

    [[nodiscard]] bool isAdjacent(int u, int v) const;
};

class SimplexHash
{
public:
    explicit SimplexHash(size_t n_points);

    [[nodiscard]] size_t encode(const std::vector<int> &vertices) const;
    [[nodiscard]] std::vector<int> decode(size_t index, int dim) const;
    [[nodiscard]] size_t binomial(int n, int k) const;

private:
    size_t n_points_;
    std::vector<std::vector<size_t>> binom_;
};

using Simplex = Clique;

class StreamingColumnGenerator
{
public:
    StreamingColumnGenerator(std::span<const double> points, size_t n_points, size_t point_dim,
                             double max_distance);

    [[nodiscard]] std::vector<int> generateColumn(size_t simplex_idx) const;
    [[nodiscard]] size_t getNumSimplices() const;
    [[nodiscard]] double getFiltrationValue(size_t simplex_idx) const;
    [[nodiscard]] int getSimplexDimension(size_t simplex_idx) const;
    [[nodiscard]] std::vector<int> simplexToVertices(size_t simplex_idx) const;
    [[nodiscard]] int verticesToSimplexIndex(const std::vector<int> &vertices) const;

private:
    std::span<const double> points_;
    size_t n_points_;
    size_t point_dim_;
    double max_distance_;

    std::vector<std::vector<int>> adjacency_;
    std::vector<double> edge_weights_;
    std::vector<Simplex> all_simplices_;
    std::map<std::vector<int>, size_t> simplex_index_map_;
    std::vector<Simplex> edge_simplices_;
    std::vector<Simplex> triangle_list_;
    std::vector<Simplex> tetra_list_;
    std::vector<Simplex> four_simplices_;
    size_t total_simplices_;

    void buildAdjacency();
    void enumerateAllSimplices();
    [[nodiscard]] double computeDistance(size_t i, size_t j) const;
};

class StreamingReducer
{
public:
    explicit StreamingReducer(bool use_bit_packed = true);

    [[nodiscard]] ::nerve::persistence::PersistenceDiagram
    reduce(const StreamingColumnGenerator &generator);
    [[nodiscard]] size_t getMemoryUsage() const;

private:
    bool use_bit_packed_;
    std::vector<int> current_column_;
    std::unordered_map<int, int> pivot_to_column_;
    std::unordered_map<int, std::vector<int>> cached_columns_;
    std::unique_ptr<algebra::compressed::BitPackedZ2Matrix> bit_matrix_;

    void loadColumn(const std::vector<int> &column);
    [[nodiscard]] std::ptrdiff_t findPivot() const;
    [[nodiscard]] bool hasCachedColumn(int pivot) const;
    void xorWithCachedColumn(int pivot);
    void cacheColumn(int pivot, size_t col_idx);
    void clearCurrentColumn();
};

} // namespace nerve::persistence::streaming
