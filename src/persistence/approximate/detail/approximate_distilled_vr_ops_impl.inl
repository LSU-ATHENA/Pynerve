#include "nerve/persistence/approximate/distilled_vr_filtration.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::distilled {

namespace {

struct MorseSimplex {
    std::vector<int> vertices;
    int dimension;
    double filtration_value;
    int index;

    bool operator<(const MorseSimplex& other) const {
        if (filtration_value != other.filtration_value) {
            return filtration_value < other.filtration_value;
        }
        return dimension < other.dimension;
    }
};

struct MorsePairing {
    int simplex_a;
    int simplex_b;
    bool is_critical;
};

bool isCoface(const MorseSimplex& a, const MorseSimplex& b) {
    if (a.dimension != b.dimension + 1) return false;
    return std::ranges::includes(a.vertices, b.vertices);
}

int nextSimplexIndex(size_t current_size) {
    if (current_size >= static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("distilled simplex count exceeds index range");
    }
    return static_cast<int>(current_size);
}

bool hasFiniteSafePointCoordinates(const std::vector<double>& points, size_t required_values,
                                   size_t point_dim) {
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim)) /
        4.0L;
    for (size_t i = 0; i < required_values; ++i) {
        const double value = points[i];
        if (!std::isfinite(value) ||
            std::abs(static_cast<long double>(value)) > safe_abs) {
            return false;
        }
    }
    return true;
}

std::vector<int> findCofaces(const MorseSimplex& simplex,
                             const std::vector<MorseSimplex>& all_simplices) {
    std::vector<int> cofaces;
    for (size_t i = 0; i < all_simplices.size(); ++i) {
        if (isCoface(all_simplices[i], simplex)) {
            cofaces.push_back(static_cast<int>(i));
        }
    }
    return cofaces;
}

std::vector<MorsePairing> buildMorseVectorField(
    const std::vector<MorseSimplex>& simplices,
    const DistilledVRConfig& config) {

    std::vector<MorsePairing> pairings;
    std::vector<bool> is_paired(simplices.size(), false);

    for (size_t i = 0; i < simplices.size(); ++i) {
        if (is_paired[i]) continue;

        auto cofaces = findCofaces(simplices[i], simplices);
        int unpaired_coface = -1;

        for (int coface_idx : cofaces) {
            if (!is_paired[coface_idx] &&
                simplices[coface_idx].filtration_value <= config.max_radius) {
                unpaired_coface = coface_idx;
                break;  // Take first valid coface
            }
        }

        if (unpaired_coface >= 0) {
            MorsePairing pair;
            pair.simplex_a = unpaired_coface;
            pair.simplex_b = static_cast<int>(i);
            pair.is_critical = false;
            pairings.push_back(pair);

            is_paired[i] = true;
            is_paired[unpaired_coface] = true;
        } else {
            MorsePairing pair;
            pair.simplex_a = static_cast<int>(i);
            pair.simplex_b = -1;
            pair.is_critical = true;
            pairings.push_back(pair);
        }
    }

    return pairings;
}

void validateDistilledInputs(const std::vector<double>& points,
                             size_t point_dim,
                             size_t num_points,
                             const DistilledVRConfig& config) {
    if (point_dim == 0) {
        throw std::invalid_argument("point_dim must be positive");
    }
    if (num_points > std::numeric_limits<size_t>::max() / point_dim) {
        throw std::invalid_argument("points does not contain num_points * point_dim values");
    }
    const size_t required_values = num_points * point_dim;
    if (points.size() != required_values) {
        throw std::invalid_argument("points does not contain num_points * point_dim values");
    }
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("point count exceeds distilled simplex index range");
    }
    if (config.max_dim < 0) {
        throw std::invalid_argument("max_dim must be non-negative");
    }
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0) {
        throw std::invalid_argument("max_radius must be finite and non-negative");
    }
    if (!hasFiniteSafePointCoordinates(points, required_values, point_dim)) {
        throw std::invalid_argument("point coordinates must be finite and distance-safe");
    }
}

std::vector<MorseSimplex> buildReducedVRComplex(
    const std::vector<double>& points,
    size_t point_dim,
    size_t num_points,
    const DistilledVRConfig& config) {

    std::vector<MorseSimplex> simplices;
    std::set<std::vector<int>> emitted;
    std::map<std::pair<int, int>, double> edge_filtration;

    for (size_t i = 0; i < num_points; ++i) {
        MorseSimplex s;
        s.vertices = {static_cast<int>(i)};
        s.dimension = 0;
        s.filtration_value = 0.0;
        s.index = nextSimplexIndex(simplices.size());
        simplices.push_back(s);
        emitted.insert(s.vertices);
    }

    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d) {
                double diff = points[i * point_dim + d] - points[j * point_dim + d];
                const double contribution = diff * diff;
                if (!std::isfinite(contribution) ||
                    dist_sq > std::numeric_limits<double>::max() - contribution) {
                    dist_sq = std::numeric_limits<double>::infinity();
                    break;
                }
                dist_sq += contribution;
            }
            double dist = std::sqrt(dist_sq);

            if (std::isfinite(dist) && dist <= config.max_radius) {
                MorseSimplex s;
                s.vertices = {static_cast<int>(i), static_cast<int>(j)};
                s.dimension = 1;
                s.filtration_value = dist;
                s.index = nextSimplexIndex(simplices.size());
                simplices.push_back(s);
                emitted.insert(s.vertices);
                edge_filtration[{static_cast<int>(i), static_cast<int>(j)}] = dist;
            }
        }
    }

    for (int dim = 2; dim <= config.max_dim; ++dim) {
        std::vector<size_t> lower_simplices;
        for (size_t i = 0; i < simplices.size(); ++i) {
            if (simplices[i].dimension == dim - 1) {
                lower_simplices.push_back(i);
            }
        }

        for (size_t idx_a = 0; idx_a < lower_simplices.size(); ++idx_a) {
            for (size_t idx_b = idx_a + 1; idx_b < lower_simplices.size(); ++idx_b) {
                const auto& a = simplices[lower_simplices[idx_a]];
                const auto& b = simplices[lower_simplices[idx_b]];

                std::vector<int> common;
                std::set_intersection(a.vertices.begin(), a.vertices.end(),
                                     b.vertices.begin(), b.vertices.end(),
                                     std::back_inserter(common));

                if (static_cast<int>(common.size()) == dim - 1) {
                    std::vector<int> new_vertices = a.vertices;
                    for (int v : b.vertices) {
                        if (std::ranges::find(common, v) == common.end()) {
                            new_vertices.push_back(v);
                            break;
                        }
                    }
                    std::ranges::sort(new_vertices);
                    new_vertices.erase(std::unique(new_vertices.begin(), new_vertices.end()),
                                       new_vertices.end());
                    if (static_cast<int>(new_vertices.size()) != dim + 1 ||
                        emitted.contains(new_vertices)) {
                        continue;
                    }

                    double max_edge = 0.0;
                    bool complete = true;
                    for (size_t i = 0; i < new_vertices.size(); ++i) {
                        for (size_t j = i + 1; j < new_vertices.size(); ++j) {
                            const std::pair<int, int> edge{
                                std::min(new_vertices[i], new_vertices[j]),
                                std::max(new_vertices[i], new_vertices[j])};
                            const auto it = edge_filtration.find(edge);
                            if (it == edge_filtration.end()) {
                                complete = false;
                                break;
                            }
                            max_edge = std::max(max_edge, it->second);
                        }
                        if (!complete) {
                            break;
                        }
                    }

                    if (complete && max_edge <= config.max_radius) {
                        MorseSimplex s;
                        s.vertices = new_vertices;
                        s.dimension = dim;
                        s.filtration_value = max_edge;
                        s.index = nextSimplexIndex(simplices.size());
                        simplices.push_back(s);
                        emitted.insert(s.vertices);
                    }
                }
            }
        }
    }

    std::ranges::sort(simplices, [](const MorseSimplex& a, const MorseSimplex& b) {
        return a.filtration_value < b.filtration_value ||
               (a.filtration_value == b.filtration_value && a.dimension < b.dimension);
    });

    for (size_t i = 0; i < simplices.size(); ++i) {
        simplices[i].index = nextSimplexIndex(i);
    }

    return simplices;
}

}  // namespace

DistilledFiltration buildDistilledFiltration(
    const std::vector<double>& points,
    size_t point_dim,
    size_t num_points,
    const DistilledVRConfig& config) {

    validateDistilledInputs(points, point_dim, num_points, config);

    DistilledFiltration result;
    result.config = config;

    auto start = std::chrono::high_resolution_clock::now();

    auto start_build = std::chrono::high_resolution_clock::now();
    auto simplices = buildReducedVRComplex(points, point_dim, num_points, config);
    auto end_build = std::chrono::high_resolution_clock::now();
    result.original_complex_size = static_cast<int>(simplices.size());
    result.build_time_ms = std::chrono::duration<double, std::milli>(
        end_build - start_build).count();

    auto start_morse = std::chrono::high_resolution_clock::now();
    auto pairings = buildMorseVectorField(simplices, config);
    auto end_morse = std::chrono::high_resolution_clock::now();
    result.morse_time_ms = std::chrono::duration<double, std::milli>(
        end_morse - start_morse).count();

    std::vector<bool> is_critical(simplices.size(), true);
    for (const auto& pair : pairings) {
        if (!pair.is_critical) {
            is_critical[pair.simplex_a] = false;
            is_critical[pair.simplex_b] = false;
        }
    }

    for (size_t i = 0; i < simplices.size(); ++i) {
        if (is_critical[i]) {
            DistilledSimplex ds;
            ds.vertices = simplices[i].vertices;
            ds.dimension = simplices[i].dimension;
            ds.filtration_value = simplices[i].filtration_value;
            ds.original_index = simplices[i].index;
            result.simplices.push_back(ds);
        }
    }

    std::ranges::sort(result.simplices, {}, &DistilledSimplex::filtration_value);

    result.distilled_size = static_cast<int>(result.simplices.size());
    result.reduction_ratio =
        result.original_complex_size > 0
            ? static_cast<double>(result.distilled_size) / result.original_complex_size
            : 0.0;

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(
        end - start).count();

    return result;
}

DistilledPersistenceResult computePersistenceDistilled(
    const DistilledFiltration& filtration,
    const DistilledVRConfig& config) {

    DistilledPersistenceResult result;

    auto start = std::chrono::high_resolution_clock::now();

    int n = filtration.distilled_size;
    std::vector<std::vector<int>> boundary_matrix(n);

    for (size_t i = 0; i < filtration.simplices.size(); ++i) {
        const auto& simplex = filtration.simplices[i];

        for (size_t j = 0; j < i; ++j) {
            const auto& candidate = filtration.simplices[j];
            if (candidate.dimension == simplex.dimension - 1) {
                if (std::ranges::includes(simplex.vertices, candidate.vertices)) {
                    boundary_matrix[i].push_back(static_cast<int>(j));
                }
            }
        }
    }

    if (config.use_bit_parallel && n > 1000) {
        using namespace bitparallel;

        std::vector<BitColumn> bit_columns;
        bit_columns.reserve(boundary_matrix.size());
        int max_row = n;

        for (const auto& col : boundary_matrix) {
            bit_columns.push_back(buildBitColumn(col, max_row));
        }

        BitParallelConfig bp_config;
        bp_config = getOptimalBitParallelConfig(bit_columns.size(), max_row);

        std::vector<double> filtration_values;
        filtration_values.reserve(filtration.simplices.size());
        for (const auto& simplex : filtration.simplices) {
            filtration_values.push_back(simplex.filtration_value);
        }

        auto bp_result = reduceMatrixBitParallel(bit_columns, bp_config, filtration_values);

        for (const auto& pair : bp_result.pairs) {
            DistilledPair dp;
            if (pair.death_index >= 0) {
                dp.birth_index = pair.birth_index;
                dp.death_index = pair.death_index;
            } else {
                dp.birth_index = pair.birth_index;
                dp.death_index = -1;
            }
            if (dp.birth_index >= 0 &&
                static_cast<size_t>(dp.birth_index) < filtration.simplices.size()) {
                dp.birth_time = filtration.simplices[dp.birth_index].filtration_value;
                dp.dimension = filtration.simplices[dp.birth_index].dimension;
            }
            if (dp.death_index >= 0 &&
                static_cast<size_t>(dp.death_index) < filtration.simplices.size()) {
                dp.death_time = filtration.simplices[dp.death_index].filtration_value;
            } else {
                dp.death_time = std::numeric_limits<double>::infinity();
            }
            result.pairs.push_back(dp);
        }

        result.used_bit_parallel = true;
    } else {
        std::unordered_map<int, int> pivot_to_column;

        for (size_t col_idx = 0; col_idx < boundary_matrix.size(); ++col_idx) {
            auto& col = boundary_matrix[col_idx];

            std::ranges::sort(col);

            while (!col.empty()) {
                int pivot = col.back();
                auto it = pivot_to_column.find(pivot);
                if (it != pivot_to_column.end()) {
                    std::vector<int> result_col;
                    std::set_symmetric_difference(
                        col.begin(), col.end(),
                        boundary_matrix[it->second].begin(),
                        boundary_matrix[it->second].end(),
                        std::back_inserter(result_col)
                    );
                    col = std::move(result_col);
                } else {
                    pivot_to_column[pivot] = static_cast<int>(col_idx);

                    DistilledPair pair;
                    pair.birth_index = pivot;
                    pair.death_index = static_cast<int>(col_idx);
                    pair.birth_time = filtration.simplices[static_cast<size_t>(pivot)].filtration_value;
                    pair.death_time = filtration.simplices[col_idx].filtration_value;
                    pair.dimension = filtration.simplices[static_cast<size_t>(pivot)].dimension;
                    result.pairs.push_back(pair);
                    break;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(
        end - start).count();

    return result;
}

DistilledVRConfig getOptimalDistilledVRConfig(
    size_t num_points,
    int max_dim,
    double max_radius) {

    DistilledVRConfig config;
    config.max_dim = max_dim;
    config.max_radius = max_radius;

    // Use bit-parallel for large complexes
    config.use_bit_parallel = (num_points > 1000);

    // Parallel construction for large datasets
    config.parallel_construction = (num_points > 10000);

    return config;
}

DistilledSpeedupEstimate estimateDistilledSpeedup(
    size_t num_points,
    int max_dim,
    double max_radius) {

    DistilledSpeedupEstimate estimate{};
    if (num_points == 0 || max_dim < 0 || !(max_radius > 0.0) || !std::isfinite(max_radius)) {
        estimate.memory_reduction = 1.0;
        estimate.cache_efficiency_speedup = 1.0;
        estimate.bit_parallel_speedup = 1.0;
        estimate.total_speedup = 1.0;
        return estimate;
    }

    const double dimension_base = max_dim <= 2 ? 2.0 : (max_dim <= 4 ? 3.0 : 4.0);
    const double radius_density = std::clamp(max_radius / (1.0 + max_radius), 0.0, 1.0);
    const double radius_factor = std::clamp(1.25 - 0.75 * radius_density, 0.5, 1.25);
    const double size_factor = std::clamp(std::log2(static_cast<double>(num_points) + 1.0) / 12.0,
                                          0.75, 1.6);

    estimate.memory_reduction = std::max(1.0, dimension_base * radius_factor * size_factor);
    estimate.cache_efficiency_speedup = 1.0 + 0.35 * std::log2(estimate.memory_reduction);
    estimate.bit_parallel_speedup =
        num_points > 1000 ? std::clamp(std::sqrt(static_cast<double>(num_points) / 1000.0),
                                       1.0, 16.0)
                          : 1.0;
    estimate.total_speedup = estimate.cache_efficiency_speedup * estimate.bit_parallel_speedup;
    return estimate;
}

}  // namespace nerve::persistence::distilled
