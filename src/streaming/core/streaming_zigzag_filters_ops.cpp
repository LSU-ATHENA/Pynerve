
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/streaming/zigzag_filters.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace nerve::streaming::detail
{

namespace
{

using algebra::Simplex;
using algebra::SimplicialComplex;

struct CanonicalFilteredComplex
{
    std::vector<std::pair<Simplex, double>> simplices;
    std::unordered_map<Simplex, double, Simplex::Hash> filtration_by_simplex;
};

// SplitMix64 PRNG Constants
constexpr uint64_t SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EBULL;

uint64_t splitmix64(uint64_t x)
{
    x += SPLITMIX_INCREMENT;
    x = (x ^ (x >> 30U)) * SPLITMIX_MULTIPLIER_1;
    x = (x ^ (x >> 27U)) * SPLITMIX_MULTIPLIER_2;
    return x ^ (x >> 31U);
}

CanonicalFilteredComplex canonicalizeFilteredSimplices(const SimplicialComplex &complex,
                                                       Size max_dimension)
{
    CanonicalFilteredComplex canonical;
    for (const auto &entry : complex.getFilteredSimplices())
    {
        if (entry.first.dimension() > max_dimension)
        {
            continue;
        }
        const auto inserted = canonical.filtration_by_simplex.emplace(entry.first, entry.second);
        if (!inserted.second)
        {
            inserted.first->second = std::min(inserted.first->second, entry.second);
        }
    }

    canonical.simplices.reserve(canonical.filtration_by_simplex.size());
    for (const auto &entry : canonical.filtration_by_simplex)
    {
        canonical.simplices.push_back(entry);
    }

    std::ranges::sort(canonical.simplices, {}, [](const auto &s) {
        return std::tuple(s.first.dimension(), s.second, s.first);
    });
    return canonical;
}

double
faceFiltration(const Simplex &face, double inherited_filtration,
               const std::unordered_map<Simplex, double, Simplex::Hash> &filtration_by_simplex)
{
    const auto it = filtration_by_simplex.find(face);
    if (it == filtration_by_simplex.end())
    {
        return inherited_filtration;
    }
    return std::min(inherited_filtration, it->second);
}

void addSimplexClosure(
    SimplicialComplex &target, const Simplex &simplex, double filtration,
    const std::unordered_map<Simplex, double, Simplex::Hash> &filtration_by_simplex,
    std::unordered_set<Simplex, Simplex::Hash> &inserted)
{
    if (inserted.find(simplex) != inserted.end())
    {
        return;
    }

    for (const auto &face : simplex.faces(core::DeterminismContract{}))
    {
        addSimplexClosure(target, face, faceFiltration(face, filtration, filtration_by_simplex),
                          filtration_by_simplex, inserted);
    }

    target.addSimplexWithFiltration(simplex, filtration);
    inserted.insert(simplex);
}

std::vector<Index> collectVertices(const std::vector<std::pair<Simplex, double>> &simplices)
{
    std::vector<Index> vertices;
    for (const auto &entry : simplices)
    {
        const auto &simplex_vertices = entry.first.vertices();
        vertices.insert(vertices.end(), simplex_vertices.begin(), simplex_vertices.end());
    }

    std::ranges::sort(vertices);
    const auto [first, last] = std::ranges::unique(vertices);
    vertices.erase(first, last);
    return vertices;
}

std::unordered_set<Index> selectLandmarks(const std::vector<Index> &vertices, uint32_t seed)
{
    std::unordered_set<Index> selected;
    if (vertices.empty())
    {
        return selected;
    }

    size_t target_count =
        static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(vertices.size()))));
    target_count = std::max<size_t>(1, target_count);
    target_count = std::min(target_count, vertices.size());

    std::vector<std::pair<uint64_t, Index>> ranked;
    ranked.reserve(vertices.size());
    for (const auto vertex : vertices)
    {
        const uint64_t key =
            splitmix64(static_cast<uint64_t>(vertex) ^ (static_cast<uint64_t>(seed) << 1U));
        ranked.emplace_back(key, vertex);
    }

    std::ranges::sort(ranked, {}, &std::pair<uint64_t, Index>::first);

    for (size_t i = 0; i < target_count; ++i)
    {
        selected.insert(ranked[i].second);
    }
    return selected;
}

bool simplexIsSupportedByLandmarks(const Simplex &simplex,
                                   const std::unordered_set<Index> &landmarks)
{
    for (const auto vertex : simplex.vertices())
    {
        if (landmarks.find(vertex) == landmarks.end())
        {
            return false;
        }
    }
    return true;
}

std::unordered_set<Simplex, Simplex::Hash>
selectSparseHighDimSimplices(const std::vector<std::pair<Simplex, double>> &simplices,
                             uint32_t seed)
{
    std::vector<std::pair<std::pair<double, uint64_t>, Simplex>> ranked;
    for (const auto &entry : simplices)
    {
        if (entry.first.dimension() < 2)
        {
            continue;
        }
        const uint64_t simplex_hash = static_cast<uint64_t>(Simplex::Hash{}(entry.first));
        ranked.push_back(
            {{entry.second, splitmix64(simplex_hash ^ (static_cast<uint64_t>(seed) << 1U))},
             entry.first});
    }

    std::ranges::sort(ranked, {}, [](const auto &r) {
        return std::tuple(r.first.first, r.first.second, r.second);
    });

    std::unordered_set<Simplex, Simplex::Hash> selected;
    const size_t keep_count = (ranked.size() + 1) / 2;
    for (size_t i = 0; i < keep_count; ++i)
    {
        selected.insert(ranked[i].second);
    }
    return selected;
}

SimplicialComplex buildFilteredSubcomplex(const CanonicalFilteredComplex &canonical,
                                          const std::function<bool(const Simplex &)> &keep)
{
    SimplicialComplex reduced;
    std::unordered_set<Simplex, Simplex::Hash> inserted;
    for (const auto &entry : canonical.simplices)
    {
        if (!keep(entry.first))
        {
            continue;
        }
        addSimplexClosure(reduced, entry.first, entry.second, canonical.filtration_by_simplex,
                          inserted);
    }
    return reduced;
}

} // namespace

std::vector<Pair> computeWitnessModePairs(const SimplicialComplex &complex, Size max_dimension,
                                          uint32_t deterministic_seed)
{
    const auto canonical = canonicalizeFilteredSimplices(complex, max_dimension);
    if (canonical.simplices.empty())
    {
        return {};
    }

    const auto vertices = collectVertices(canonical.simplices);
    const auto landmarks = selectLandmarks(vertices, deterministic_seed);
    auto witness_complex = buildFilteredSubcomplex(canonical, [&landmarks](const Simplex &simplex) {
        return simplexIsSupportedByLandmarks(simplex, landmarks);
    });

    const auto exact = persistence::computeExactPersistenceZ2(witness_complex, max_dimension);
    return exact.pairs;
}

std::vector<Pair> computeSparseModePairs(const SimplicialComplex &complex, Size max_dimension,
                                         uint32_t deterministic_seed)
{
    const auto canonical = canonicalizeFilteredSimplices(complex, max_dimension);
    if (canonical.simplices.empty())
    {
        return {};
    }

    const auto selected_high =
        selectSparseHighDimSimplices(canonical.simplices, deterministic_seed);
    auto sparse_complex =
        buildFilteredSubcomplex(canonical, [&selected_high](const Simplex &simplex) {
            if (simplex.dimension() <= 1)
            {
                return true;
            }
            return selected_high.find(simplex) != selected_high.end();
        });

    const auto exact = persistence::computeExactPersistenceZ2(sparse_complex, max_dimension);
    return exact.pairs;
}

} // namespace nerve::streaming::detail
