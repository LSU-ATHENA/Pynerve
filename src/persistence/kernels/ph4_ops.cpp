
#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/kernels/ph4_ops.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <tuple>
#include <unordered_map>

namespace nerve::persistence
{
namespace
{

using FilteredSimplex = std::pair<algebra::Simplex, double>;

constexpr Size kPh4MaxReportedDimension = 4;
constexpr std::size_t kBytesPerMiB = 1024ULL * 1024ULL;

struct PreparedComplex
{
    algebra::SimplicialComplex complex;
    std::vector<FilteredSimplex> ordered_simplices;
    std::unordered_map<algebra::Simplex, Index, algebra::Simplex::Hash> input_index_by_simplex;
    std::size_t estimated_memory_bytes = 0;
};

bool simplexFiltrationLess(const FilteredSimplex &lhs, const FilteredSimplex &rhs)
{
    if (lhs.second != rhs.second)
    {
        return lhs.second < rhs.second;
    }
    if (lhs.first.dimension() != rhs.first.dimension())
    {
        return lhs.first.dimension() < rhs.first.dimension();
    }
    return lhs.first < rhs.first;
}

std::size_t bytesToMiB(std::size_t bytes)
{
    if (bytes == 0)
    {
        return 0;
    }
    return (bytes + kBytesPerMiB - 1) / kBytesPerMiB;
}

algebra::Simplex toAlgebraSimplex(const Simplex &simplex)
{
    return algebra::Simplex(simplex.vertices);
}

void insertClosure(
    const algebra::Simplex &simplex, double filtration,
    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> &filtrations)
{
    if (simplex.numVertices() == 0 || !std::isfinite(filtration))
    {
        return;
    }

    const auto [it, inserted] = filtrations.emplace(simplex, filtration);
    if (!inserted)
    {
        if (filtration >= it->second)
        {
            return;
        }
        it->second = filtration;
    }

    for (const auto &face : simplex.faces({}))
    {
        insertClosure(face, filtration, filtrations);
    }
}

std::size_t estimateReductionMemoryBytes(const std::vector<FilteredSimplex> &simplices)
{
    std::size_t bytes = simplices.size() * (sizeof(FilteredSimplex) + sizeof(Pair));
    for (const auto &[simplex, _filtration] : simplices)
    {
        bytes += simplex.numVertices() * sizeof(Index);
        if (simplex.dimension() > 0)
        {
            bytes += simplex.numVertices() * (sizeof(Size) + sizeof(Index) + sizeof(double));
        }
    }
    return bytes;
}

PreparedComplex prepareComplex(const std::vector<Simplex> &input,
                               const ::nerve::core::DeterminismContract &contract)
{
    PreparedComplex prepared;
    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> filtrations;
    filtrations.reserve(input.size() * 4);
    prepared.input_index_by_simplex.reserve(input.size());

    for (Index i = 0; i < static_cast<Index>(input.size()); ++i)
    {
        const Simplex &simplex = input[static_cast<std::size_t>(i)];
        if (simplex.dimension() < 0 || simplex.vertices.empty())
        {
            continue;
        }
        if (!std::isfinite(simplex.value))
        {
            auto error = errors::InvalidArgumentError(__FILE__, __LINE__, __func__);
            error.setArgumentName("complex")
                .setArgumentValue(static_cast<int>(i))
                .setReason("PH4 simplex filtration must be finite");
            throw error;
        }

        const algebra::Simplex algebra_simplex = toAlgebraSimplex(simplex);
        insertClosure(algebra_simplex, simplex.value, filtrations);
        const auto [index_it, inserted] =
            prepared.input_index_by_simplex.emplace(algebra_simplex, i);
        if (!inserted && simplex.value < input[static_cast<std::size_t>(index_it->second)].value)
        {
            index_it->second = i;
        }
    }

    prepared.ordered_simplices.reserve(filtrations.size());
    for (const auto &[simplex, filtration] : filtrations)
    {
        prepared.ordered_simplices.emplace_back(simplex, filtration);
    }
    std::ranges::sort(prepared.ordered_simplices, simplexFiltrationLess);

    for (const auto &[simplex, filtration] : prepared.ordered_simplices)
    {
        prepared.complex.addSimplexWithFiltration(simplex, filtration, contract);
    }

    prepared.estimated_memory_bytes = estimateReductionMemoryBytes(prepared.ordered_simplices);
    return prepared;
}

Diagram diagramFromExactResult(const ExactPersistenceResult &exact)
{
    Diagram diagram;
    for (const auto &pair : exact.pairs)
    {
        diagram.addPair(pair);
    }
    auto &pairs = diagram.getPairs();
    std::ranges::sort(pairs, [](const Pair &lhs, const Pair &rhs) {
        return std::tie(lhs.dimension, lhs.birth, lhs.death, lhs.birth_index, lhs.death_index) <
               std::tie(rhs.dimension, rhs.birth, rhs.death, rhs.birth_index, rhs.death_index);
    });
    return diagram;
}

} // namespace

PH4::PH4() = default;

PH4::PH4(PH4Algorithm algorithm, ComputeMode mode, std::size_t max_memory_mb)
    : algorithm_(algorithm)
    , compute_mode_(mode)
    , max_memory_mb_(max_memory_mb)
    , witness_strategy_(WitnessStrategy::RANDOM_SAMPLING)
    , computation_time_ms_(0)
    , memory_used_mb_(0)
    , budget_exceeded_(false)
{}

std::pair<Diagram, StabilityCertificate>
PH4::computePersistenceWithCertificate(const std::vector<Simplex> &complex,
                                       const ::nerve::core::DeterminismContract &contract)
{
    const auto start = std::chrono::steady_clock::now();
    Diagram diagram = compute(complex, contract);
    const auto end = std::chrono::steady_clock::now();
    computation_time_ms_ = static_cast<std::size_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    return {diagram, computeStabilityCertificate(diagram)};
}

std::pair<CompactSummary, StabilityCertificate>
PH4::computePersistenceBudgeted(const std::vector<Simplex> &complex, std::size_t max_memory_mb,
                                const ::nerve::core::DeterminismContract &contract)
{
    const std::size_t original_budget = max_memory_mb_;
    max_memory_mb_ = max_memory_mb;
    try
    {
        const auto start = std::chrono::steady_clock::now();
        Diagram diagram = compute(complex, contract);
        CompactSummary summary = computeCompactSummary(diagram, max_memory_mb);
        StabilityCertificate certificate = computeStabilityCertificate(diagram);
        const auto end = std::chrono::steady_clock::now();
        computation_time_ms_ = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        max_memory_mb_ = original_budget;
        return {summary, certificate};
    }
    catch (...)
    {
        max_memory_mb_ = original_budget;
        throw;
    }
}

Diagram PH4::compute(const std::vector<Simplex> &complex,
                     const ::nerve::core::DeterminismContract &contract)
{
    switch (compute_mode_)
    {
        case ComputeMode::APPROXIMATE:
            return computeApproximatePersistence(complex, contract);
        case ComputeMode::BUDGETED:
            return computeSparsePersistence(complex, contract);
        case ComputeMode::EXACT:
        default:
            break;
    }

    switch (algorithm_)
    {
        case PH4Algorithm::EXACT_COMPUTE:
            return computeExactPersistence(complex, contract);
        case PH4Algorithm::BUDGETED_COMPUTE:
        case PH4Algorithm::SPARSE_MATRIX:
            return computeSparsePersistence(complex, contract);
        case PH4Algorithm::WITNESS_SAMPLING:
            return computeApproximatePersistence(complex, contract);
    }
    return computeExactPersistence(complex, contract);
}

Diagram PH4::computePersistenceApproximate(const std::vector<Simplex> &complex,
                                           const ::nerve::core::DeterminismContract &contract)
{
    return computeApproximatePersistence(complex, contract);
}

std::pair<std::vector<Index>, StabilityCertificate>
PH4::sampleWitnessesWithCertificate(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                                    WitnessStrategy strategy,
                                    const ::nerve::core::DeterminismContract &contract)
{
    auto witnesses = sampleWitnesses(complex, num_witnesses, strategy, contract);
    StabilityCertificate certificate(static_cast<double>(witnesses.size()), 1e-10, false);
    return {witnesses, certificate};
}

std::vector<Index> PH4::sampleWitnesses(const std::vector<Simplex> &complex,
                                        std::size_t num_witnesses, WitnessStrategy strategy,
                                        const ::nerve::core::DeterminismContract &contract)
{
    switch (strategy)
    {
        case WitnessStrategy::MAX_PERSISTENCE:
            return sampleMaxPersistenceWitnesses(complex, num_witnesses, contract);
        case WitnessStrategy::RANDOM_SAMPLING:
        case WitnessStrategy::LANDMARK_SAMPLING:
        case WitnessStrategy::COFACTOR_PERSISTENCE:
        default:
            return sampleLandmarkWitnesses(complex, num_witnesses, contract);
    }
}

void PH4::setAlgorithm(PH4Algorithm algorithm)
{
    algorithm_ = algorithm;
}
void PH4::setComputeMode(ComputeMode mode)
{
    compute_mode_ = mode;
}
void PH4::setMaxMemory(std::size_t max_memory_mb)
{
    max_memory_mb_ = max_memory_mb;
}
void PH4::setWitnessStrategy(WitnessStrategy strategy)
{
    witness_strategy_ = strategy;
}
std::size_t PH4::getComputationTimeMs() const
{
    return computation_time_ms_;
}
std::size_t PH4::getMemoryUsedMb() const
{
    return memory_used_mb_;
}
bool PH4::wasBudgetExceeded() const
{
    return budget_exceeded_;
}

Diagram PH4::computeExactPersistence(const std::vector<Simplex> &complex,
                                     const ::nerve::core::DeterminismContract &contract)
{
    PreparedComplex prepared = prepareComplex(complex, contract);
    const std::size_t required_memory_mb = bytesToMiB(prepared.estimated_memory_bytes);
    checkMemoryBudget(required_memory_mb);
    memory_used_mb_ = required_memory_mb;
    const auto exact = computeExactPersistenceZ2(prepared.complex, kPh4MaxReportedDimension);
    return diagramFromExactResult(exact);
}

Diagram PH4::computeSparsePersistence(const std::vector<Simplex> &complex,
                                      const ::nerve::core::DeterminismContract &contract)
{
    return computeExactPersistence(complex, contract);
}

Diagram PH4::computeApproximatePersistence(const std::vector<Simplex> &complex,
                                           const ::nerve::core::DeterminismContract &contract)
{
    /*
     * The PH4 vector API carries simplex filtrations but no metric landmark
     * structure. Approximate mode therefore delegates to the same exact Z2
     * reducer instead of manufacturing lifetimes from a sampled subset.
     */
    return computeExactPersistence(complex, contract);
}

std::vector<Index> PH4::sampleLandmarkWitnesses(const std::vector<Simplex> &complex,
                                                std::size_t num_witnesses,
                                                const ::nerve::core::DeterminismContract &contract)
{
    std::vector<Index> indices;
    if (complex.empty() || num_witnesses == 0)
    {
        return indices;
    }
    std::vector<Index> candidates(complex.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::mt19937_64 rng(0xC0FFEEULL);
    if (contract.rng_seed_provided)
    {
        uint64_t seed = 0;
        for (size_t i = 0; i < 8; ++i)
        {
            seed |= static_cast<uint64_t>(contract.rng_seed[i]) << (8 * i);
        }
        rng.seed(seed);
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const size_t keep = std::min(num_witnesses, candidates.size());
    indices.insert(indices.end(), candidates.begin(),
                   candidates.begin() + static_cast<std::ptrdiff_t>(keep));
    return indices;
}

std::vector<Index>
PH4::sampleMaxPersistenceWitnesses(const std::vector<Simplex> &complex, std::size_t num_witnesses,
                                   const ::nerve::core::DeterminismContract &contract)
{
    if (complex.empty() || num_witnesses == 0)
    {
        return {};
    }

    const PreparedComplex prepared = prepareComplex(complex, contract);
    const auto exact = computeExactPersistenceZ2(prepared.complex, kPh4MaxReportedDimension);

    std::unordered_map<Index, double> best_lifetime_by_input;
    for (const auto &pair : exact.pairs)
    {
        const double lifetime =
            pair.isInfinite() ? std::numeric_limits<double>::infinity() : pair.lifetime();
        const Index exact_indices[] = {pair.birth_index, pair.death_index};
        for (Index exact_index : exact_indices)
        {
            if (exact_index < 0 ||
                static_cast<std::size_t>(exact_index) >= prepared.ordered_simplices.size())
            {
                continue;
            }
            const auto &simplex =
                prepared.ordered_simplices[static_cast<std::size_t>(exact_index)].first;
            const auto input_it = prepared.input_index_by_simplex.find(simplex);
            if (input_it == prepared.input_index_by_simplex.end())
            {
                continue;
            }
            auto [score_it, inserted] = best_lifetime_by_input.emplace(input_it->second, lifetime);
            if (!inserted && lifetime > score_it->second)
            {
                score_it->second = lifetime;
            }
        }
    }

    if (best_lifetime_by_input.empty())
    {
        return sampleLandmarkWitnesses(complex, num_witnesses, contract);
    }

    std::vector<std::pair<double, Index>> ranked;
    ranked.reserve(best_lifetime_by_input.size());
    for (const auto &[index, lifetime] : best_lifetime_by_input)
    {
        ranked.emplace_back(lifetime, index);
    }
    std::ranges::sort(ranked, [](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
        {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });

    std::vector<Index> indices;
    const size_t keep = std::min(num_witnesses, ranked.size());
    indices.reserve(keep);
    for (size_t i = 0; i < keep; ++i)
    {
        indices.push_back(ranked[i].second);
    }
    return indices;
}

void PH4::checkMemoryBudget(std::size_t required_memory_mb)
{
    if (required_memory_mb > max_memory_mb_)
    {
        budget_exceeded_ = true;
        auto error = errors::BudgetExceededError(__FILE__, __LINE__, __func__);
        error.setBudgetType("PH4 reduction memory")
            .setMemoryBudgetMb(static_cast<double>(max_memory_mb_))
            .setMemoryUsedMb(static_cast<double>(required_memory_mb));
        throw error;
    }
    budget_exceeded_ = false;
}

StabilityCertificate PH4::computeStabilityCertificate(const Diagram &diagram)
{
    double max_lifetime = 0.0;
    for (const auto &pair : diagram.getPairs())
    {
        max_lifetime = std::max(max_lifetime, pair.lifetime());
    }
    return StabilityCertificate(max_lifetime, 0.0, true);
}

CompactSummary PH4::computeCompactSummary(const Diagram &diagram, std::size_t max_memory_mb)
{
    CompactSummary summary;
    for (const auto &pair : diagram.getPairs())
    {
        summary.addPair(pair);
    }
    const std::size_t summary_mb = bytesToMiB(summary.getTotalPairs() * sizeof(Pair));
    memory_used_mb_ = std::min(max_memory_mb, std::max(memory_used_mb_, summary_mb));
    return summary;
}

} // namespace nerve::persistence
