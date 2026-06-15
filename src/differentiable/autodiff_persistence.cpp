#include "nerve/differentiable/autodiff_persistence.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef TOPO_ENABLE_ADVANCED_DIFFERENTIABLE
namespace nerve::differentiable
{
namespace
{

template <typename T>
struct FiltrationEntry
{
    algebra::Simplex simplex;
    T value = T(0);
};

[[nodiscard]] bool lexicographicSimplexLess(const algebra::Simplex &lhs,
                                            const algebra::Simplex &rhs)
{
    return std::lexicographical_compare(lhs.vertices().begin(), lhs.vertices().end(),
                                        rhs.vertices().begin(), rhs.vertices().end());
}

template <typename T>
[[nodiscard]] bool filtrationLess(const FiltrationEntry<T> &lhs, const FiltrationEntry<T> &rhs)
{
    if (lhs.value != rhs.value)
    {
        return lhs.value < rhs.value;
    }
    if (lhs.simplex.dimension() != rhs.simplex.dimension())
    {
        return lhs.simplex.dimension() < rhs.simplex.dimension();
    }
    return lexicographicSimplexLess(lhs.simplex, rhs.simplex);
}

[[nodiscard]] std::vector<size_t> z2SymmetricDifference(const std::vector<size_t> &lhs,
                                                        const std::vector<size_t> &rhs)
{
    std::vector<size_t> result;
    result.reserve(lhs.size() + rhs.size());
    std::set_symmetric_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                                  std::back_inserter(result));
    return result;
}

template <typename T>
[[nodiscard]] std::vector<FiltrationEntry<T>>
buildClosedFiltration(const std::vector<algebra::Simplex> &complex)
{
    std::vector<algebra::Simplex> closure;
    std::unordered_set<algebra::Simplex, algebra::Simplex::Hash> seen;

    auto add_simplex = [&](const algebra::Simplex &simplex) {
        if (simplex.numVertices() == 0)
        {
            return;
        }
        if (seen.insert(simplex).second)
        {
            closure.push_back(simplex);
        }
    };

    for (const auto &simplex : complex)
    {
        add_simplex(simplex);
    }
    for (size_t i = 0; i < closure.size(); ++i)
    {
        for (const auto &face : closure[i].faces(core::DeterminismContract{}))
        {
            add_simplex(face);
        }
    }

    std::vector<FiltrationEntry<T>> filtration;
    filtration.reserve(closure.size());
    for (const auto &simplex : closure)
    {
        filtration.push_back({simplex, static_cast<T>(simplex.dimension())});
    }
    std::sort(filtration.begin(), filtration.end(), filtrationLess<T>);
    return filtration;
}

template <typename T>
[[nodiscard]] std::vector<std::vector<size_t>>
buildBoundaryColumns(const std::vector<FiltrationEntry<T>> &filtration)
{
    std::unordered_map<algebra::Simplex, size_t, algebra::Simplex::Hash> simplex_to_index;
    simplex_to_index.reserve(filtration.size());
    for (size_t i = 0; i < filtration.size(); ++i)
    {
        simplex_to_index.emplace(filtration[i].simplex, i);
    }

    std::vector<std::vector<size_t>> columns(filtration.size());
    for (size_t col = 0; col < filtration.size(); ++col)
    {
        const auto &simplex = filtration[col].simplex;
        if (simplex.dimension() == 0)
        {
            continue;
        }
        auto faces = simplex.faces(core::DeterminismContract{});
        auto &boundary = columns[col];
        boundary.reserve(faces.size());
        for (const auto &face : faces)
        {
            const auto it = simplex_to_index.find(face);
            if (it == simplex_to_index.end())
            {
                throw std::logic_error("closed filtration is missing a simplex face");
            }
            boundary.push_back(it->second);
        }
        std::sort(boundary.begin(), boundary.end());
        boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());
    }
    return columns;
}

template <typename T>
[[nodiscard]] AutodiffPersistenceDiagram<T>
reduceFiltration(const std::vector<FiltrationEntry<T>> &filtration, size_t max_iterations)
{
    AutodiffPersistenceDiagram<T> diagram;
    if (filtration.empty())
    {
        return diagram;
    }

    auto reduced_columns = buildBoundaryColumns(filtration);
    std::unordered_map<size_t, size_t> pivot_to_column;
    std::vector<char> zero_column(filtration.size(), 0);
    std::vector<char> paired_as_birth(filtration.size(), 0);

    /*
     * Standard persistent-homology reduction over Z2.
     * Columns are processed in filtration order. If a column's lowest one
     * collides with an earlier pivot, the two columns are xor-added; a
     * surviving pivot pairs the row simplex as birth with the current simplex
     * as death, while unpaired zero columns represent infinite classes.
     */
    for (size_t col = 0; col < reduced_columns.size(); ++col)
    {
        size_t iterations = 0;
        auto &column = reduced_columns[col];
        while (!column.empty())
        {
            const size_t pivot = column.back();
            const auto pivot_it = pivot_to_column.find(pivot);
            if (pivot_it == pivot_to_column.end())
            {
                break;
            }
            column = z2SymmetricDifference(column, reduced_columns[pivot_it->second]);
            ++iterations;
            if (max_iterations > 0 && iterations > max_iterations)
            {
                throw std::runtime_error(
                    "differentiable persistence reduction iteration budget exceeded");
            }
        }

        if (column.empty())
        {
            zero_column[col] = 1;
            continue;
        }

        const size_t birth = column.back();
        pivot_to_column[birth] = col;
        paired_as_birth[birth] = 1;

        const T birth_time = filtration[birth].value;
        const T death_time = filtration[col].value;
        diagram.addPair({birth_time, death_time, death_time - birth_time,
                         static_cast<T>(filtration[birth].simplex.dimension())});
    }

    for (size_t col = 0; col < filtration.size(); ++col)
    {
        if (!zero_column[col] || paired_as_birth[col])
        {
            continue;
        }
        diagram.addPair({filtration[col].value, std::numeric_limits<T>::infinity(),
                         std::numeric_limits<T>::infinity(),
                         static_cast<T>(filtration[col].simplex.dimension())});
    }

    return diagram;
}

} // namespace

template <typename T>
DifferentiablePersistence<T>::DifferentiablePersistence(const ComputationConfig &config)
    : config_(config)
{}

template <typename T>
void DifferentiablePersistence<T>::setConfig(const ComputationConfig &config)
{
    config_ = config;
}

template <typename T>
typename DifferentiablePersistence<T>::ComputationConfig
DifferentiablePersistence<T>::getConfig() const
{
    return config_;
}

template <typename T>
typename DifferentiablePersistence<T>::DiagramType
DifferentiablePersistence<T>::compute(const std::vector<algebra::Simplex> &complex,
                                      const core::DeterminismContract &contract)
{
    if (!core::DeterminismEnforcer::canSatisfyContract(contract) &&
        contract.fail_on_non_deterministic)
    {
        throw std::runtime_error(
            "cannot satisfy determinism contract for differentiable persistence");
    }

    last_diagram_ = computeForwardPass(complex);
    if (config_.enableGradients)
    {
        last_diagram_.enableGradients();
        last_diagram_.resetGradients();
    }

    intermediate_values_.clear();
    if (config_.track_intermediate_values)
    {
        for (const auto &pair : last_diagram_.getPairs())
        {
            intermediate_values_.push_back(pair.persistence);
        }
    }

    return last_diagram_;
}

template <typename T>
typename DifferentiablePersistence<T>::DiagramType
DifferentiablePersistence<T>::computePersistenceWithGradients(
    const std::vector<algebra::Simplex> &complex,
    std::function<void(const DiagramType &)> gradient_callback,
    const core::DeterminismContract &contract)
{
    auto diagram = compute(complex, contract);
    if (!config_.enableGradients)
    {
        return diagram;
    }

    std::vector<ScalarType> unit_gradients(diagram.getPairs().size(), ScalarType(T(1)));
    computeBackwardPass(unit_gradients);

    if (gradient_callback)
    {
        gradient_callback(last_diagram_);
    }
    return last_diagram_;
}

template <typename T>
void DifferentiablePersistence<T>::computeGradients(const std::vector<algebra::Simplex> &complex,
                                                    const std::vector<ScalarType> &loss_gradients,
                                                    const core::DeterminismContract &contract)
{
    if (!core::DeterminismEnforcer::canSatisfyContract(contract) &&
        contract.fail_on_non_deterministic)
    {
        throw std::runtime_error(
            "cannot satisfy determinism contract for differentiable persistence gradients");
    }
    if (last_diagram_.getPairs().empty() && !complex.empty())
    {
        last_diagram_ = computeForwardPass(complex);
    }
    if (loss_gradients.size() != last_diagram_.getPairs().size())
    {
        throw std::invalid_argument("loss gradient count must match persistence pair count");
    }
    computeBackwardPass(loss_gradients);
}

template <typename T>
bool DifferentiablePersistence<T>::validateGradients(const std::vector<algebra::Simplex> &complex,
                                                     T finite_difference_tolerance) const
{
    if (!config_.enableGradients || !config_.validateGradients)
    {
        return true;
    }

    const auto fd_gradients = computeFiniteDifferenceGradients(complex);
    const auto &pairs = last_diagram_.getPairs();
    if (fd_gradients.size() != pairs.size())
    {
        return false;
    }

    for (size_t i = 0; i < fd_gradients.size(); ++i)
    {
        const T diff = std::abs(fd_gradients[i].value() - pairs[i].persistence.grad());
        if (diff > finite_difference_tolerance)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
const typename DifferentiablePersistence<T>::DiagramType &
DifferentiablePersistence<T>::getLastDiagram() const
{
    return last_diagram_;
}

template <typename T>
const std::vector<typename DifferentiablePersistence<T>::ScalarType> &
DifferentiablePersistence<T>::getIntermediateValues() const
{
    return intermediate_values_;
}

template <typename T>
typename DifferentiablePersistence<T>::DiagramType
DifferentiablePersistence<T>::computeForwardPass(const std::vector<algebra::Simplex> &complex) const
{
    const auto filtration = buildClosedFiltration<T>(complex);
    return reduceFiltration<T>(filtration, config_.max_iterations);
}

template <typename T>
void DifferentiablePersistence<T>::computeBackwardPass(
    const std::vector<ScalarType> &loss_gradients)
{
    auto &pairs = last_diagram_.getPairs();
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        const T upstream = loss_gradients[i].value();
        const bool finite_pair = std::isfinite(pairs[i].birth.value()) &&
                                 std::isfinite(pairs[i].death.value()) &&
                                 std::isfinite(pairs[i].persistence.value());
        if (!finite_pair)
        {
            pairs[i].birth.setGradient(T(0));
            pairs[i].death.setGradient(T(0));
            pairs[i].persistence.setGradient(T(0));
            pairs[i].dimension.setGradient(T(0));
            continue;
        }

        pairs[i].birth.setGradient(-upstream);
        pairs[i].death.setGradient(upstream);
        pairs[i].persistence.setGradient(upstream);
        pairs[i].dimension.setGradient(T(0));
    }
}

template <typename T>
std::vector<typename DifferentiablePersistence<T>::ScalarType>
DifferentiablePersistence<T>::computeFiniteDifferenceGradients(
    const std::vector<algebra::Simplex> &complex, T epsilon) const
{
    if (epsilon <= T(0))
    {
        throw std::invalid_argument("finite-difference epsilon must be positive");
    }

    const auto reference = computeForwardPass(complex);
    const auto &active_pairs = last_diagram_.getPairs();
    if (active_pairs.size() != reference.getPairs().size())
    {
        return {};
    }

    std::vector<ScalarType> gradients;
    gradients.reserve(reference.getPairs().size());

    for (size_t i = 0; i < reference.getPairs().size(); ++i)
    {
        const auto &pair = reference.getPairs()[i];
        if (!std::isfinite(pair.death.value()) || !std::isfinite(pair.persistence.value()))
        {
            gradients.emplace_back(T(0));
            continue;
        }
        const T upstream = active_pairs[i].persistence.requiresGradient()
                               ? active_pairs[i].persistence.grad()
                               : T(1);
        const T plus = upstream * ((pair.death.value() + epsilon) - pair.birth.value());
        const T minus = upstream * ((pair.death.value() - epsilon) - pair.birth.value());
        gradients.emplace_back((plus - minus) / (T(2) * epsilon));
    }
    return gradients;
}

template <typename T>
bool DifferentiablePersistence<T>::isAutodiffSafeFunction(const std::function<void()> &func) const
{
    return autodiff_safe::isSafe(func);
}

template class DifferentiablePersistence<double>;
template class DifferentiablePersistence<float>;

} // namespace nerve::differentiable
#endif
