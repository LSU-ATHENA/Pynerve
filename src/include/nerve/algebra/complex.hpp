
#pragma once
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"

#include <span>

namespace nerve::algebra
{

class SimplicialComplex
{
public:
    SimplicialComplex() = default;
    void addSimplex(const Simplex &simplex, const core::DeterminismContract &contract = {});
    void addSimplexWithFiltration(const Simplex &simplex, double filtration,
                                  const core::DeterminismContract &contract = {});
    void removeSimplex(const Simplex &simplex, const core::DeterminismContract &contract = {});
    void clear();
    void add_simplex(const Simplex &simplex, const core::DeterminismContract &contract = {})
    {
        addSimplex(simplex, contract);
    }
    void add_simplex_with_filtration(const Simplex &simplex, double filtration,
                                     const core::DeterminismContract &contract = {})
    {
        addSimplexWithFiltration(simplex, filtration, contract);
    }
    void remove_simplex(const Simplex &simplex, const core::DeterminismContract &contract = {})
    {
        removeSimplex(simplex, contract);
    }

    [[nodiscard]] Size size() const noexcept;
    [[nodiscard]] Size numSimplices() const noexcept;
    [[nodiscard]] Dimension maxDimension() const noexcept;
    [[nodiscard]] Vector<Simplex>
    simplicesOfDimension(Dimension dim, const core::DeterminismContract &contract = {}) const;
    [[nodiscard]] Vector<Simplex>
    getSimplices(const core::DeterminismContract &contract = {}) const;
    [[nodiscard]] double getFiltration(const Simplex &simplex) const;
    void setFiltration(const Simplex &simplex, double filtration);
    [[nodiscard]] Vector<std::pair<Simplex, double>>
    getFilteredSimplices(const core::DeterminismContract &contract = {}) const;
    [[nodiscard]] Vector<std::pair<Simplex, double>>
    get_filtered_simplices(const core::DeterminismContract &contract = {}) const
    {
        return getFilteredSimplices(contract);
    }

private:
    void cleanupIsolatedSimplices(const core::DeterminismContract &contract = {});
    Vector<Simplex> simplices_;
    std::unordered_map<Simplex, double, Simplex::Hash> filtration_values_;
};
} // namespace nerve::algebra
