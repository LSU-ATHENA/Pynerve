
#include "nerve/algebra/complex.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace nerve::algebra
{

namespace
{

void validateSimplex(const Simplex &simplex)
{
    if (simplex.numVertices() == 0)
    {
        throw std::invalid_argument("simplex must contain at least one vertex");
    }
    for (const Index vertex : simplex.vertices())
    {
        if (vertex < 0)
        {
            throw std::invalid_argument("simplex vertices must be non-negative");
        }
    }
}

void validateFiltration(double filtration)
{
    if (!std::isfinite(filtration))
    {
        throw std::invalid_argument("simplex filtration must be finite");
    }
}

void enforceDeterminismContract(const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        throw std::runtime_error("cannot satisfy simplicial complex determinism contract");
    }
}

} // namespace

void SimplicialComplex::addSimplex(const Simplex &simplex,
                                   const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);
    validateSimplex(simplex);
    const auto [it, inserted] =
        filtration_values_.try_emplace(simplex, static_cast<double>(simplex.dimension()));
    if (inserted)
    {
        simplices_.emplace_back(simplex);
    }
}

void SimplicialComplex::addSimplexWithFiltration(const Simplex &simplex, double filtration,
                                                 const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);
    validateSimplex(simplex);
    validateFiltration(filtration);
    const auto [it, inserted] = filtration_values_.try_emplace(simplex, filtration);
    if (inserted)
    {
        simplices_.emplace_back(simplex);
    }
    else
    {
        it->second = filtration;
    }
}
void SimplicialComplex::removeSimplex(const Simplex &simplex,
                                      const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);
    validateSimplex(simplex);
    auto it = std::ranges::find(simplices_, simplex);
    if (it == simplices_.end())
    {
        return;
    }

    bool has_dependent_simplices = false;
    for (const auto &other : simplices_)
    {
        if (other.dimension() > simplex.dimension() && simplex.isFaceOf(other))
        {
            has_dependent_simplices = true;
            break;
        }
    }

    if (has_dependent_simplices)
    {
        return;
    }

    simplices_.erase(it);
    filtration_values_.erase(simplex);
}
void SimplicialComplex::cleanupIsolatedSimplices(const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);
    if (simplices_.size() < 2)
        return;
    std::vector<Simplex> to_remove;
    for (const auto &simplex : simplices_)
    {
        bool has_connection = false;
        for (const auto &other_simplex : simplices_)
        {
            if (simplex != other_simplex)
            {
                const auto &vertices1 = simplex.vertices();
                const auto &vertices2 = other_simplex.vertices();
                for (const auto &vertex : vertices1)
                {
                    if (std::ranges::find(vertices2, vertex) != vertices2.end())
                    {
                        has_connection = true;
                        break;
                    }
                }
                if (has_connection)
                    break;
            }
        }
        if (!has_connection)
        {
            to_remove.push_back(simplex);
        }
    }
    for (const auto &simplex : to_remove)
    {
        auto it = std::ranges::find(simplices_, simplex);
        if (it != simplices_.end())
        {
            simplices_.erase(it);
            filtration_values_.erase(simplex);
        }
    }
}
void SimplicialComplex::clear()
{
    simplices_.clear();
    filtration_values_.clear();
}
Size SimplicialComplex::size() const noexcept
{
    return simplices_.size();
}
Size SimplicialComplex::numSimplices() const noexcept
{
    return simplices_.size();
}
Vector<Simplex> SimplicialComplex::getSimplices(const core::DeterminismContract &contract) const
{
    enforceDeterminismContract(contract);
    return simplices_;
}
Dimension SimplicialComplex::maxDimension() const noexcept
{
    Dimension max_dim = -1;
    for (const auto &simplex : simplices_)
    {
        max_dim = std::max(max_dim, static_cast<Dimension>(simplex.dimension()));
    }
    return max_dim;
}
Vector<Simplex>
SimplicialComplex::simplicesOfDimension(Dimension dim,
                                        const core::DeterminismContract &contract) const
{
    enforceDeterminismContract(contract);
    Vector<Simplex> result;
    for (const auto &simplex : simplices_)
    {
        if (simplex.dimension() == static_cast<Size>(dim))
        {
            result.push_back(simplex);
        }
    }
    return result;
}
double SimplicialComplex::getFiltration(const Simplex &simplex) const
{
    auto it = filtration_values_.find(simplex);
    if (it != filtration_values_.end())
    {
        return it->second;
    }
    return static_cast<double>(simplex.dimension());
}
void SimplicialComplex::setFiltration(const Simplex &simplex, double filtration)
{
    validateSimplex(simplex);
    validateFiltration(filtration);
    const auto [it, inserted] = filtration_values_.try_emplace(simplex, filtration);
    if (inserted)
    {
        simplices_.emplace_back(simplex);
    }
    else
    {
        it->second = filtration;
    }
}
Vector<std::pair<Simplex, double>>
SimplicialComplex::getFilteredSimplices(const core::DeterminismContract &contract) const
{
    enforceDeterminismContract(contract);
    Vector<std::pair<Simplex, double>> result;
    for (const auto &simplex : simplices_)
    {
        double filtration = getFiltration(simplex);
        result.emplace_back(simplex, filtration);
    }
    return result;
}
} // namespace nerve::algebra
