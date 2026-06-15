
#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <limits>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

struct ExactPersistenceResult
{
    std::vector<Pair> pairs;
    std::vector<Size> betti_numbers;
    Size reduction_operations = 0;
};

class IncrementalExactZ2
{
public:
    explicit IncrementalExactZ2(Size max_dim = std::numeric_limits<Size>::max());

    void setMaxDim(Size max_dim);
    void clear();

    // Returns false when the update cannot be applied incrementally
    // (e.g., non-monotonic filtration insert), so callers can request recompute.
    bool addSimplex(const algebra::Simplex &simplex, double filtration);

    void rebuildFromComplex(const algebra::SimplicialComplex &complex);
    ExactPersistenceResult snapshot() const;

private:
    using Column = std::vector<Size>;

    Size max_dim_;
    std::vector<std::pair<algebra::Simplex, double>> simplices_;
    std::unordered_map<algebra::Simplex, Size, algebra::Simplex::Hash> index_of_simplex_;
    std::vector<Column> reduced_columns_;
    std::vector<Index> low_;
    std::vector<Index> low_row_to_col_;
    Size reduction_operations_;
    double last_filtration_;
};

ExactPersistenceResult computeExactPersistenceZ2(const algebra::SimplicialComplex &complex,
                                                 Size max_dim = std::numeric_limits<Size>::max());

ExactPersistenceResult computeExactCohomologyZ2(const algebra::SimplicialComplex &complex,
                                                Size max_dim = std::numeric_limits<Size>::max());

ExactPersistenceResult
computeExactCohomologyZ2(const algebra::SimplicialComplex &complex, Size max_dim,
                         const std::vector<std::vector<int>> &neighbors,
                         const std::unordered_map<std::uint64_t, double> &edge_weights);

inline ExactPersistenceResult
compute_exact_persistence_z2(const algebra::SimplicialComplex &complex,
                             Size max_dim = std::numeric_limits<Size>::max())
{
    return computeExactPersistenceZ2(complex, max_dim);
}

} // namespace nerve::persistence
