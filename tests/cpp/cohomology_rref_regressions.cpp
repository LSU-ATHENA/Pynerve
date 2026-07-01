
#include "nerve/persistence/cohomology/cohomology_rref.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::persistence::CohomologyRREF;
using nerve::persistence::detail::computeRref;
using nerve::persistence::detail::matrixRank;
using nerve::persistence::detail::multiplyMatrixVector;
using nerve::persistence::detail::nullspaceBasis;
using nerve::persistence::detail::RrefResult;
using nerve::persistence::detail::submatrix;

bool check_rref_computation_small()
{
    std::vector<std::vector<double>> m = {{1, 0, 1}, {0, 1, 1}, {0, 0, 0}};
    auto res = computeRref(m);
    if (res.matrix.empty())
        return false;
    if (res.pivot_columns.empty())
        return false;
    return true;
}

bool check_matrix_dimensions_valid()
{
    std::vector<std::vector<double>> m = {{1, 0}, {0, 1}, {1, 1}};
    auto res = computeRref(m);
    if (res.matrix.size() != 3)
        return false;
    for (const auto &row : res.matrix)
    {
        if (row.size() != 2)
            return false;
    }
    return true;
}

bool check_rank_known_case()
{
    std::vector<std::vector<double>> id = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    if (matrixRank(id) != 3)
        return false;

    std::vector<std::vector<double>> rank1 = {{1, 1}, {0, 0}, {0, 0}};
    if (matrixRank(rank1) != 1)
        return false;

    std::vector<std::vector<double>> zero = {{0, 0}, {0, 0}};
    if (matrixRank(zero) != 0)
        return false;
    return true;
}

bool check_rref_invariants()
{
    std::vector<std::vector<double>> m = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    auto res = computeRref(m);
    for (size_t i = 0; i < res.pivot_columns.size(); ++i)
    {
        Size pc = res.pivot_columns[i];
        if (pc >= res.matrix[0].size())
            return false;
        if (std::abs(res.matrix[i][pc] - 1.0) > 1e-12)
            return false;
        for (size_t j = 0; j < i; ++j)
        {
            if (std::abs(res.matrix[j][pc]) > 1e-12)
                return false;
        }
    }
    return true;
}

bool check_cohomology_rref_class()
{
    CohomologyRREF rref;
    std::vector<std::vector<double>> m = {{1, 1}, {0, 1}};
    auto rref_m = rref.computeRREF(m);
    auto pivots = rref.findPivots(rref_m);
    auto rank = rref.computeRank(rref_m);
    auto ns = rref.computeNullSpace(rref_m);
    if (pivots.empty())
        return false;
    if (rank == 0)
        return false;
    return true;
}

bool check_nullspace_basis()
{
    std::vector<std::vector<double>> m = {{1, 0, 1}, {0, 1, 1}, {0, 0, 0}};
    auto ns = nullspaceBasis(m);
    if (ns.size() != 1)
        return false;
    const auto &v = ns[0];
    if (v.size() != 3)
        return false;
    return true;
}

bool check_submatrix()
{
    std::vector<std::vector<double>> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<nerve::Index> rows = {0, 2};
    std::vector<nerve::Index> cols = {1};
    auto sub = submatrix(m, rows, cols);
    if (sub.size() != 2)
        return false;
    if (sub[0].size() != 1)
        return false;
    return true;
}

bool check_multiply_matrix_vector()
{
    std::vector<std::vector<double>> m = {{1, 0}, {0, 1}};
    std::vector<double> v = {3, 4};
    auto res = multiplyMatrixVector(m, v);
    if (res.size() != 2)
        return false;
    if (std::abs(res[0] - 3.0) > 1e-12)
        return false;
    if (std::abs(res[1] - 4.0) > 1e-12)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_rref_computation_small())
    {
        std::cerr << "FAIL: rref computation small\n";
        return 1;
    }
    if (!check_matrix_dimensions_valid())
    {
        std::cerr << "FAIL: matrix dimensions valid\n";
        return 1;
    }
    if (!check_rank_known_case())
    {
        std::cerr << "FAIL: rank known case\n";
        return 1;
    }
    if (!check_rref_invariants())
    {
        std::cerr << "FAIL: rref invariants\n";
        return 1;
    }
    if (!check_cohomology_rref_class())
    {
        std::cerr << "FAIL: cohomology rref class\n";
        return 1;
    }
    if (!check_nullspace_basis())
    {
        std::cerr << "FAIL: nullspace basis\n";
        return 1;
    }
    if (!check_submatrix())
    {
        std::cerr << "FAIL: submatrix\n";
        return 1;
    }
    if (!check_multiply_matrix_vector())
    {
        std::cerr << "FAIL: multiply matrix vector\n";
        return 1;
    }
    return 0;
}
