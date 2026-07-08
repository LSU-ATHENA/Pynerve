
#include "nerve/algebra/boundary.hpp"

#include <algorithm>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>

namespace nerve::algebra
{

namespace
{

Dimension safeMaxDimension(const SimplicialComplex &complex)
{
    return complex.maxDimension();
}

auto filtrationOrder(const SimplicialComplex &complex)
{
    return [&complex](const Simplex &a, const Simplex &b) {
        const double fa = complex.getFiltration(a);
        const double fb = complex.getFiltration(b);
        if (fa != fb)
        {
            return fa < fb;
        }
        if (a.dimension() != b.dimension())
        {
            return a.dimension() < b.dimension();
        }
        return a < b;
    };
}

Index checkedIndex(Size value)
{
    if (value > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        throw std::length_error("Boundary matrix index exceeds Index range");
    }
    return static_cast<Index>(value);
}

} // namespace

BoundaryMatrix::BoundaryMatrix(const SimplicialComplex &complex)
    : rows_(0)
    , cols_(0)
    , dimension_(0)
{
    buildFromComplex(complex);
}

BoundaryMatrix::BoundaryMatrix(const SimplicialComplex &complex, Size dimension)
    : rows_(0)
    , cols_(0)
    , dimension_(dimension)
{
    buildKDimensional(complex, dimension);
}

void BoundaryMatrix::buildFromComplex(const SimplicialComplex &complex)
{
    entries_.clear();
    simplex_to_col_.clear();
    simplex_to_row_.clear();
    col_to_simplex_.clear();
    row_to_simplex_.clear();
    filtration_values_.clear();
    row_filtration_values_.clear();
    col_heights_.clear();
    max_column_height_ = 0;
    last_low_row_to_col_.clear();

    const Dimension max_dim = safeMaxDimension(complex);
    if (max_dim < 0)
    {
        rows_ = 0;
        cols_ = 0;
        dimension_ = 0;
        validateMatrix();
        return;
    }
    dimension_ = static_cast<Size>(max_dim);

    buildIndexMaps(complex);
    col_heights_.assign(cols_, 0);
    auto boundary_entries = computeBoundaryEntries(complex);
    for (const auto &entry : boundary_entries)
    {
        setEntry(entry.row, entry.col, entry.value);
    }

    filtration_values_.reserve(col_to_simplex_.size());
    for (const auto &simplex : col_to_simplex_)
    {
        filtration_values_.emplace_back(complex.getFiltration(simplex));
    }

    row_filtration_values_.reserve(row_to_simplex_.size());
    for (const auto &simplex : row_to_simplex_)
    {
        row_filtration_values_.emplace_back(complex.getFiltration(simplex));
    }

    validateMatrix();
}

void BoundaryMatrix::buildKDimensional(const SimplicialComplex &complex, Size k)
{
    entries_.clear();
    simplex_to_col_.clear();
    simplex_to_row_.clear();
    col_to_simplex_.clear();
    row_to_simplex_.clear();
    filtration_values_.clear();
    row_filtration_values_.clear();
    col_heights_.clear();
    max_column_height_ = 0;
    last_low_row_to_col_.clear();
    dimension_ = k;

    auto k_simplices = complex.simplicesOfDimension(static_cast<Dimension>(k));
    Vector<Simplex> km1_simplices;
    if (k > 0)
    {
        km1_simplices = complex.simplicesOfDimension(static_cast<Dimension>(k - 1));
    }

    const auto order = filtrationOrder(complex);
    std::ranges::sort(k_simplices, order);
    std::ranges::sort(km1_simplices, order);

    Size col_idx = 0;
    for (const auto &simplex : k_simplices)
    {
        simplex_to_col_[simplex] = checkedIndex(col_idx++);
        col_to_simplex_.push_back(simplex);
    }

    Size row_idx = 0;
    for (const auto &simplex : km1_simplices)
    {
        simplex_to_row_[simplex] = checkedIndex(row_idx++);
        row_to_simplex_.push_back(simplex);
    }

    rows_ = row_idx;
    cols_ = col_idx;
    col_heights_.assign(cols_, 0);

    for (const auto &simplex : k_simplices)
    {
        auto faces = simplex.faces({});
        auto col = simplex_to_col_[simplex];
        for (const auto &face : faces)
        {
            auto row_it = simplex_to_row_.find(face);
            if (row_it != simplex_to_row_.end())
            {
                auto row = row_it->second;
                double coeff = computeBoundaryCoefficient(simplex, face);
                setEntry(row, col, coeff);
            }
        }
    }

    filtration_values_.reserve(col_to_simplex_.size());
    for (const auto &simplex : col_to_simplex_)
    {
        filtration_values_.push_back(complex.getFiltration(simplex));
    }

    row_filtration_values_.reserve(row_to_simplex_.size());
    for (const auto &simplex : row_to_simplex_)
    {
        row_filtration_values_.push_back(complex.getFiltration(simplex));
    }

    validateMatrix();
}

void BoundaryMatrix::buildIndexMaps(const SimplicialComplex &complex)
{
    Size col_idx = 0;
    const Dimension max_dim = safeMaxDimension(complex);
    if (max_dim < 0)
    {
        rows_ = 0;
        cols_ = 0;
        return;
    }
    const auto order = filtrationOrder(complex);
    for (Size k = 0; k <= static_cast<Size>(max_dim); ++k)
    {
        auto k_simplices = complex.simplicesOfDimension(static_cast<Dimension>(k));
        std::ranges::sort(k_simplices, order);
        for (const auto &simplex : k_simplices)
        {
            simplex_to_col_[simplex] = checkedIndex(col_idx++);
            col_to_simplex_.push_back(simplex);
        }
    }

    Size row_idx = 0;
    for (Size k = 1; k <= static_cast<Size>(max_dim); ++k)
    {
        auto km1_simplices = complex.simplicesOfDimension(static_cast<Dimension>(k - 1));
        std::ranges::sort(km1_simplices, order);
        for (const auto &simplex : km1_simplices)
        {
            simplex_to_row_[simplex] = checkedIndex(row_idx++);
            row_to_simplex_.push_back(simplex);
        }
    }

    rows_ = row_idx;
    cols_ = col_idx;
}

std::vector<MatrixEntry> BoundaryMatrix::computeBoundaryEntries(const SimplicialComplex &complex)
{
    std::vector<MatrixEntry> entries;
    const Dimension max_dim = safeMaxDimension(complex);
    if (max_dim < 1)
    {
        return entries;
    }
    const auto order = filtrationOrder(complex);
    for (Size k = 1; k <= static_cast<Size>(max_dim); ++k)
    {
        auto k_simplices = complex.simplicesOfDimension(static_cast<Dimension>(k));
        std::ranges::sort(k_simplices, order);
        for (const auto &simplex : k_simplices)
        {
            auto faces = simplex.faces({});
            auto col_it = simplex_to_col_.find(simplex);
            if (col_it == simplex_to_col_.end())
            {
                continue;
            }

            auto col = col_it->second;
            for (const auto &face : faces)
            {
                auto row_it = simplex_to_row_.find(face);
                if (row_it == simplex_to_row_.end())
                {
                    continue;
                }
                auto row = row_it->second;
                double coeff = computeBoundaryCoefficient(simplex, face);
                entries.emplace_back(row, col, coeff);
            }
        }
    }
    return entries;
}

int BoundaryMatrix::computeBoundaryCoefficient(const Simplex &simplex, const Simplex &face)
{
    const auto &vertices = simplex.vertices();
    for (Size i = 0; i < vertices.size(); ++i)
    {
        std::vector<Index> temp_vertices;
        temp_vertices.reserve(vertices.size() - 1);
        for (Size j = 0; j < vertices.size(); ++j)
        {
            if (j != i)
            {
                temp_vertices.push_back(vertices[j]);
            }
        }
        if (Simplex(temp_vertices) == face)
        {
            return (i % 2 == 0) ? 1 : -1;
        }
    }
    return 0;
}

void BoundaryMatrix::validateMatrix() const
{
    if (rows_ != row_to_simplex_.size() || cols_ != col_to_simplex_.size())
    {
        throw std::runtime_error("Matrix dimensions inconsistent with indexing maps");
    }

    for (const auto &entry : entries_)
    {
        auto [row_col, _value] = entry;
        auto [row, col] = row_col;
        if (row >= rows_ || col >= cols_)
        {
            throw std::runtime_error("Matrix entry out of bounds");
        }
    }
}

} // namespace nerve::algebra
