#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace nerve
{

using Dimension = int;
using Index = std::int64_t;
using Size = std::size_t;

} // namespace nerve

namespace nerve::algebra
{

class Simplex
{
public:
    Simplex() = default;
    explicit Simplex(const std::vector<int> &vertices);
    Simplex(std::initializer_list<int> vertices);

    Size dimension() const noexcept;
    Size numVertices() const noexcept;
    const std::vector<int> &vertices() const noexcept;
    bool contains(int vertex) const;
    int operator[](Size i) const;
    int at(Size i) const;

    std::vector<Simplex> faces() const;
    std::vector<Simplex> kFaces(Size k) const;
    Simplex faceWithoutVertex(int vertex) const;
    bool isFaceOf(const Simplex &other) const;

    double volume(const std::vector<std::vector<double>> &coords) const
    {
        (void)coords;
        return 0.0;
    }

    bool operator==(const Simplex &other) const { return vertices_ == other.vertices_; }
    bool operator<(const Simplex &other) const { return vertices_ < other.vertices_; }

    struct Hash
    {
        Size operator()(const Simplex &simplex) const noexcept;
    };

private:
    std::vector<int> vertices_;
    void canonicalize();
};

class SimplexSet
{
public:
    using iterator = std::unordered_set<Simplex, Simplex::Hash>::iterator;
    using const_iterator = std::unordered_set<Simplex, Simplex::Hash>::const_iterator;

    bool insert(const Simplex &simplex);
    bool erase(const Simplex &simplex);
    bool contains(const Simplex &simplex) const;
    Size size() const noexcept;
    bool empty() const noexcept;
    void clear();
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    SimplexSet intersection(const SimplexSet &other) const;
    SimplexSet unionSet(const SimplexSet &other) const;
    SimplexSet difference(const SimplexSet &other) const;
    SimplexSet kSimplices(Size k) const;
    Size numKSimplices(Size k) const;
    Size maxDimension() const;
    SimplexSet boundary() const;
    SimplexSet kBoundary(Size k) const;
    SimplexSet star(const Simplex &simplex) const;
    SimplexSet link(const Simplex &simplex) const;
    std::vector<Simplex> toVector() const;

private:
    std::unordered_set<Simplex, Simplex::Hash> simplices_;
};

class SimplicialComplex
{
public:
    SimplicialComplex() = default;
    void addSimplex(const Simplex &simplex) { simplices_.insert(simplex); }
    void removeSimplex(const Simplex &simplex) { simplices_.erase(simplex); }
    void clear() { simplices_.clear(); }

    Size size() const noexcept { return simplices_.size(); }
    Size numSimplices() const noexcept { return simplices_.size(); }
    int maxDimension() const noexcept
    {
        int max_d = -1;
        for (const auto &s : simplices_)
        {
            if (s.dimension() > max_d)
                max_d = s.dimension();
        }
        return max_d;
    }
    std::vector<Simplex> simplicesOfDimension(int dim) const
    {
        std::vector<Simplex> result;
        for (const auto &s : simplices_)
        {
            if (s.dimension() == dim)
                result.push_back(s);
        }
        return result;
    }
    std::vector<Simplex> getSimplices() const
    {
        std::vector<Simplex> result(simplices_.begin(), simplices_.end());
        return result;
    }

private:
    std::unordered_set<Simplex, Simplex::Hash> simplices_;
};

class BoundaryMatrix
{
public:
    BoundaryMatrix() = default;
    explicit BoundaryMatrix(const SimplicialComplex &complex);
    BoundaryMatrix(const SimplicialComplex &complex, Size dimension);

    Size rows() const noexcept;
    Size cols() const noexcept;
    Size dimension() const noexcept;
    bool isEmpty() const noexcept;

private:
    std::vector<std::vector<double>> entries_;
    Size rows_ = 0;
    Size cols_ = 0;
    Size dimension_ = 0;
};

class ChainComplex
{
public:
    explicit ChainComplex(const SimplicialComplex &complex);
    const BoundaryMatrix &boundary(Size k) const;
    BoundaryMatrix &boundary(Size k);
    Size rank(Size k) const;
    Size bettiNumber(Size k) const;
    Size maxDimension() const noexcept;

private:
    std::vector<BoundaryMatrix> boundary_matrices_;
    Size max_dimension_;
};

class SIMDDistanceCalculator
{
public:
    SIMDDistanceCalculator();

    double euclideanDistance(const double *a, const double *b, size_t dim);
    double manhattanDistance(const double *a, const double *b, size_t dim);
    double cosineDistance(const double *a, const double *b, size_t dim);

    std::vector<double> batchEuclideanDistances(const double *points, size_t num_points,
                                                size_t dim);

    bool hasAvx512() const noexcept;
    bool hasAvx2() const noexcept;
    bool hasFma() const noexcept;

    double euclideanDistanceScalar(const double *a, const double *b, size_t dim);
    double manhattanDistanceScalar(const double *a, const double *b, size_t dim);
    double cosineDistanceScalar(const double *a, const double *b, size_t dim);
};

int simplexDimension(const std::vector<int> &vertices);
std::vector<int> simplexVertices(const std::vector<int> &vertices);
std::vector<std::vector<int>> enumerateFaces(const std::vector<int> &simplex);

namespace distance
{
double euclidean(const double *a, const double *b, size_t dim);
double manhattan(const double *a, const double *b, size_t dim);
double cosine(const double *a, const double *b, size_t dim);
} // namespace distance

namespace geometry
{
double triangleArea(const double *a, const double *b, const double *c);
double tetrahedronVolume(const double *a, const double *b, const double *c, const double *d);
} // namespace geometry

} // namespace nerve::algebra
