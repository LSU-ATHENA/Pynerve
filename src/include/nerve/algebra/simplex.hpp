
#pragma once
#include "nerve/config.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <compare>
#include <functional>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#define SIMPLEX_INDEX nerve::Index
#define SIMPLEX_SIZE nerve::Size

namespace nerve::algebra
{

class Simplex
{
public:
    Simplex() = default;
    explicit Simplex(const std::vector<SIMPLEX_INDEX> &vertices);
    Simplex(std::initializer_list<Index> vertices);

    [[nodiscard]] Size dimension() const noexcept;
    [[nodiscard]] Size numVertices() const noexcept;
    [[nodiscard]] const std::vector<Index> &vertices() const noexcept;
    [[nodiscard]] bool contains(Index vertex) const;
    [[nodiscard]] Index operator[](Size i) const;
    [[nodiscard]] Index at(Size i) const;
    Simplex boundary(Size k, const core::DeterminismContract &contract) const;
    Simplex boundary(Size k = 1) const;
    std::vector<Simplex> faces(const core::DeterminismContract &contract) const;
    std::vector<Simplex> cofaces(const std::vector<Simplex> &complex,
                                 const core::DeterminismContract &contract) const;
    std::vector<Simplex> faces() const;
    std::vector<Simplex> cofaces(const std::vector<Simplex> &complex) const;
    std::vector<Simplex> kFaces(Size k, const core::DeterminismContract &contract) const;
    std::vector<Simplex> kFaces(Size k) const;
    Simplex faceWithoutVertex(Index vertex, const core::DeterminismContract &contract) const;
    Simplex faceWithoutVertex(Index vertex) const;
    std::vector<Simplex> facesWithoutVertices(const std::vector<Index> &vertices,
                                              const core::DeterminismContract &contract) const;
    std::vector<Simplex> facesWithoutVertices(const std::vector<Index> &vertices) const;
    std::vector<Simplex> potentialCofaces(const std::vector<Index> &available_vertices,
                                          const core::DeterminismContract &contract) const;
    std::vector<Simplex> potentialCofaces(const std::vector<Index> &available_vertices) const;
    std::vector<Simplex> kCofaces(Size k, const std::vector<Simplex> &complex,
                                  const core::DeterminismContract &contract) const;
    std::vector<Simplex> kCofaces(Size k, const std::vector<Simplex> &complex) const;
    [[nodiscard]] bool operator==(const Simplex &other) const
    {
        return vertices_ == other.vertices_;
    }
    [[nodiscard]] bool operator<(const Simplex &other) const { return vertices_ < other.vertices_; }
    bool isFaceOf(const Simplex &other) const;
    double volume(const core::ownership_utils::PointView &coords,
                  const core::DeterminismContract &contract) const;
    double volume(const core::ownership_utils::PointView &coords) const;
    double barycentricCoordinate(const core::ownership_utils::PointView &point,
                                 const core::ownership_utils::PointView &coords,
                                 const core::DeterminismContract &contract) const;
    double barycentricCoordinate(const core::ownership_utils::PointView &point,
                                 const core::ownership_utils::PointView &coords) const;
    std::vector<double> circumcenter(const core::ownership_utils::PointView &coords,
                                     const core::DeterminismContract &contract) const;
    std::vector<double> circumcenter(const core::ownership_utils::PointView &coords) const;
    double circumradius(const core::ownership_utils::PointView &coords,
                        const core::DeterminismContract &contract) const;
    double circumradius(const core::ownership_utils::PointView &coords) const;
    bool containsPoint(const core::ownership_utils::PointView &point,
                       const core::ownership_utils::PointView &coords,
                       const core::DeterminismContract &contract) const;
    bool containsPoint(const core::ownership_utils::PointView &point,
                       const core::ownership_utils::PointView &coords) const;
    Simplex join(const Simplex &other, const core::DeterminismContract &contract) const;
    Simplex join(const Simplex &other) const;
    Simplex meet(const Simplex &other, const core::DeterminismContract &contract) const;
    Simplex meet(const Simplex &other) const;
    Simplex star(const std::vector<Simplex> &complex,
                 const core::DeterminismContract &contract) const;
    Simplex star(const std::vector<Simplex> &complex) const;
    Simplex link(const std::vector<Simplex> &complex,
                 const core::DeterminismContract &contract) const;
    Simplex link(const std::vector<Simplex> &complex) const;
    struct Hash
    {
        Size operator()(const Simplex &simplex) const noexcept;
    };
    std::string toString() const;
    void sortVertices();

private:
    std::vector<Index> vertices_;
    void canonicalize();
    std::vector<Index> computeFaceIndices(Size i) const;
    double simplexVolume(const std::vector<std::vector<double>> &coords) const;
    double computeDeterminant(const std::vector<std::vector<double>> &matrix) const;
    Size factorial(Size n) const;
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
    iterator begin() { return simplices_.begin(); }
    iterator end() { return simplices_.end(); }
    const_iterator begin() const { return simplices_.begin(); }
    const_iterator end() const { return simplices_.end(); }
    const_iterator cbegin() const { return simplices_.cbegin(); }
    const_iterator cend() const { return simplices_.cend(); }
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
std::vector<Simplex> generateAllFaces(const Simplex &simplex);
std::vector<Simplex> generateKFaces(const Simplex &simplex, Size k);
Simplex join(const Simplex &s1, const Simplex &s2);
Simplex meet(const Simplex &s1, const Simplex &s2);
bool areCompatible(const Simplex &s1, const Simplex &s2);
} // namespace nerve::algebra
