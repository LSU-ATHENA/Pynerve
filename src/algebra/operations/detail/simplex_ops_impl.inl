#include "nerve/algebra/simplex.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nerve::algebra {
namespace {

std::vector<Index> vertices_without_position(const std::vector<Index>& vertices, Size position) {
    std::vector<Index> face_vertices;
    if (position >= vertices.size()) {
        return face_vertices;
    }
    face_vertices.reserve(vertices.size() - 1);
    for (Size i = 0; i < vertices.size(); ++i) {
        if (i != position) {
            face_vertices.push_back(vertices[i]);
        }
    }
    return face_vertices;
}

std::vector<Simplex> faces_with_vertex_count(const std::vector<Index>& vertices,
                                             Size vertex_count) {
    std::vector<Simplex> result;
    if (vertex_count == 0 || vertices.size() < vertex_count) {
        return result;
    }

    std::vector<bool> selection(vertices.size());
    std::fill(selection.end() - static_cast<std::ptrdiff_t>(vertex_count), selection.end(), true);
    do {
        std::vector<Index> face_vertices;
        face_vertices.reserve(vertex_count);
        for (Size i = 0; i < vertices.size(); ++i) {
            if (selection[i]) {
                face_vertices.push_back(vertices[i]);
            }
        }
        result.emplace_back(face_vertices);
    } while (std::next_permutation(selection.begin(), selection.end()));

    return result;
}

bool strictContractUnsatisfied(const core::DeterminismContract& contract) {
    return contract.level == core::DeterminismLevel::STRICT &&
           !core::DeterminismEnforcer::canSatisfyContract(contract);
}

}  // namespace

Simplex::Simplex(const std::vector<Index>& vertices) : vertices_(vertices) {
    canonicalize();
}
Simplex::Simplex(std::initializer_list<Index> vertices) : vertices_(vertices) {
    canonicalize();
}

[[nodiscard]] Size Simplex::dimension() const noexcept {
    return vertices_.empty() ? 0 : vertices_.size() - 1;
}

[[nodiscard]] Size Simplex::numVertices() const noexcept {
    return vertices_.size();
}
const std::vector<Index>& Simplex::vertices() const noexcept {
    return vertices_;
}
bool Simplex::contains(Index vertex) const {
    return std::ranges::find(vertices_, vertex) != vertices_.end();
}
Index Simplex::operator[](Size i) const {
    return vertices_[i];
}
Index Simplex::at(Size i) const {
    if (i >= vertices_.size()) {
        throw std::out_of_range("Simplex index out of range");
    }
    return vertices_[i];
}
Simplex Simplex::boundary(Size k) const {
    if (k == 0 || k > vertices_.size()) {
        return Simplex();
    }
    return Simplex(vertices_without_position(vertices_, k - 1));
}
Simplex Simplex::boundary(Size k, const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(Size) const>(&Simplex::boundary);
    return (this->*impl)(k);
}
std::vector<Simplex> Simplex::faces(const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)() const>(&Simplex::faces);
    return (this->*impl)();
}
std::vector<Simplex> Simplex::faces() const {
    if (vertices_.size() <= 1) {
        return {};
    }
    return faces_with_vertex_count(vertices_, vertices_.size() - 1);
}
std::vector<Simplex> Simplex::cofaces(const std::vector<Simplex>& complex) const {
    std::vector<Simplex> cofaces;
    for (const auto& simplex : complex) {
        if (isFaceOf(simplex)) {
            cofaces.push_back(simplex);
        }
    }
    return cofaces;
}
std::vector<Simplex> Simplex::cofaces(
    const std::vector<Simplex>& complex,
    const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)(
        const std::vector<Simplex>&) const>(&Simplex::cofaces);
    return (this->*impl)(complex);
}
std::vector<Simplex> Simplex::kFaces(Size k, const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)(Size) const>(
        &Simplex::kFaces);
    return (this->*impl)(k);
}
std::vector<Simplex> Simplex::kFaces(Size k) const {
    if (k > dimension() || vertices_.size() < k + 1) {
        return {};
    }
    return faces_with_vertex_count(vertices_, k + 1);
}
Simplex Simplex::faceWithoutVertex(Index vertex) const {
    auto it = std::ranges::find(vertices_, vertex);
    if (it == vertices_.end()) {
        return Simplex();
    }
    return Simplex(vertices_without_position(vertices_, static_cast<Size>(it - vertices_.begin())));
}
Simplex Simplex::faceWithoutVertex(Index vertex, const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(Index) const>(
        &Simplex::faceWithoutVertex);
    return (this->*impl)(vertex);
}
std::vector<Simplex> Simplex::facesWithoutVertices(const std::vector<Index>& vertices) const {
    std::vector<Simplex> result;
    const auto face_impl = static_cast<Simplex (Simplex::*)(Index) const>(
        &Simplex::faceWithoutVertex);
    for (Index vertex : vertices) {
        auto face = (this->*face_impl)(vertex);
        if (face.numVertices() > 0) {
            result.push_back(face);
        }
    }
    return result;
}
std::vector<Simplex> Simplex::facesWithoutVertices(
    const std::vector<Index>& vertices,
    const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)(
        const std::vector<Index>&) const>(&Simplex::facesWithoutVertices);
    return (this->*impl)(vertices);
}
std::vector<Simplex> Simplex::potentialCofaces(const std::vector<Index>& available_vertices) const {
    std::vector<Simplex> cofaces;
    for (Index vertex : available_vertices) {
        if (!contains(vertex)) {
            std::vector<Index> coface_vertices = vertices_;
            coface_vertices.push_back(vertex);
            cofaces.emplace_back(coface_vertices);
        }
    }
    return cofaces;
}
std::vector<Simplex> Simplex::potentialCofaces(
    const std::vector<Index>& available_vertices,
    const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)(
        const std::vector<Index>&) const>(&Simplex::potentialCofaces);
    return (this->*impl)(available_vertices);
}
std::vector<Simplex> Simplex::kCofaces(Size k, const std::vector<Simplex>& complex) const {
    std::vector<Simplex> kCofaces;
    for (const auto& simplex : complex) {
        if (simplex.dimension() == k && isFaceOf(simplex)) {
            kCofaces.push_back(simplex);
        }
    }
    return kCofaces;
}
std::vector<Simplex> Simplex::kCofaces(
    Size k,
    const std::vector<Simplex>& complex,
    const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return {};
    }
    const auto impl = static_cast<std::vector<Simplex> (Simplex::*)(
        Size, const std::vector<Simplex>&) const>(&Simplex::kCofaces);
    return (this->*impl)(k, complex);
}
bool Simplex::isFaceOf(const Simplex& other) const {
    if (vertices_.size() > other.vertices_.size()) {
        return false;
    }
    return std::ranges::includes(other.vertices_, vertices_);
}
Simplex Simplex::join(const Simplex& other, const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(const Simplex&) const>(&Simplex::join);
    return (this->*impl)(other);
}
Simplex Simplex::join(const Simplex& other) const {
    std::vector<Index> joined_vertices = vertices_;
    for (Index v : other.vertices_) {
        if (!contains(v)) {
            joined_vertices.push_back(v);
        }
    }
    return Simplex(joined_vertices);
}
Simplex Simplex::meet(const Simplex& other, const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(const Simplex&) const>(&Simplex::meet);
    return (this->*impl)(other);
}
Simplex Simplex::meet(const Simplex& other) const {
    std::vector<Index> intersection;
    std::ranges::set_intersection(vertices_, other.vertices_, std::back_inserter(intersection));
    return Simplex(intersection);
}
Simplex Simplex::star(const std::vector<Simplex>& complex) const {
    std::vector<Simplex> star_simplices;
    for (const auto& simplex : complex) {
        if (isFaceOf(simplex)) {
            star_simplices.push_back(simplex);
        }
    }
    if (!star_simplices.empty()) {
        auto max_it = std::max_element(star_simplices.begin(), star_simplices.end(),
            [](const Simplex& a, const Simplex& b) {
                return a.dimension() < b.dimension();
            });
        return *max_it;
    }
    return Simplex();
}
Simplex Simplex::star(const std::vector<Simplex>& complex,
                      const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(const std::vector<Simplex>&) const>(
        &Simplex::star);
    return (this->*impl)(complex);
}
Simplex Simplex::link(const std::vector<Simplex>& complex) const {
    const auto star_impl = static_cast<Simplex (Simplex::*)(
        const std::vector<Simplex>&) const>(&Simplex::star);
    const auto faces_impl = static_cast<std::vector<Simplex> (Simplex::*)() const>(
        &Simplex::faces);
    const auto meet_impl = static_cast<Simplex (Simplex::*)(const Simplex&) const>(
        &Simplex::meet);
    auto star_simplex = (this->*star_impl)(complex);
    auto boundary_faces = (this->*faces_impl)();
    if (boundary_faces.empty()) {
        return star_simplex;
    }
    return (star_simplex.*meet_impl)(boundary_faces[0]);
}
Simplex Simplex::link(const std::vector<Simplex>& complex,
                      const core::DeterminismContract& contract) const {
    if (strictContractUnsatisfied(contract)) {
        return Simplex();
    }
    const auto impl = static_cast<Simplex (Simplex::*)(
        const std::vector<Simplex>&) const>(&Simplex::link);
    return (this->*impl)(complex);
}
Size Simplex::Hash::operator()(const Simplex& simplex) const noexcept {
    Size hash = 0;
    for (Index vertex : simplex.vertices()) {
        hash ^= std::hash<Index>{}(vertex) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}
std::string Simplex::toString() const {
    std::ostringstream oss;
    oss << "[";
    for (Size i = 0; i < vertices_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vertices_[i];
    }
    oss << "]";
    return oss.str();
}
void Simplex::sortVertices() {
    std::ranges::sort(vertices_);
}
void Simplex::canonicalize() {
    sortVertices();
    const auto [first, last] = std::ranges::unique(vertices_);
    vertices_.erase(first, last);
}
std::vector<Index> Simplex::computeFaceIndices(Size i) const {
    std::vector<Index> face_indices;
    face_indices.reserve(vertices_.size() - 1);
    for (Size j = 0; j < vertices_.size(); ++j) {
        if (j != i) {
            face_indices.push_back(vertices_[j]);
        }
    }
    return face_indices;
}

std::vector<Simplex> generateAllFaces(const Simplex &simplex)
{
    std::vector<Simplex> faces;
    auto idx = simplex.faces();
    for (const auto &f : idx)
    {
        faces.push_back(f);
    }
    return faces;
}

Simplex join(const Simplex &s1, const Simplex &s2)
{
    return s1.join(s2);
}

} // namespace nerve::algebra
