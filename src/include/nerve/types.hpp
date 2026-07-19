#pragma once
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <vector>
namespace nerve
{
using Index = std::int32_t;
using Size = std::size_t;
using Dimension = std::int32_t;
using Field = double;
using PackedWord = std::uint64_t;
using Offset = std::int64_t;

inline constexpr Size kBitsPerPackedWord = 64;
inline constexpr Size kPackedWordBytes = sizeof(PackedWord);

template <typename T>
concept Numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <typename T>
concept IndexLike = std::is_integral_v<T> && !std::is_same_v<T, bool>;

template <typename T, typename = void>
struct is_numeric : std::false_type
{};

template <Numeric T>
struct is_numeric<T, void> : std::true_type
{};

template <typename T, typename = void>
struct is_index_like : std::false_type
{};

template <IndexLike T>
struct is_index_like<T, void> : std::true_type
{};

template <typename T>
using Vector = std::vector<T>;
template <typename T>
using UniquePtr = std::unique_ptr<T>;
template <typename T>
using SharedPtr = std::shared_ptr<T>;
template <typename T>
using Span = std::vector<T>;
template <typename T>
using ConstSpan = std::vector<T>;
struct Simplex
{
    Vector<Index> vertices;
    Field value = 0.0;
    Index simplex_index = 0;

    Simplex() = default;
    Simplex(Vector<Index> verts, Field val)
        : vertices(std::move(verts))
        , value(val)
    {}

    [[nodiscard]] Dimension dimension() const noexcept
    {
        return vertices.empty() ? Dimension{-1} : static_cast<Dimension>(vertices.size() - 1);
    }

    [[nodiscard]] Field filtration_value() const noexcept { return value; }
    [[nodiscard]] Field &filtration_value() noexcept { return value; }
    [[nodiscard]] Index index() const noexcept { return simplex_index; }
    [[nodiscard]] Index &index() noexcept { return simplex_index; }

    [[nodiscard]] Dimension dim() const noexcept { return dimension(); }

    bool operator==(const Simplex &other) const { return vertices == other.vertices; }
    bool operator<(const Simplex &other) const { return vertices < other.vertices; }
    auto begin() noexcept { return vertices.begin(); }
    auto end() noexcept { return vertices.end(); }
    auto begin() const noexcept { return vertices.begin(); }
    auto end() const noexcept { return vertices.end(); }
    auto cbegin() const noexcept { return vertices.cbegin(); }
    auto cend() const noexcept { return vertices.cend(); }
    [[nodiscard]] ConstSpan<Index> asSpan() const noexcept { return vertices; }
    [[nodiscard]] Span<Index> asSpan() noexcept { return vertices; }
};
struct Pair
{
    Field birth = 0.0;
    Field death = 0.0;
    Dimension dimension = 0;
    Index birth_index = -1;
    Index death_index = -1;

    Pair() = default;
    constexpr Pair(Field b, Field d, Dimension dim = 0, Index bi = -1, Index di = -1) noexcept
        : birth(b), death(d), dimension(dim), birth_index(bi), death_index(di)
    {}

    [[nodiscard]] constexpr Field lifetime() const noexcept { return death - birth; }
    [[nodiscard]] constexpr bool isInfinite() const noexcept
    {
        return death == std::numeric_limits<Field>::infinity();
    }
    auto operator<=>(const Pair &other) const = default;
};
enum class MemoryLocation
{
    Host,
    Device,
    Pinned
};
struct DeviceInfo
{
    int device_id;
    std::string name;
    std::size_t total_memory;
    std::size_t available_memory;
    int compute_capability;

    [[nodiscard]] bool operator==(const DeviceInfo &other) const
    {
        return device_id == other.device_id && name == other.name;
    }
    [[nodiscard]] bool operator<(const DeviceInfo &other) const
    {
        return std::tie(device_id, name) < std::tie(other.device_id, other.name);
    }
};
} // namespace nerve
