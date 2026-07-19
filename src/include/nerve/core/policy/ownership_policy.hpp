
#pragma once
#include "nerve/core/policy/owned_buffer.hpp"

#include <algorithm>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nerve::core
{

template <typename T>
concept BufferCompatible = std::is_trivially_copyable_v<T>;

enum class OwnershipType
{
    BORROWED,
    OWNED,
    VIEW,
    COPIED,
    HUGE_BUFFER,
    PINNED
};
template <BufferCompatible T>
class BufferView
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T *;
    using const_iterator = const T *;
    BufferView() noexcept
        : data_(nullptr)
        , size_(0)
    {}
    BufferView(T *data, size_type size) noexcept
        : data_(data)
        , size_(size)
    {}
    BufferView(const std::vector<std::remove_const_t<T>> &vector) noexcept
        : data_(const_cast<T *>(vector.data()))
        , size_(vector.size())
    {}
    BufferView(const BufferView &) = default;
    BufferView &operator=(const BufferView &) = default;
    BufferView(BufferView &&) noexcept = default;
    BufferView &operator=(BufferView &&) noexcept = default;
    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    T &operator[](size_type index) noexcept { return data_[index]; }
    const T &operator[](size_type index) const noexcept { return data_[index]; }
    size_type size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    iterator begin() noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + size_; }
    BufferView subspan(size_type offset, size_type count) const noexcept
    {
        return BufferView(data_ + offset, count);
    }

private:
    T *data_;
    size_type size_;
};
template <typename T>
class PinnedBufferView
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T *;
    using const_iterator = const T *;
    PinnedBufferView() noexcept
        : data_(nullptr)
        , size_(0)
        , numa_node_(-1)
    {}
    PinnedBufferView(T *data, size_type size, int numaNode = -1) noexcept
        : data_(data)
        , size_(size)
        , numa_node_(numaNode)
    {}
    PinnedBufferView(const std::vector<std::remove_const_t<T>> &vector, int numaNode = -1) noexcept
        : data_(const_cast<T *>(vector.data()))
        , size_(vector.size())
        , numa_node_(numaNode)
    {}
    PinnedBufferView(const PinnedBufferView &) = default;
    PinnedBufferView &operator=(const PinnedBufferView &) = default;
    PinnedBufferView(PinnedBufferView &&) noexcept = default;
    PinnedBufferView &operator=(PinnedBufferView &&) noexcept = default;
    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    T &operator[](size_type index) noexcept { return data_[index]; }
    const T &operator[](size_type index) const noexcept { return data_[index]; }
    size_type size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    int numaNode() const noexcept { return numa_node_; }
    iterator begin() noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + size_; }
    PinnedBufferView subspan(size_type offset, size_type count) const noexcept
    {
        return PinnedBufferView(data_ + offset, count, numa_node_);
    }
    bool isNumaBound() const noexcept { return numa_node_ >= 0; }
    void bindToNuma(int node) noexcept { numa_node_ = node; }

private:
    T *data_;
    size_type size_;
    int numa_node_;
};
class PointBuffer
{
public:
    using value_type = double;
    using size_type = std::size_t;
    PointBuffer() noexcept
        : data_(nullptr)
        , size_(0)
        , dimension_(0)
    {}
    PointBuffer(double *data, size_type num_points, size_type dim) noexcept
        : data_(data)
        , size_(num_points)
        , dimension_(dim)
    {}
    PointBuffer(const std::vector<std::vector<double>> &points)
        : data_(nullptr)
        , size_(0)
        , dimension_(0)
    {
        if (!points.empty())
        {
            dimension_ = points[0].size();
            size_ = points.size();
            for (const auto &point : points)
            {
                if (point.size() != dimension_)
                {
                    throw std::invalid_argument("PointBuffer requires rectangular input");
                }
            }
            if (dimension_ != 0 && size_ > std::numeric_limits<size_type>::max() / dimension_)
            {
                throw std::length_error("PointBuffer size overflow");
            }
            data_ = new double[size_ * dimension_];
            for (size_type i = 0; i < size_; ++i)
            {
                std::copy(points[i].begin(), points[i].end(), data_ + i * dimension_);
            }
        }
    }
    PointBuffer(PointBuffer &&other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , dimension_(other.dimension_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.dimension_ = 0;
    }
    ~PointBuffer() { delete[] data_; }
    PointBuffer &operator=(PointBuffer &&other) noexcept
    {
        if (this != &other)
        {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            dimension_ = other.dimension_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.dimension_ = 0;
        }
        return *this;
    }
    double *data() noexcept { return data_; }
    const double *data() const noexcept { return data_; }
    double *getPoint(size_type index) noexcept { return data_ + index * dimension_; }
    const double *getPoint(size_type index) const noexcept { return data_ + index * dimension_; }
    double getCoordinate(size_type point_index, size_type coord_index) const noexcept
    {
        return data_[point_index * dimension_ + coord_index];
    }
    size_type size() const noexcept { return size_; }
    size_type dimension() const noexcept { return dimension_; }
    size_type totalSize() const noexcept { return size_ * dimension_; }
    bool empty() const noexcept { return size_ == 0; }
    double *release() noexcept
    {
        double *result = data_;
        data_ = nullptr;
        size_ = 0;
        dimension_ = 0;
        return result;
    }

private:
    double *data_;
    size_type size_;
    size_type dimension_;
};
namespace ownership_utils
{
template <typename T>
BufferView<T> makeView(T *data, std::size_t size) noexcept
{
    return BufferView<T>(data, size);
}
template <typename T>
BufferView<T> makeView(const std::vector<std::remove_const_t<T>> &vector) noexcept
{
    return BufferView<T>(vector);
}
template <typename T>
OwnedBuffer<T> makeOwned(std::size_t size)
{
    return OwnedBuffer<T>(size);
}
template <typename T>
struct is_buffer_view : std::false_type
{};
template <typename T>
struct is_buffer_view<BufferView<T>> : std::true_type
{};
template <typename T>
struct is_owned_buffer : std::false_type
{};
template <typename T>
struct is_owned_buffer<OwnedBuffer<T>> : std::true_type
{};
using PointView = BufferView<const double>;
using OwnedPointBuffer = PointBuffer;
using PinnedPointView = PinnedBufferView<const double>;
template <typename T>
using HugeBufferView = PinnedBufferView<T>;
template <typename T>
PinnedBufferView<T> makePinnedView(T *data, std::size_t size, int numaNode = -1) noexcept
{
    return PinnedBufferView<T>(data, size, numaNode);
}
template <typename T>
PinnedBufferView<T> makePinnedView(const std::vector<T> &vector, int numaNode = -1) noexcept
{
    return PinnedBufferView<T>(vector, numaNode);
}
template <typename T>
PinnedBufferView<T> makeHugeBufferView(T *data, std::size_t size, int numaNode = 0) noexcept
{
    return PinnedBufferView<T>(data, size, numaNode);
}
static_assert(std::is_trivially_destructible_v<BufferView<double>>,
              "BufferView must be trivially destructible for hot-path performance");
static_assert(std::is_trivially_copyable_v<BufferView<double>>,
              "BufferView must be trivially copyable (non-owning view, std::span semantics)");
static_assert(std::is_trivially_destructible_v<PinnedBufferView<double>>,
              "PinnedBufferView must be trivially destructible for hot-path performance");
static_assert(std::is_trivially_copyable_v<PinnedBufferView<double>>,
              "PinnedBufferView must be trivially copyable (non-owning view, std::span semantics)");
static_assert(sizeof(BufferView<double>) <= 16, "BufferView must be small for hot-path efficiency");
static_assert(sizeof(PinnedBufferView<double>) <= 24,
              "PinnedBufferView must be small for hot-path efficiency");
template <typename T>
constexpr bool is_zero_copy_compatible_v =
    std::is_same_v<T, BufferView<typename T::value_type>> ||
    std::is_same_v<T, PinnedBufferView<typename T::value_type>>;
template <typename T>
constexpr bool is_huge_buffer_compatible_v =
    std::is_same_v<T, PinnedBufferView<typename T::value_type>>;
} // namespace ownership_utils

} // namespace nerve::core
