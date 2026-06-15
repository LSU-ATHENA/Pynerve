#pragma once

#define NERVE_CUDA_TILE_API_DECLARATIONS_ONLY
#include "nerve/gpu/cuda_tile_api.hpp"
#undef NERVE_CUDA_TILE_API_DECLARATIONS_ONLY

namespace nerve::gpu::tile
{

namespace detail
{
inline bool checkedTileRegion(int totalRows, int totalCols, size_t pitch, int rowStart,
                              int colStart, int rows, int cols, size_t &offset);

inline void checkCudaTileCopy(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
} // namespace detail

// Inline Implementation: TileBuffer

template <typename T>
inline TileBuffer<T>::TileBuffer(int rows, int cols, TileLayout layout)
    : rows_(rows)
    , cols_(cols)
    , pitch_(static_cast<size_t>(cols))
    , layout_(layout)
    , owning_(true)
{
    if (rows_ <= 0 || cols_ <= 0)
    {
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
        data_ = nullptr;
        owning_ = false;
        return;
    }
    const size_t row_count = static_cast<size_t>(rows_);
    const size_t col_count = static_cast<size_t>(cols_);
    if (row_count > (std::numeric_limits<size_t>::max() / col_count))
    {
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
        data_ = nullptr;
        owning_ = false;
        return;
    }
    const size_t elements = row_count * col_count;
    if (elements > (std::numeric_limits<size_t>::max() / sizeof(T)))
    {
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
        data_ = nullptr;
        owning_ = false;
        return;
    }
    const size_t total_bytes = elements * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void **>(&data_), total_bytes) != cudaSuccess)
    {
        data_ = nullptr;
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
        owning_ = false;
    }
}

template <typename T>
inline TileBuffer<T>::TileBuffer(T *data, int rows, int cols, bool owning)
    : data_(data)
    , rows_(rows)
    , cols_(cols)
    , pitch_(static_cast<size_t>(cols))
    , layout_(TileLayout::kRowMajor)
    , owning_(owning && data != nullptr)
{
    if (data_ == nullptr || rows_ <= 0 || cols_ <= 0)
    {
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
    }
}

template <typename T>
inline TileBuffer<T>::~TileBuffer()
{
    if (owning_ && data_ != nullptr)
    {
        (void)cudaFree(data_);
    }
}

template <typename T>
inline TileBuffer<T>::TileBuffer(TileBuffer &&other) noexcept
    : data_(other.data_)
    , rows_(other.rows_)
    , cols_(other.cols_)
    , pitch_(other.pitch_)
    , layout_(other.layout_)
    , owning_(other.owning_)
{
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.pitch_ = 0;
    other.owning_ = false;
}

template <typename T>
inline TileBuffer<T> &TileBuffer<T>::operator=(TileBuffer &&other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    if (owning_ && data_ != nullptr)
    {
        (void)cudaFree(data_);
    }
    data_ = other.data_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    pitch_ = other.pitch_;
    layout_ = other.layout_;
    owning_ = other.owning_;
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.pitch_ = 0;
    other.owning_ = false;
    return *this;
}

template <typename T>
inline void TileBuffer<T>::copyFromHost(const T *hostData)
{
    if (data_ == nullptr || hostData == nullptr || rows_ <= 0 || cols_ <= 0)
    {
        return;
    }
    const size_t total_bytes = bytes();
    if (total_bytes == 0)
    {
        return;
    }
    detail::checkCudaTileCopy(cudaMemcpy(data_, hostData, total_bytes, cudaMemcpyHostToDevice),
                              "TileBuffer::copyFromHost");
}

template <typename T>
inline void TileBuffer<T>::copyToHost(T *hostData) const
{
    if (data_ == nullptr || hostData == nullptr || rows_ <= 0 || cols_ <= 0)
    {
        return;
    }
    const size_t total_bytes = bytes();
    if (total_bytes == 0)
    {
        return;
    }
    detail::checkCudaTileCopy(cudaMemcpy(hostData, data_, total_bytes, cudaMemcpyDeviceToHost),
                              "TileBuffer::copyToHost");
}

template <typename T>
inline void TileBuffer<T>::copyFromDevice(const T *deviceData)
{
    if (data_ == nullptr || deviceData == nullptr || rows_ <= 0 || cols_ <= 0)
    {
        return;
    }
    const size_t total_bytes = bytes();
    if (total_bytes == 0)
    {
        return;
    }
    detail::checkCudaTileCopy(cudaMemcpy(data_, deviceData, total_bytes, cudaMemcpyDeviceToDevice),
                              "TileBuffer::copyFromDevice");
}

template <typename T>
inline TileView<T> TileBuffer<T>::view(int rowStart, int colStart, int rows, int cols)
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, rowStart, colStart, rows, cols, offset))
    {
        return TileView<T>(nullptr, 0, 0, 0);
    }
    T *base = data_ + offset;
    return TileView<T>(base, rows, cols, pitch_);
}

template <typename T>
inline TileView<const T> TileBuffer<T>::view(int rowStart, int colStart, int rows, int cols) const
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, rowStart, colStart, rows, cols, offset))
    {
        return TileView<const T>(nullptr, 0, 0, 0);
    }
    const T *base = data_ + offset;
    return TileView<const T>(base, rows, cols, pitch_);
}

// Inline Implementation: TileView

template <typename T>
inline TileView<T>::TileView(T *data, int rows, int cols, size_t pitch)
    : data_(data)
    , rows_(rows)
    , cols_(cols)
    , pitch_(pitch == 0 ? static_cast<size_t>(cols) : pitch)
{
    if (data_ == nullptr || rows_ <= 0 || cols_ <= 0 || pitch_ < static_cast<size_t>(cols_))
    {
        data_ = nullptr;
        rows_ = 0;
        cols_ = 0;
        pitch_ = 0;
    }
}

template <typename T>
inline T &TileView<T>::at(int row, int col)
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, row, col, 1, 1, offset))
    {
        throw std::out_of_range("TileView::at index out of bounds");
    }
    return data_[offset];
}

template <typename T>
inline const T &TileView<T>::at(int row, int col) const
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, row, col, 1, 1, offset))
    {
        throw std::out_of_range("TileView::at index out of bounds");
    }
    return data_[offset];
}

template <typename T>
inline T *TileView<T>::row(int r)
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, r, 0, 1, cols_, offset))
    {
        return nullptr;
    }
    return data_ + offset;
}

template <typename T>
inline const T *TileView<T>::row(int r) const
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, r, 0, 1, cols_, offset))
    {
        return nullptr;
    }
    return data_ + offset;
}

template <typename T>
inline TileView<T> TileView<T>::subview(int rowStart, int colStart, int rows, int cols)
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, rowStart, colStart, rows, cols, offset))
    {
        return TileView<T>(nullptr, 0, 0, 0);
    }
    T *base = data_ + offset;
    return TileView<T>(base, rows, cols, pitch_);
}

template <typename T>
inline TileView<const T> TileView<T>::subview(int rowStart, int colStart, int rows, int cols) const
{
    size_t offset = 0;
    if (data_ == nullptr ||
        !detail::checkedTileRegion(rows_, cols_, pitch_, rowStart, colStart, rows, cols, offset))
    {
        return TileView<const T>(nullptr, 0, 0, 0);
    }
    const T *base = data_ + offset;
    return TileView<const T>(base, rows, cols, pitch_);
}

} // namespace nerve::gpu::tile
