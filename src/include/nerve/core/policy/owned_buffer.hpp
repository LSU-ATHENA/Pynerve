#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace nerve::core
{

template <typename T>
class OwnedBuffer
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T *;
    using const_iterator = const T *;

    OwnedBuffer() noexcept
        : data_(nullptr)
        , size_(0)
        , capacity_(0)
    {}
    explicit OwnedBuffer(size_type size)
        : data_(size > 0 ? new T[size] : nullptr)
        , size_(size)
        , capacity_(size)
    {}
    OwnedBuffer(const std::vector<T> &vector)
        : data_(new T[vector.size()])
        , size_(vector.size())
        , capacity_(vector.size())
    {
        std::copy(vector.begin(), vector.end(), data_);
    }
    OwnedBuffer(const OwnedBuffer &other)
        : data_(other.size_ > 0 ? new T[other.size_] : nullptr)
        , size_(other.size_)
        , capacity_(other.size_)
    {
        std::copy(other.begin(), other.end(), data_);
    }
    OwnedBuffer(OwnedBuffer &&other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    ~OwnedBuffer() { delete[] data_; }

    OwnedBuffer &operator=(const OwnedBuffer &other)
    {
        if (this != &other)
        {
            delete[] data_;
            data_ = other.size_ > 0 ? new T[other.size_] : nullptr;
            size_ = other.size_;
            capacity_ = other.size_;
            std::copy(other.begin(), other.end(), data_);
        }
        return *this;
    }
    OwnedBuffer &operator=(OwnedBuffer &&other) noexcept
    {
        if (this != &other)
        {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    T &operator[](size_type index) noexcept { return data_[index]; }
    const T &operator[](size_type index) const noexcept { return data_[index]; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    iterator begin() noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + size_; }

    void resize(size_type new_size)
    {
        if (new_size > capacity_)
        {
            T *new_data = new T[new_size];
            std::copy(begin(), end(), new_data);
            delete[] data_;
            data_ = new_data;
            size_ = new_size;
            capacity_ = new_size;
        }
        else
        {
            size_ = new_size;
        }
    }
    void reserve(size_type new_capacity)
    {
        if (new_capacity > capacity_)
        {
            T *new_data = new T[new_capacity];
            const size_type copy_size = std::min(size_, new_capacity);
            std::copy(begin(), begin() + copy_size, new_data);
            delete[] data_;
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    void clear() noexcept { size_ = 0; }
    T *release() noexcept
    {
        T *result = data_;
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        return result;
    }

private:
    T *data_;
    size_type size_;
    size_type capacity_;
};

} // namespace nerve::core
