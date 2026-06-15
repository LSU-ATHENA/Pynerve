
#pragma once
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nerve::threading
{

// Thread-safe vector for concurrent collections
template <typename T>
class ConcurrentVector
{
private:
    std::vector<T> data_;
    mutable std::shared_mutex mutex_;

public:
    using value_type = T;
    using allocator_type = std::allocator<T>;
    using size_type = typename std::vector<T>::size_type;
    using difference_type = typename std::vector<T>::difference_type;
    using pointer = typename std::vector<T>::pointer;
    using const_pointer = typename std::vector<T>::const_pointer;

    class LockedReference
    {
    public:
        LockedReference(std::unique_lock<std::shared_mutex> &&lock, T *ptr) noexcept
            : lock_(std::move(lock))
            , ptr_(ptr)
        {}

        LockedReference(const LockedReference &) = delete;
        LockedReference &operator=(const LockedReference &) = delete;
        LockedReference(LockedReference &&) noexcept = default;
        LockedReference &operator=(LockedReference &&) noexcept = default;

        LockedReference &operator=(const T &value)
        {
            *ptr_ = value;
            return *this;
        }

        LockedReference &operator=(T &&value)
        {
            *ptr_ = std::move(value);
            return *this;
        }

        operator T() const
            requires std::copy_constructible<T>
        {
            return *ptr_;
        }

        T &get() noexcept { return *ptr_; }

        const T &get() const noexcept { return *ptr_; }

        T *operator->() noexcept { return ptr_; }

        const T *operator->() const noexcept { return ptr_; }

    private:
        std::unique_lock<std::shared_mutex> lock_;
        T *ptr_;
    };

    class ConstLockedReference
    {
    public:
        ConstLockedReference(std::shared_lock<std::shared_mutex> &&lock, const T *ptr) noexcept
            : lock_(std::move(lock))
            , ptr_(ptr)
        {}

        ConstLockedReference(const ConstLockedReference &) = delete;
        ConstLockedReference &operator=(const ConstLockedReference &) = delete;
        ConstLockedReference(ConstLockedReference &&) noexcept = default;
        ConstLockedReference &operator=(ConstLockedReference &&) noexcept = default;

        operator T() const
            requires std::copy_constructible<T>
        {
            return *ptr_;
        }

        const T &get() const noexcept { return *ptr_; }

        const T *operator->() const noexcept { return ptr_; }

    private:
        std::shared_lock<std::shared_mutex> lock_;
        const T *ptr_;
    };

    using reference = LockedReference;
    using const_reference = ConstLockedReference;

    // Constructors
    ConcurrentVector() = default;

    explicit ConcurrentVector(size_t count)
    {
        std::unique_lock lock(mutex_);
        data_.resize(count);
    }

    ConcurrentVector(size_t count, const T &value)
    {
        std::unique_lock lock(mutex_);
        data_.assign(count, value);
    }

    template <typename InputIt>
    ConcurrentVector(InputIt first, InputIt last)
    {
        std::unique_lock lock(mutex_);
        data_.assign(first, last);
    }

    ConcurrentVector(std::initializer_list<T> init)
    {
        std::unique_lock lock(mutex_);
        data_ = init;
    }

    // Copy constructor (deep copy with lock)
    ConcurrentVector(const ConcurrentVector &other)
    {
        std::shared_lock lock(other.mutex_);
        data_ = other.data_;
    }

    ConcurrentVector(ConcurrentVector &&other) noexcept
    {
        std::unique_lock lock(other.mutex_);
        data_ = std::move(other.data_);
    }

    // Assignment operators
    ConcurrentVector &operator=(const ConcurrentVector &other)
    {
        if (this != &other)
        {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::shared_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);
            data_ = other.data_;
        }
        return *this;
    }

    ConcurrentVector &operator=(ConcurrentVector &&other) noexcept
    {
        if (this != &other)
        {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::unique_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // Element access (thread-safe)
    reference at(size_type pos)
    {
        std::unique_lock lock(mutex_);
        T *ptr = &data_.at(pos);
        return reference(std::move(lock), ptr);
    }

    const_reference at(size_type pos) const
    {
        std::shared_lock lock(mutex_);
        const T *ptr = &data_.at(pos);
        return const_reference(std::move(lock), ptr);
    }

    reference operator[](size_type pos)
    {
        std::unique_lock lock(mutex_);
        T *ptr = &data_[pos];
        return reference(std::move(lock), ptr);
    }

    const_reference operator[](size_type pos) const
    {
        std::shared_lock lock(mutex_);
        const T *ptr = &data_[pos];
        return const_reference(std::move(lock), ptr);
    }

    reference front()
    {
        std::unique_lock lock(mutex_);
        if (data_.empty())
        {
            throw std::out_of_range("ConcurrentVector::front on empty vector");
        }
        T *ptr = &data_.front();
        return reference(std::move(lock), ptr);
    }

    const_reference front() const
    {
        std::shared_lock lock(mutex_);
        if (data_.empty())
        {
            throw std::out_of_range("ConcurrentVector::front on empty vector");
        }
        const T *ptr = &data_.front();
        return const_reference(std::move(lock), ptr);
    }

    reference back()
    {
        std::unique_lock lock(mutex_);
        if (data_.empty())
        {
            throw std::out_of_range("ConcurrentVector::back on empty vector");
        }
        T *ptr = &data_.back();
        return reference(std::move(lock), ptr);
    }

    const_reference back() const
    {
        std::shared_lock lock(mutex_);
        if (data_.empty())
        {
            throw std::out_of_range("ConcurrentVector::back on empty vector");
        }
        const T *ptr = &data_.back();
        return const_reference(std::move(lock), ptr);
    }

    // Modifiers (thread-safe)
    void push_back(const T &value)
    {
        std::unique_lock lock(mutex_);
        data_.push_back(value);
    }

    void push_back(T &&value)
    {
        std::unique_lock lock(mutex_);
        data_.push_back(std::move(value));
    }

    template <typename... Args>
    reference emplaceBack(Args &&...args)
    {
        std::unique_lock lock(mutex_);
        T &value = data_.emplace_back(std::forward<Args>(args)...);
        return reference(std::move(lock), &value);
    }

    void pop_back()
    {
        std::unique_lock lock(mutex_);
        if (data_.empty())
        {
            throw std::out_of_range("ConcurrentVector::pop_back on empty vector");
        }
        data_.pop_back();
    }

    void clear() noexcept
    {
        std::unique_lock lock(mutex_);
        data_.clear();
    }

    void resize(size_type count)
    {
        std::unique_lock lock(mutex_);
        data_.resize(count);
    }

    void resize(size_type count, const T &value)
    {
        std::unique_lock lock(mutex_);
        data_.resize(count, value);
    }

    void reserve(size_type new_cap)
    {
        std::unique_lock lock(mutex_);
        data_.reserve(new_cap);
    }

    void shrinkToFit()
    {
        std::unique_lock lock(mutex_);
        data_.shrinkToFit();
    }

    // Capacity (thread-safe)
    bool empty() const noexcept
    {
        std::shared_lock lock(mutex_);
        return data_.empty();
    }

    size_type size() const noexcept
    {
        std::shared_lock lock(mutex_);
        return data_.size();
    }

    size_type maxSize() const noexcept
    {
        std::shared_lock lock(mutex_);
        return data_.max_size();
    }

    size_type capacity() const noexcept
    {
        std::shared_lock lock(mutex_);
        return data_.capacity();
    }

    // Operations (thread-safe)
    void swap(ConcurrentVector &other) noexcept
    {
        if (this != &other)
        {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::unique_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);
            data_.swap(other.data_);
        }
    }

    // Copy operations (thread-safe)
    std::vector<T> copy() const
    {
        std::shared_lock lock(mutex_);
        return data_;
    }

    error::Result<std::vector<T>> safeCopy() const
    {
        try
        {
            std::shared_lock lock(mutex_);
            return error::Result<std::vector<T>>::ok(data_);
        }
        catch (const std::exception &e)
        {
            return error::Result<std::vector<T>>::err(
                error::TDAErrorCode::AllocationFailed,
                std::string("Failed to copy concurrent vector: ") + e.what());
        }
    }

    // Atomic operations for specific use cases
    bool tryPushBack(const T &value)
    {
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock)
        {
            return false;
        }
        data_.push_back(value);
        return true;
    }

    bool tryPushBack(T &&value)
    {
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock)
        {
            return false;
        }
        data_.push_back(std::move(value));
        return true;
    }

    // Iterator support (read-only)
    class constIterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T *;
        using reference = const T &;

        constIterator()
            : owner_(nullptr)
            , index_(0)
            , lock_(nullptr)
        {}

        reference operator*() const { return owner_->data_[index_]; }

        pointer operator->() const { return &owner_->data_[index_]; }

        constIterator &operator++()
        {
            ++index_;
            return *this;
        }

        constIterator operator++(int)
        {
            constIterator tmp = *this;
            ++index_;
            return tmp;
        }

        constIterator &operator--()
        {
            --index_;
            return *this;
        }

        constIterator operator--(int)
        {
            constIterator tmp = *this;
            --index_;
            return tmp;
        }

        bool operator==(const constIterator &other) const
        {
            return owner_ == other.owner_ && index_ == other.index_;
        }

        bool operator!=(const constIterator &other) const { return !(*this == other); }

        difference_type operator-(const constIterator &other) const
        {
            return static_cast<difference_type>(index_) -
                   static_cast<difference_type>(other.index_);
        }

        constIterator operator+(difference_type n) const
        {
            return constIterator(owner_, index_ + n, lock_);
        }

        constIterator operator-(difference_type n) const
        {
            return constIterator(owner_, index_ - n, lock_);
        }

    private:
        constIterator(const ConcurrentVector *owner, size_type index)
            : owner_(owner)
            , index_(index)
            , lock_(std::make_shared<std::shared_lock<std::shared_mutex>>(owner->mutex_))
        {}

        constIterator(const ConcurrentVector *owner, bool at_end)
            : owner_(owner)
            , index_(0)
            , lock_(std::make_shared<std::shared_lock<std::shared_mutex>>(owner->mutex_))
        {
            if (at_end)
            {
                index_ = owner_->data_.size();
            }
        }

        constIterator(const ConcurrentVector *owner, size_type index,
                      std::shared_ptr<std::shared_lock<std::shared_mutex>> lock)
            : owner_(owner)
            , index_(index)
            , lock_(std::move(lock))
        {}

        const ConcurrentVector *owner_;
        size_type index_;
        std::shared_ptr<std::shared_lock<std::shared_mutex>> lock_;

        friend class ConcurrentVector;
    };

    constIterator begin() const { return constIterator(this, size_type{0}); }

    constIterator end() const { return constIterator(this, true); }

    constIterator cbegin() const { return begin(); }
    constIterator cend() const { return end(); }
};

// Specialization for common types
using ConcurrentVectorInt = ConcurrentVector<int>;
using ConcurrentVectorDouble = ConcurrentVector<double>;
using ConcurrentVectorSize = ConcurrentVector<size_t>;

// Factory functions
template <typename T>
error::Result<ConcurrentVector<T>> makeConcurrentVector()
{
    try
    {
        return error::Result<ConcurrentVector<T>>::ok(ConcurrentVector<T>());
    }
    catch (const std::bad_alloc &)
    {
        return error::Result<ConcurrentVector<T>>::err(error::TDAErrorCode::AllocationFailed,
                                                       "Failed to allocate concurrent vector");
    }
}

template <typename T>
error::Result<ConcurrentVector<T>> makeConcurrentVector(size_t count)
{
    try
    {
        return error::Result<ConcurrentVector<T>>::ok(ConcurrentVector<T>(count));
    }
    catch (const std::bad_alloc &)
    {
        return error::Result<ConcurrentVector<T>>::err(
            error::TDAErrorCode::AllocationFailed,
            "Failed to allocate concurrent vector with count");
    }
}

template <typename T>
error::Result<ConcurrentVector<T>> makeConcurrentVector(size_t count, const T &value)
{
    try
    {
        return error::Result<ConcurrentVector<T>>::ok(ConcurrentVector<T>(count, value));
    }
    catch (const std::bad_alloc &)
    {
        return error::Result<ConcurrentVector<T>>::err(
            error::TDAErrorCode::AllocationFailed,
            "Failed to allocate concurrent vector with initial value");
    }
}

} // namespace nerve::threading
