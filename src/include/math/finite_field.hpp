#pragma once
#include "nerve/error/error_registry.hpp"

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace nerve::math
{

namespace detail
{
constexpr bool isPrime(int n)
{
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (int i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
        {
            return false;
        }
    }

    return true;
}
} // namespace detail

// Template-based finite field implementation
template <int P>
class FiniteField
{
private:
    int value_;

    // Compile-time validation of prime characteristic
    static_assert(P > 1, "Field characteristic must be greater than 1");
    static constexpr bool is_prime_v = detail::isPrime(P);
    static_assert(is_prime_v, "Field characteristic must be prime");

public:
    static constexpr int characteristic = P;
    using value_type = int;

    // Constructors
    constexpr FiniteField()
        : value_(0)
    {}

    constexpr explicit FiniteField(int value)
        : value_(mod(value))
    {}

    // Copy and move
    constexpr FiniteField(const FiniteField &) = default;
    constexpr FiniteField(FiniteField &&) = default;
    constexpr FiniteField &operator=(const FiniteField &) = default;
    constexpr FiniteField &operator=(FiniteField &&) = default;

    // Field arithmetic operations
    constexpr FiniteField operator+(const FiniteField &other) const
    {
        return FiniteField(value_ + other.value_);
    }

    constexpr FiniteField operator-(const FiniteField &other) const
    {
        return FiniteField(value_ - other.value_);
    }

    constexpr FiniteField operator*(const FiniteField &other) const
    {
        return FiniteField(value_ * other.value_);
    }

    constexpr FiniteField operator/(const FiniteField &other) const
    {
        if (other.value_ == 0)
        {
            throw std::invalid_argument("Division by zero in finite field");
        }
        return FiniteField(value_ * inverse(other.value_));
    }

    // Unary operations
    constexpr FiniteField operator+() const { return *this; }

    constexpr FiniteField operator-() const { return FiniteField(-value_); }

    // Assignment operators
    constexpr FiniteField &operator+=(const FiniteField &other)
    {
        value_ = mod(value_ + other.value_);
        return *this;
    }

    constexpr FiniteField &operator-=(const FiniteField &other)
    {
        value_ = mod(value_ - other.value_);
        return *this;
    }

    constexpr FiniteField &operator*=(const FiniteField &other)
    {
        value_ = mod(value_ * other.value_);
        return *this;
    }

    constexpr FiniteField &operator/=(const FiniteField &other)
    {
        if (other.value_ == 0)
        {
            throw std::invalid_argument("Division by zero in finite field");
        }
        value_ = mod(value_ * inverse(other.value_));
        return *this;
    }

    // Comparison operations
    constexpr bool operator==(const FiniteField &other) const { return value_ == other.value_; }

    constexpr bool operator!=(const FiniteField &other) const { return value_ != other.value_; }

    constexpr bool operator<(const FiniteField &other) const { return value_ < other.value_; }

    constexpr bool operator<=(const FiniteField &other) const { return value_ <= other.value_; }

    constexpr bool operator>(const FiniteField &other) const { return value_ > other.value_; }

    constexpr bool operator>=(const FiniteField &other) const { return value_ >= other.value_; }

    // Utility functions
    [[nodiscard]] std::string to_string() const { return std::to_string(value_); }

    constexpr explicit operator int() const { return value_; }

    constexpr int toInt() const { return value_; }

    // Field properties
    static constexpr bool isZero(const FiniteField &x) { return x.value_ == 0; }

    static constexpr bool isOne(const FiniteField &x) { return x.value_ == 1; }

    static constexpr FiniteField zero() { return FiniteField(0); }

    static constexpr FiniteField one() { return FiniteField(1); }

    // Power operation
    constexpr FiniteField pow(int exponent) const
    {
        if (exponent < 0)
        {
            return pow(-exponent).inverse();
        }

        FiniteField result(1);
        FiniteField base(*this);

        while (exponent > 0)
        {
            if (exponent % 2 == 1)
            {
                result = result * base;
            }
            base = base * base;
            exponent /= 2;
        }

        return result;
    }

    // Multiplicative inverse
    constexpr FiniteField inverse() const
    {
        if (value_ == 0)
        {
            throw std::invalid_argument("Zero has no multiplicative inverse in finite field");
        }
        return FiniteField(inverse(value_));
    }

    // Additive inverse
    constexpr FiniteField additiveInverse() const { return FiniteField(-value_); }

    // Characteristic and order
    static constexpr int getCharacteristic() { return P; }

    static constexpr int getOrder() { return P; }

    // Check if element is a generator of multiplicative group
    constexpr bool isGenerator() const
    {
        if (value_ <= 1)
            return false;

        // Check if element^(P-1) = 1
        if (pow(P - 1) != one())
            return false;

        // Check if element^((P-1)/q) != 1 for all prime divisors q of P-1
        int n = P - 1;
        for (int q = 2; q * q <= n; ++q)
        {
            if (n % q == 0)
            {
                if (pow((P - 1) / q) == one())
                    return false;
                while (n % q == 0)
                    n /= q;
            }
        }
        if (n > 1 && pow((P - 1) / n) == one())
            return false;

        return true;
    }

    // Stream output
    friend std::ostream &operator<<(std::ostream &os, const FiniteField &field)
    {
        return os << field.value_ << " (mod " << P << ")";
    }

private:
    // Modular reduction
    static constexpr int mod(int x)
    {
        int r = x % P;
        return (r < 0) ? r + P : r;
    }

    // Modular inverse using extended Euclidean algorithm
    static constexpr int inverse(int a) { return modPow(a, P - 2); }

    // Modular exponentiation (compile-time friendly)
    static constexpr int modPow(int base, int exp)
    {
        int result = 1;
        base = mod(base);

        while (exp > 0)
        {
            if (exp % 2 == 1)
            {
                result = mod(result * base);
            }
            base = mod(base * base);
            exp /= 2;
        }

        return result;
    }
};

// Common field types
using Z2 = FiniteField<2>;
using Z3 = FiniteField<3>;
using Z5 = FiniteField<5>;
using Z7 = FiniteField<7>;
using Z11 = FiniteField<11>;
using Z13 = FiniteField<13>;
using Z17 = FiniteField<17>;
using Z19 = FiniteField<19>;
using Z23 = FiniteField<23>;
using Z29 = FiniteField<29>;
using Z31 = FiniteField<31>;

// Type traits for field properties
template <typename T>
struct is_finite_field : std::false_type
{};

template <int P>
struct is_finite_field<FiniteField<P>> : std::true_type
{
    static constexpr int characteristic = P;
};

template <typename T>
constexpr bool is_finite_field_v = is_finite_field<T>::value;

// Field operations utilities
namespace field_ops
{

template <typename Field>
constexpr Field zero()
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return Field::zero();
}

template <typename Field>
constexpr Field one()
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return Field::one();
}

template <typename Field>
constexpr Field additiveInverse(const Field &x)
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return x.additiveInverse();
}

template <typename Field>
constexpr Field multiplicativeInverse(const Field &x)
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return x.inverse();
}

template <typename Field>
constexpr Field pow(const Field &base, int exponent)
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return base.pow(exponent);
}

// Check if field element is zero or one
template <typename Field>
constexpr bool isZero(const Field &x)
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return Field::isZero(x);
}

template <typename Field>
constexpr bool isOne(const Field &x)
{
    static_assert(is_finite_field_v<Field>, "Type must be a finite field");
    return Field::isOne(x);
}

} // namespace field_ops

// Factory functions with error handling
template <int P>
error::Result<FiniteField<P>> makeFiniteField(int value)
{
    try
    {
        return error::Result<FiniteField<P>>::ok(FiniteField<P>(value));
    }
    catch (const std::exception &e)
    {
        return error::Result<FiniteField<P>>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create finite field element: ") + e.what());
    }
}

// Specialized factory functions for common fields
inline error::Result<Z2> makeZ2(int value)
{
    return makeFiniteField<2>(value);
}

inline error::Result<Z3> makeZ3(int value)
{
    return makeFiniteField<3>(value);
}

inline error::Result<Z5> makeZ5(int value)
{
    return makeFiniteField<5>(value);
}

inline error::Result<Z7> makeZ7(int value)
{
    return makeFiniteField<7>(value);
}

} // namespace nerve::math
