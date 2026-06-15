
#pragma once

#include "nerve/errors/detail/error_result.hpp"

#include <cstdint>
#include <stdexcept>

namespace nerve::math
{

template <typename T = uint64_t>
class FiniteField
{
public:
    static errors::ErrorResult<FiniteField<T>> create(T characteristic)
    {
        if (!isPrime(characteristic))
        {
            return errors::ErrorResult<FiniteField<T>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                              "Characteristic must be prime");
        }

        return errors::ErrorResult<FiniteField<T>>::ok(FiniteField(characteristic));
    }

    T add(T a, T b) const { return (a + b) % characteristic_; }

    T subtract(T a, T b) const { return (a + characteristic_ - b) % characteristic_; }

    T multiply(T a, T b) const { return (a * b) % characteristic_; }

    errors::ErrorResult<T> divide(T a, T b) const
    {
        auto inv_result = inverse(b);
        if (inv_result.isError())
        {
            return errors::ErrorResult<T>::error(inv_result.error().code,
                                                 inv_result.error().message);
        }
        return errors::ErrorResult<T>::ok(multiply(a, inv_result.value()));
    }

    errors::ErrorResult<T> inverse(T a) const
    {
        if (a == 0)
        {
            return errors::ErrorResult<T>::error(errors::ErrorCode::E50_PH_ABORT,
                                                 "Zero has no multiplicative inverse");
        }

        // Use extended Euclidean algorithm
        return extendedGcd(a, characteristic_);
    }

    T pow(T base, T exponent) const
    {
        T result = 1;
        T current = base % characteristic_;

        while (exponent > 0)
        {
            if (exponent % 2 == 1)
            {
                result = multiply(result, current);
            }
            current = multiply(current, current);
            exponent /= 2;
        }

        return result;
    }

    T getCharacteristic() const { return characteristic_; }

    T getZero() const { return 0; }

    T getOne() const { return 1; }

    bool isZero(T a) const { return a == 0; }

    bool isOne(T a) const { return a == 1; }

private:
    explicit FiniteField(T characteristic)
        : characteristic_(characteristic)
    {}

    static bool isPrime(T n)
    {
        if (n <= 1)
            return false;
        if (n <= 3)
            return true;
        if (n % 2 == 0 || n % 3 == 0)
            return false;

        for (T i = 5; i * i <= n; i += 6)
        {
            if (n % i == 0 || n % (i + 2) == 0)
            {
                return false;
            }
        }
        return true;
    }

    errors::ErrorResult<T> extendedGcd(T a, T b) const
    {
        // Extended Euclidean algorithm to find inverse of a mod b
        T old_r = a, r = b;
        T old_s = 1, s = 0;
        T old_t = 0, t = 1;

        while (r != 0)
        {
            T quotient = old_r / r;

            T temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            T temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;

            T temp_t = t;
            t = old_t - quotient * t;
            old_t = temp_t;
        }

        // old_r is the GCD
        if (old_r != 1)
        {
            return errors::ErrorResult<T>::error(errors::ErrorCode::E50_PH_ABORT,
                                                 "Inverse does not exist");
        }

        // old_s is the modular inverse
        T inverse = old_s;
        if (inverse < 0)
        {
            inverse += b;
        }

        return errors::ErrorResult<T>::ok(inverse);
    }

    T characteristic_;
};

// Common finite field types
using Z2 = FiniteField<uint64_t>; // Field with characteristic 2
using Z3 = FiniteField<uint64_t>; // Field with characteristic 3
using Z5 = FiniteField<uint64_t>; // Field with characteristic 5

// Factory functions for common fields
inline errors::ErrorResult<Z2> createZ2()
{
    return Z2::create(2);
}

inline errors::ErrorResult<Z3> createZ3()
{
    return Z3::create(3);
}

inline errors::ErrorResult<Z5> createZ5()
{
    return Z5::create(5);
}

} // namespace nerve::math
