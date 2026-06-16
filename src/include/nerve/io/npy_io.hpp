#pragma once
#include "nerve/core_types.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nerve::io
{

enum class NpyDataType : uint8_t
{
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    Uint32 = 4,
    Uint64 = 5,
};

struct NpyHeader
{
    NpyDataType dtype = NpyDataType::Float64;
    std::vector<Size> shape;
    bool fortran_order = false;
};

struct NpyArray
{
    NpyHeader header;
    std::vector<uint8_t> data;

    template <typename T>
    const T *as() const
    {
        return reinterpret_cast<const T *>(data.data());
    }

    template <typename T>
    Size count() const
    {
        return data.size() / sizeof(T);
    }
};

NpyArray loadNpy(const std::string &path);
void saveNpy(const std::string &path, const NpyArray &array);
NpyArray loadNpyFromMemory(const std::vector<uint8_t> &buffer);
std::vector<uint8_t> saveNpyToMemory(const NpyArray &array);

std::string npyDtypeToString(NpyDataType dtype);
NpyDataType npyDtypeFromString(const std::string &s);
Size npyDtypeSize(NpyDataType dtype);

template <typename T>
NpyDataType npyDtypeFor()
{
    if constexpr (std::is_same_v<T, float>)
        return NpyDataType::Float32;
    else if constexpr (std::is_same_v<T, double>)
        return NpyDataType::Float64;
    else if constexpr (std::is_same_v<T, int32_t>)
        return NpyDataType::Int32;
    else if constexpr (std::is_same_v<T, int64_t>)
        return NpyDataType::Int64;
    else if constexpr (std::is_same_v<T, uint32_t>)
        return NpyDataType::Uint32;
    else if constexpr (std::is_same_v<T, uint64_t>)
        return NpyDataType::Uint64;
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported type for NPY I/O");
        return NpyDataType::Float64;
    }
}

template <typename T>
NpyArray makeNpyArray(const std::vector<T> &values, const std::vector<Size> &shape)
{
    NpyArray arr;
    arr.header.dtype = npyDtypeFor<T>();
    arr.header.shape = shape;
    arr.header.fortran_order = false;
    arr.data.resize(values.size() * sizeof(T));
    std::memcpy(arr.data.data(), values.data(), values.size() * sizeof(T));
    return arr;
}

} // namespace nerve::io
