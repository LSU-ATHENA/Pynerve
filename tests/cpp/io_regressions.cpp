#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/io/npy_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Size;

constexpr double kTol = 1e-10;

bool check_npy_roundtrip()
{
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<Size> shape = {2, 3};

    auto arr = nerve::io::makeNpyArray(data, shape);

    if (arr.header.dtype != nerve::io::NpyDataType::Float64)
    {
        std::cerr << "wrong dtype\n";
        return false;
    }
    if (arr.header.shape.size() != shape.size())
    {
        std::cerr << "shape rank mismatch\n";
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (arr.header.shape[i] != shape[i])
        {
            std::cerr << "shape dim " << i << " mismatch\n";
            return false;
        }
    }
    if (arr.header.fortran_order)
    {
        std::cerr << "fortran order should be false\n";
        return false;
    }

    auto buf = nerve::io::saveNpyToMemory(arr);
    if (buf.empty())
    {
        std::cerr << "serialized buffer empty\n";
        return false;
    }

    auto loaded = nerve::io::loadNpyFromMemory(buf);
    if (loaded.header.dtype != arr.header.dtype)
    {
        std::cerr << "roundtrip dtype mismatch\n";
        return false;
    }
    if (loaded.header.shape != arr.header.shape)
    {
        std::cerr << "roundtrip shape mismatch\n";
        return false;
    }
    if (loaded.data.size() != arr.data.size())
    {
        std::cerr << "roundtrip data size mismatch\n";
        return false;
    }

    return true;
}

bool check_npy_dtype_conversion()
{
    if (nerve::io::npyDtypeFor<float>() != nerve::io::NpyDataType::Float32)
    {
        std::cerr << "float->Float32 failed\n";
        return false;
    }
    if (nerve::io::npyDtypeFor<double>() != nerve::io::NpyDataType::Float64)
    {
        std::cerr << "double->Float64 failed\n";
        return false;
    }
    if (nerve::io::npyDtypeFor<int32_t>() != nerve::io::NpyDataType::Int32)
    {
        std::cerr << "int32_t->Int32 failed\n";
        return false;
    }

    std::string f32_str = nerve::io::npyDtypeToString(nerve::io::NpyDataType::Float32);
    std::string f64_str = nerve::io::npyDtypeToString(nerve::io::NpyDataType::Float64);

    auto f32_back = nerve::io::npyDtypeFromString(f32_str);
    auto f64_back = nerve::io::npyDtypeFromString(f64_str);

    if (f32_back != nerve::io::NpyDataType::Float32)
    {
        std::cerr << "string roundtrip Float32 failed\n";
        return false;
    }
    if (f64_back != nerve::io::NpyDataType::Float64)
    {
        std::cerr << "string roundtrip Float64 failed\n";
        return false;
    }

    return true;
}

bool check_npy_dtype_size()
{
    if (nerve::io::npyDtypeSize(nerve::io::NpyDataType::Float32) != 4)
    {
        std::cerr << "Float32 size wrong\n";
        return false;
    }
    if (nerve::io::npyDtypeSize(nerve::io::NpyDataType::Float64) != 8)
    {
        std::cerr << "Float64 size wrong\n";
        return false;
    }
    if (nerve::io::npyDtypeSize(nerve::io::NpyDataType::Int32) != 4)
    {
        std::cerr << "Int32 size wrong\n";
        return false;
    }
    return true;
}

bool check_npy_array_view()
{
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto arr = nerve::io::makeNpyArray(data, {2, 2});

    const float *ptr = arr.as<float>();
    if (!ptr)
    {
        std::cerr << "null view pointer\n";
        return false;
    }
    if (arr.count<float>() != data.size())
    {
        std::cerr << "element count mismatch\n";
        return false;
    }
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (std::abs(ptr[i] - data[i]) > 1e-7f)
        {
            std::cerr << "data mismatch at " << i << "\n";
            return false;
        }
    }

    return true;
}

bool check_malformed_npy_rejected()
{
    std::vector<uint8_t> garbage = {0, 1, 2, 3, 4, 5};
    bool threw = false;
    try
    {
        auto arr = nerve::io::loadNpyFromMemory(garbage);
        static_cast<void>(arr);
    }
    catch (...)
    {
        threw = true;
    }
    if (!threw)
    {
        std::cerr << "malformed NPY should throw\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_npy_roundtrip())
    {
        std::cerr << "FAIL: npy roundtrip\n";
        return 1;
    }
    if (!check_npy_dtype_conversion())
    {
        std::cerr << "FAIL: npy dtype conversion\n";
        return 1;
    }
    if (!check_npy_dtype_size())
    {
        std::cerr << "FAIL: npy dtype size\n";
        return 1;
    }
    if (!check_npy_array_view())
    {
        std::cerr << "FAIL: npy array view\n";
        return 1;
    }
    if (!check_malformed_npy_rejected())
    {
        std::cerr << "FAIL: malformed npy rejected\n";
        return 1;
    }
    return 0;
}
