## NPY I/O

```cpp
namespace nerve::io {

enum class NpyDataType : uint8_t {
    Float32 = 0,
    Float64 = 1,
    Int32   = 2,
    Int64   = 3,
    Uint32  = 4,
    Uint64  = 5,
};

struct NpyHeader {
    NpyDataType dtype = NpyDataType::Float64;
    std::vector<Size> shape;
    bool fortran_order = false;
};

struct NpyArray {
    NpyHeader header;
    std::vector<uint8_t> data;

    template<typename T>
    const T* as() const;

    template<typename T>
    Size count() const;
};

NpyArray loadNpy(const std::string& path);
void saveNpy(const std::string& path, const NpyArray& array);

NpyArray loadNpyFromMemory(const std::vector<uint8_t>& buffer);
std::vector<uint8_t> saveNpyToMemory(const NpyArray& array);

template<typename T>
NpyArray makeNpyArray(const std::vector<T>& values,
                       const std::vector<Size>& shape);

}
```


### NPY format (NumPy v1.0)

- Magic string: `\x93NUMPY`
- Version: 1 byte major, 1 byte minor
- Header length: 2 bytes (little-endian)
- Header: Python dict literal as ASCII (dtype, shape, fortran_order)
- Data: raw binary, row-major (C order) by default

**Header example:**
```
{'descr': '<f8', 'fortran_order': False, 'shape': (100, 3), }
```

The header is a valid Python dict literal parsed at load time.


### Supported dtypes

The supported dtypes are: Float32 (NumPy float32, C float, 4 bytes), Float64 (NumPy float64, C double, 8 bytes), Int32 (NumPy int32, C int32_t, 4 bytes), Int64 (NumPy int64, C int64_t, 8 bytes), Uint32 (NumPy uint32, C uint32_t, 4 bytes), and Uint64 (NumPy uint64, C uint64_t, 8 bytes).


### Usage

```cpp
std::vector<double> data(300);
NpyArray arr = makeNpyArray(data, {100, 3});
saveNpy("points.npy", arr);

NpyArray loaded = loadNpy("points.npy");
const double* ptr = loaded.as<double>();
Size count = loaded.count<double>();  // 300
```

### Python

```python
import pynerve.io as nio

arr = nio.load_npy("points.npy")
# arr is a numpy-compatible array loader

# Save from numpy
import numpy as np
data = np.random.randn(100, 3).astype(np.float32)
nio.save_npy("points.npy", data)
```

### Limitations

- Only NPY v1.0
- Fortran-order arrays are read but silently converted to row-major in memory
- Complex dtypes not supported
- Object arrays and structured dtypes not supported

### Cross-references

- `pynerve.io.io`: I/O module overview
- `pynerve.serialization.arrow`: Arrow columnar format (alternative for large arrays)
