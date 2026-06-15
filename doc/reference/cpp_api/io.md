# `nerve::io` -- Input/Output

```cpp
#include <nerve/io/npy_reader.hpp>
#include <nerve/io/hdf5_reader.hpp>

namespace nerve::io;

class NpyReader {
public:
    explicit NpyReader(const std::string& path);

    std::vector<double> readAll();
    std::vector<double> readChunk(size_t offset, size_t count);
    std::vector<size_t> shape() const;
    size_t numElements() const;
    size_t elementSize() const;
};

class Hdf5Reader {
public:
    explicit Hdf5Reader(const std::string& path);

    std::vector<double> readDataset(const std::string& name);
    std::vector<double> readChunk(const std::string& name,
                                   size_t offset, size_t count);
    bool hasDataset(const std::string& name) const;
    std::vector<std::string> datasetNames() const;
};

class AsyncReader {
public:
    AsyncReader(const std::string& path, size_t buffer_size = 1ULL << 20);

    // Start async read; returns future
    std::future<std::vector<double>> readAsync(size_t offset, size_t count);

    // Synchronous wait for all pending reads
    void waitAll();
};
```

**Cost (NpyReader::readAll):** O(n) to read + parse header.

**Cost (Hdf5Reader::readChunk):** O(chunk_size) with HDF5 chunk cache.

<- [C++ API Overview](index.md)
