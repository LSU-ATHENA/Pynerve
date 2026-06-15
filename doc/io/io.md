# I/O

## Quick start

```python
import pynerve.io as nio

nio.save_diagram(diagram, "output.json", format="json")
nio.save_diagram(diagram, "output.txt", format="text")
nio.save_diagram(diagram, "output.bin", format="binary")

diagram = nio.load_diagram("output.json")

arr = nio.load_npy("points.npy")
```

All I/O operations are in `nerve::io` (C++) and `pynerve.io` (Python). Supports
persistence diagram serialization, NumPy NPY format, memory-mapped files, and
asynchronous I/O via io_uring / dispatch_io.


## Topics

- **[diagram_io.md](diagram_io.md)** -- Save/load diagrams in text, JSON, and binary formats
- **[npy.md](npy.md)** -- NPY format reader/writer, in-memory buffers
- **[mmap.md](mmap.md)** -- Memory-mapped file I/O, zero-copy reads, large diagram support
- **[async.md](async.md)** -- Async I/O engine, io_uring on Linux, dispatch_io on macOS


## Utility functions

```cpp
Size preadFull(int fd, void* buffer, Size size, Size offset);
Size pwriteFull(int fd, const void* buffer, Size size, Size offset);
```

These retry on `EINTR` and loop until all bytes are transferred or an error
occurs. Use these instead of raw `pread`/`pwrite` for reliability.

```cpp
// Example: reliable read from a file
std::vector<char> buffer(file_size);
preadFull(fd, buffer.data(), buffer.size(), 0);
```


### Python path resolution

```python
import pynerve.io as nio

# Paths are resolved relative to the working directory
nio.save_diagram(diagram, "diagrams/output.json")

# Use absolute paths for clarity
nio.save_diagram(diagram, "/data/experiments/run_001/diagram.json")

# Directory creation is NOT automatic - create dirs first
import os
os.makedirs("diagrams", exist_ok=True)
```


### Format auto-detection

When loading, the format is auto-detected from file extension:

The `.txt` extension maps to Text format. `.json` maps to JSON. `.bin` maps to Binary. `.npy` maps to NumPy. `.nvf` maps to FlatBuffers (from the serialization module).


### Complexity notes

Text diagram serialization costs O(pairs * avg_line_len) and produces human-readable output. Binary diagram serialization costs O(pairs) and is compact and fast. JSON diagram serialization costs O(pairs) and is interoperable. NPY load and save cost O(elements) and are NumPy compatible. Mmap file open is O(1) virtual memory mapping with zero-copy reads. Async read with io_uring is O(1) syscall submission with kernel-batched I/O. Async read via thread pool is O(1) context switch as a fallback path.

For streaming workloads (concurrent reads/writes of many small files), async
I/O reduces syscall overhead by batching. For single large files, mmap is
simpler and equally fast for sequential access.


### Common pitfalls

1. **Binary portability**: Binary format is little-endian. Loading on a
   big-endian system requires byte-swapping.

2. **JSON size**: JSON format is ~3x larger than binary. Not recommended
   for diagrams with >100k pairs.

3. **Text parsing**: Infinity marker `inf` is locale-independent in Pynerve
   but may cause issues when parsed by other tools.

4. **Missing directories**: Neither `save_diagram` nor `load_diagram`
   create parent directories. Use `os.makedirs` first.

5. **File locks**: No file locking is performed. Concurrent writes to the
   same file produce undefined results.



## FAQ

**Which format should I use for saving diagrams?** Use binary format for speed and exact IEEE 754 round-trips. Use JSON for interoperability with other tools. Use text format for human readability and debugging. For diagrams with more than 100,000 pairs, avoid JSON as it is roughly 3x larger than binary.

**When should I use mmap instead of regular I/O?** Use mmap for large files (hundreds of megabytes or more) where the zero-copy read path provides roughly 100x speedup over `loadDiagramFromFile`. For small files, the overhead of setting up the mapping is not justified.

**When should I use async I/O?** Use async I/O (io_uring on Linux) when reading or writing many small files concurrently, where batching syscalls reduces overhead. For single large files, mmap is simpler and equally fast.


### Cross-references

- `pynerve.io.diagram_io`: Diagram format details
- `pynerve.io.npy`: NPY format
- `pynerve.io.mmap`: Memory-mapped I/O
- `pynerve.io.async`: Asynchronous I/O engine
- `pynerve.serialization`: Schema-based serialization (FlatBuffers, Arrow)
