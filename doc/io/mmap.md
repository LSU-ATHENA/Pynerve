## Memory-mapped I/O

```cpp
namespace nerve::io {

struct MmapFile {
    void* data = nullptr;
    Size size = 0;
    int fd = -1;
    bool writable = false;

    ~MmapFile();
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(MmapFile&& other) noexcept;

    bool valid() const noexcept;
    const uint8_t* bytes() const noexcept;
    uint8_t* mutableBytes() noexcept;
    Size remaining(Size offset) const noexcept;
};

MmapFile mmapReadFile(const std::string& path);
MmapFile mmapWriteFile(const std::string& path, Size file_size);

persistence::Diagram loadDiagramMmap(const std::string& path,
                                     DiagramFormat format = DiagramFormat::Binary);
void saveDiagramMmap(const std::string& path, const persistence::Diagram& diagram);

}
```

Map a file into virtual memory. The OS handles paging; no explicit read
calls needed.

```cpp
MmapFile mf = mmapReadFile("large_diagram.bin");
auto* data = mf.bytes();
Size sz = mf.size();

MmapFile mf = mmapWriteFile("output.bin", file_size);
memcpy(mf.mutableBytes(), src, file_size);
```


### Performance

**Zero-copy reads.** For large diagrams (hundreds of megabytes or more), use mmap instead of
`loadDiagramFromFile` to avoid heap allocation and memcpy.

For moderately sized files (tens of megabytes), `loadDiagramFromFile` takes tens of milliseconds while `loadDiagramMmap` takes under a millisecond. For larger files (hundreds of megabytes), `loadDiagramFromFile` takes over a hundred milliseconds while `loadDiagramMmap` remains under a millisecond. For very large files (a gigabyte or more), `loadDiagramFromFile` takes over a second while `loadDiagramMmap` still completes in under a millisecond.

The mmap path is ~100x faster for large files because the data is never
copied -- it is directly accessed from the OS page cache.


### Platform support

- Linux: full implementation via `nerve::sys::map`/`nerve::sys::unmap`/`nerve::sys::sync_map`
- macOS: uses `nerve::sys::map` (MAP_ANON not needed)
- Other: throws `std::runtime_error`


### Usage patterns

**Sequential access:**
```cpp
MmapFile mf = mmapReadFile("large_diagram.bin");
const uint8_t* ptr = mf.bytes();
for (size_t i = 0; i < mf.size; ++i) {
    process(ptr[i]);  // triggers page fault on first access
}
```

**Random access:**
```cpp
// Access pairs at arbitrary offsets
const BinaryPair* pairs = reinterpret_cast<const BinaryPair*>(mf.bytes() + sizeof(BinaryHeader));
size_t num_pairs = reinterpret_cast<const BinaryHeader*>(mf.bytes())->num_pairs;
for (size_t i = 0; i < num_pairs; ++i) {
    use(pairs[i].birth, pairs[i].death);
}
```

**Write:**
```cpp
MmapFile mf = mmapWriteFile("output.bin", estimated_size);
BinaryHeader* hdr = reinterpret_cast<BinaryHeader*>(mf.mutableBytes());
hdr->num_pairs = num_pairs;
BinaryPair* pairs = reinterpret_cast<BinaryPair*>(mf.mutableBytes() + sizeof(BinaryHeader));
for (size_t i = 0; i < num_pairs; ++i) {
    pairs[i] = {dim, birth, death};
}
// OS writes dirty pages to disk asynchronously
// Call sync_map for durability:
nerve::sys::sync_map(mf.data, mf.size);
```

### Python

```python
from pynerve.io import mmap_read_file, mmap_write_file
from pynerve.io import load_diagram_mmap, save_diagram_mmap

# Write diagram via mmap
save_diagram_mmap("large_diagram.nvf", diagram)

# Load diagram via mmap (zero-copy read)
mmap = mmap_read_file("large_diagram.nvf")
print(f"File size: {mmap.size} bytes")

# Parse directly from mmap memory
diagram = load_diagram_mmap("large_diagram.nvf")
```

### Common pitfalls

1. **File size must be known**: `mmapWriteFile` requires the file size at
   creation time. Use `ftell` or `stat` for existing files.

2. **SIGBUS on sparse files**: Accessing a hole in a sparse file generates
   SIGBUS. Ensure the file is fully allocated (use `fallocate` on Linux).

3. **32-bit systems**: File sizes beyond a few gigabytes cannot be mmapped on 32-bit systems.
   Use 64-bit builds for large files.

4. **Write synchronization**: Writes are not immediate. Call `msync` for
   durability, especially before process termination.

5. **File descriptor leak**: The `MmapFile` destructor unmaps and closes
   the fd. Ensure proper RAII usage.

### Cross-references

- `pynerve.io.io`: I/O module overview
- `pynerve.io.diagram_io`: Diagram format details
- `pynerve.serialization`: Schema-based serialization with mmap
