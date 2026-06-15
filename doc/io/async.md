## Async I/O engine

```cpp
namespace nerve::io {

enum class IoBackend {
    Auto,
    Mmap,
    IoUring,
    DispatchIO,
    PosixAIO,
};

enum class IoFlags : uint32_t {
    None       = 0,
    Direct     = 1u << 0,
    Sequential = 1u << 1,
    WillNeed   = 1u << 2,
    NoCache    = 1u << 3,
};

struct IoStats {
    Size bytes_read = 0;
    Size bytes_written = 0;
    Size read_calls = 0;
    Size write_calls = 0;
    double cache_hit_rate = 0.0;
};

class IoEngine {
public:
    virtual ~IoEngine() = default;
    virtual IoBackend backend() const = 0;
    virtual IoStats stats() const = 0;

    virtual Size read(int fd, void* buffer, Size offset, Size size,
                      IoFlags flags = IoFlags::None) = 0;
    virtual Size write(int fd, const void* buffer, Size offset, Size size,
                       IoFlags flags = IoFlags::None) = 0;
    virtual bool supportsAsync() const noexcept;
    virtual void prefetch(int fd, Size offset, Size size);

    static std::unique_ptr<IoEngine> create(IoBackend backend = IoBackend::Auto);
};

}
```

### Backend selection

The `IoUring` backend requires Linux 5.1+ with liburing and provides true async with kernel submissions. `DispatchIO` requires macOS with Grand Central Dispatch and provides true async with dispatch queues. `PosixAIO` requires Linux with librt and uses a thread pool plus AIO. `Mmap` requires no special dependencies and provides synchronous access via mmap and memcpy.

`IoBackend::Auto` detects the best available backend:
1. Try io_uring on Linux 5.1+
2. Fall back to MmapIoEngine
3. If flags require async, use thread pool fallback

### io_uring details

When `NERVE_HAS_IO_URING` is defined (CMake: `find_package(uring)`), the
io_uring backend is used. It submits SQEs to the kernel ring buffer and
reaps completion events from the CQ ring.

**Configuration** (compile-time defaults):
- Queue depth: 256 entries
- Timeout: 5000 ms per submission
- Uses `IORING_SETUP_SQPOLL` for kernel-side polling (if available)

```cpp
// io_uring specific configuration
struct IoUringConfig {
    unsigned queue_depth = 256;
    unsigned timeout_ms = 5000;
    bool use_sqpoll = true;
    bool use_iopoll = false;
};
```

### Thread pool fallback

Without io_uring, async operations run on a thread pool using `pread`/`pwrite`.
Each call dispatches to the pool and returns a future. Not truly async at the
kernel level but avoids blocking the calling thread.

```cpp
class AsyncFileReader {
public:
    explicit AsyncFileReader(IoBackend backend = IoBackend::Auto);
    ~AsyncFileReader();

    void open(const std::string& path);
    void close();
    bool isOpen() const noexcept;

    Size read(void* buffer, Size offset, Size size);
    Size fileSize() const;
    IoStats stats() const;
};

class AsyncFileWriter {
public:
    explicit AsyncFileWriter(IoBackend backend = IoBackend::Auto);
    ~AsyncFileWriter();

    void open(const std::string& path);
    void close();
    bool isOpen() const noexcept;

    Size write(const void* buffer, Size offset, Size size);
    void sync();
    Size fileSize() const;
    IoStats stats() const;
};
```

### Python usage

```python
from pynerve.io import AsyncFileReader, AsyncFileWriter

reader = AsyncFileReader(backend="io_uring")
reader.open("large_diagram.bin")
data = reader.read(offset=0, size=reader.file_size())
print(f"Read {reader.stats().bytes_read} bytes")
reader.close()
```

### Performance comparison

For small random reads, io_uring achieves hundreds of thousands of IOPS with a few microseconds of latency. PosixAIO achieves hundreds of thousands of IOPS with tens of microseconds of latency. Mmap achieves around a hundred thousand IOPS with roughly ten microseconds of latency (dominated by page faults). The thread pool fallback achieves tens of thousands of IOPS with tens of microseconds of latency.

### Cross-references

- `pynerve.io.io`: I/O module overview
- `pynerve.io.mmap`: Memory-mapped alternatives
- `pynerve.core.thread_pool`: Thread pool used for fallback
