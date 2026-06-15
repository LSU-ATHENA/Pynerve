#pragma once
#include "nerve/core_types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nerve::io
{

enum class IoBackend
{
    Auto,
    Mmap,
    IoUring,
    DispatchIO,
    PosixAIO,
};

enum class IoFlags : uint32_t
{
    None = 0,
    Direct = 1u << 0,
    Sequential = 1u << 1,
    WillNeed = 1u << 2,
    NoCache = 1u << 3,
};

struct AsyncReadRequest
{
    int fd = -1;
    void *buffer = nullptr;
    Size offset = 0;
    Size size = 0;
};

struct AsyncReadResult
{
    void *buffer = nullptr;
    Size bytes_read = 0;
    int error_code = 0;
};

struct IoStats
{
    Size bytes_read = 0;
    Size bytes_written = 0;
    Size read_calls = 0;
    Size write_calls = 0;
    double cache_hit_rate = 0.0;
};

class IoEngine
{
public:
    virtual ~IoEngine() = default;
    virtual IoBackend backend() const = 0;
    virtual IoStats stats() const = 0;

    virtual Size read(int fd, void *buffer, Size offset, Size size,
                      IoFlags flags = IoFlags::None) = 0;
    virtual Size write(int fd, const void *buffer, Size offset, Size size,
                       IoFlags flags = IoFlags::None) = 0;
    virtual bool supportsAsync() const noexcept { return false; }
    virtual void prefetch(int fd, Size offset, Size size)
    {
        (void)fd;
        (void)offset;
        (void)size;
    }

    static std::unique_ptr<IoEngine> create(IoBackend backend = IoBackend::Auto);
};

class MmapIoEngine : public IoEngine
{
public:
    MmapIoEngine();
    ~MmapIoEngine() override = default;
    IoBackend backend() const override { return IoBackend::Mmap; }
    IoStats stats() const override { return stats_; }

    Size read(int fd, void *buffer, Size offset, Size size, IoFlags flags = IoFlags::None) override;
    Size write(int fd, const void *buffer, Size offset, Size size,
               IoFlags flags = IoFlags::None) override;
    void prefetch(int fd, Size offset, Size size) override;

private:
    IoStats stats_;
    static constexpr Size kMmapChunkSize = 64 * 1024 * 1024;
};

class AsyncFileReader
{
public:
    explicit AsyncFileReader(IoBackend backend = IoBackend::Auto);
    ~AsyncFileReader();

    void open(const std::string &path);
    void close();
    bool isOpen() const noexcept;

    Size read(void *buffer, Size offset, Size size);
    Size fileSize() const;
    IoStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AsyncFileWriter
{
public:
    explicit AsyncFileWriter(IoBackend backend = IoBackend::Auto);
    ~AsyncFileWriter();

    void open(const std::string &path);
    void close();
    bool isOpen() const noexcept;

    Size write(const void *buffer, Size offset, Size size);
    void sync();
    Size fileSize() const;
    IoStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Size preadFull(int fd, void *buffer, Size size, Size offset);
Size pwriteFull(int fd, const void *buffer, Size size, Size offset);

} // namespace nerve::io
