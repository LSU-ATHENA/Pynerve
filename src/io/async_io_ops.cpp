#include "nerve/io/async_io.hpp"
#include "nerve/platform.hpp"

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nerve::io
{

namespace
{
IoBackend detectBestBackend()
{
#ifdef __linux__
    return IoBackend::Mmap;
#endif
#ifdef __APPLE__
    return IoBackend::DispatchIO;
#endif
    return IoBackend::Mmap;
}
} // namespace

Size preadFull(int fd, void *buffer, Size size, Size offset)
{
#ifndef _WIN32
    uint8_t *dst = static_cast<uint8_t *>(buffer);
    Size remaining = size;
    Size off = offset;
    while (remaining > 0)
    {
        ssize_t n = pread(fd, dst, remaining, static_cast<off_t>(off));
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return size - remaining;
        }
        if (n == 0)
            break;
        dst += n;
        remaining -= static_cast<Size>(n);
        off += static_cast<Size>(n);
    }
    return size - remaining;
#else
    (void)fd; (void)buffer; (void)size; (void)offset;
    return 0;
#endif
}

Size pwriteFull(int fd, const void *buffer, Size size, Size offset)
{
#ifndef _WIN32
    const uint8_t *src = static_cast<const uint8_t *>(buffer);
    Size remaining = size;
    Size off = offset;
    while (remaining > 0)
    {
        ssize_t n = pwrite(fd, src, remaining, static_cast<off_t>(off));
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return size - remaining;
        }
        if (n == 0)
            break;
        src += n;
        remaining -= static_cast<Size>(n);
        off += static_cast<Size>(n);
    }
    return size - remaining;
#else
    (void)fd; (void)buffer; (void)size; (void)offset;
    return 0;
#endif
}

std::unique_ptr<IoEngine> IoEngine::create(IoBackend backend)
{
    if (backend == IoBackend::Auto)
        backend = detectBestBackend();
    switch (backend)
    {
#ifdef __APPLE__
        case IoBackend::DispatchIO:
#endif
        case IoBackend::Mmap:
        default:
            return std::make_unique<MmapIoEngine>();
    }
}

MmapIoEngine::MmapIoEngine() = default;

Size MmapIoEngine::read(int fd, void *buffer, Size offset, Size size, IoFlags flags)
{
    if (size == 0)
        return 0;
    bool use_mmap = !(static_cast<uint32_t>(flags) & static_cast<uint32_t>(IoFlags::Direct));
    if (use_mmap && size >= kMmapChunkSize)
    {
        Size total = 0;
        while (total < size)
        {
            Size aligned_off = (offset + total) & ~(nerve::sys::page_size() - 1);
            Size map_size = std::min(static_cast<Size>(kMmapChunkSize),
                                     size - total + ((offset + total) - aligned_off));
            void *mapped = nerve::sys::map(nullptr, map_size, nerve::sys::MAP_PROT_READ,
                                           nerve::sys::MAP_FLAG_PRIVATE, fd, aligned_off);
            if (mapped == nerve::sys::kMapFailed)
            {
                return preadFull(fd, static_cast<uint8_t *>(buffer) + total, size - total,
                                 offset + total);
            }
            Size page_off = (offset + total) - aligned_off;
            Size copy_size = std::min(size - total, map_size - page_off);
            nerve::sys::advise(mapped, map_size, nerve::sys::MAP_ADV_SEQUENTIAL);
            std::memcpy(static_cast<uint8_t *>(buffer) + total,
                        static_cast<const uint8_t *>(mapped) + page_off, copy_size);
            nerve::sys::unmap(mapped, map_size);
            total += copy_size;
        }
        stats_.bytes_read += size;
        stats_.read_calls++;
        return size;
    }
    Size result = preadFull(fd, buffer, size, offset);
    stats_.bytes_read += result;
    stats_.read_calls++;
    return result;
}

Size MmapIoEngine::write(int fd, const void *buffer, Size offset, Size size, IoFlags)
{
    Size result = pwriteFull(fd, buffer, size, offset);
    stats_.bytes_written += result;
    stats_.write_calls++;
    return result;
}

void MmapIoEngine::prefetch(int fd, Size offset, Size size)
{
#ifdef __linux__
    posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(size),
                  POSIX_FADV_WILLNEED | POSIX_FADV_SEQUENTIAL);
#else
    (void)fd;
    (void)offset;
    (void)size;
#endif
}

struct AsyncFileReader::Impl
{
    int fd = -1;
    std::string path;
    IoBackend backend;
    std::unique_ptr<IoEngine> engine;

    explicit Impl(IoBackend b)
        : backend(b)
        , engine(IoEngine::create(b))
    {}
};

AsyncFileReader::AsyncFileReader(IoBackend backend)
    : impl_(std::make_unique<Impl>(backend))
{}

AsyncFileReader::~AsyncFileReader()
{
    close();
}

void AsyncFileReader::open(const std::string &path)
{
    impl_->path = path;
#ifdef __linux__
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (impl_->fd < 0)
    {
        impl_->fd = ::open(path.c_str(), O_RDONLY);
    }
#elif defined(__APPLE__)
    impl_->fd = ::open(path.c_str(), O_RDONLY);
#elif defined(_WIN32)
    impl_->fd = -1;
#else
    impl_->fd = ::open(path.c_str(), O_RDONLY);
#endif
    if (impl_->fd < 0)
    {
        throw std::runtime_error("Cannot open file: " + path + " (" + std::strerror(errno) + ")");
    }
#ifdef __linux__
    posix_fadvise(impl_->fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
#elif defined(__APPLE__)
    fcntl(impl_->fd, F_RDAHEAD, 1);
    fcntl(impl_->fd, F_NOCACHE, 1);
#endif
}

void AsyncFileReader::close()
{
    if (impl_->fd >= 0)
    {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

bool AsyncFileReader::isOpen() const noexcept
{
    return impl_->fd >= 0;
}

Size AsyncFileReader::read(void *buffer, Size offset, Size size)
{
    return impl_->engine->read(impl_->fd, buffer, offset, size);
}

Size AsyncFileReader::fileSize() const
{
    struct stat st;
    if (fstat(impl_->fd, &st) < 0)
        return 0;
    return static_cast<Size>(st.st_size);
}

IoStats AsyncFileReader::stats() const
{
    return impl_->engine->stats();
}

struct AsyncFileWriter::Impl
{
    int fd = -1;
    std::unique_ptr<IoEngine> engine;

    explicit Impl(IoBackend b)
        : engine(IoEngine::create(b))
    {}
};

AsyncFileWriter::AsyncFileWriter(IoBackend backend)
    : impl_(std::make_unique<Impl>(backend))
{}

AsyncFileWriter::~AsyncFileWriter()
{
    close();
}

void AsyncFileWriter::open(const std::string &path)
{
#ifndef _WIN32
    impl_->fd =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
#else
    impl_->fd = -1;
#endif
    if (impl_->fd < 0)
    {
        throw std::runtime_error("Cannot create file: " + path + " (" + std::strerror(errno) + ")");
    }
}

void AsyncFileWriter::close()
{
    if (impl_->fd >= 0)
    {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

bool AsyncFileWriter::isOpen() const noexcept
{
    return impl_->fd >= 0;
}

Size AsyncFileWriter::write(const void *buffer, Size offset, Size size)
{
    return impl_->engine->write(impl_->fd, buffer, offset, size);
}

void AsyncFileWriter::sync()
{
#ifndef _WIN32
    if (impl_->fd >= 0)
        fsync(impl_->fd);
#else
    (void)impl_;
#endif
}

Size AsyncFileWriter::fileSize() const
{
    struct stat st;
    if (fstat(impl_->fd, &st) < 0)
        return 0;
    return static_cast<Size>(st.st_size);
}

IoStats AsyncFileWriter::stats() const
{
    return impl_->engine->stats();
}

} // namespace nerve::io
