#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "nerve/io/diagram_io.hpp"
#include "nerve/io/mmap_io.hpp"

namespace nerve::io
{

#if defined(__linux__) || defined(__APPLE__)
MmapFile::~MmapFile()
{
    if (data && data != MAP_FAILED)
    {
        munmap(data, size);
    }
    if (fd >= 0)
    {
        close(fd);
    }
}

MmapFile::MmapFile(MmapFile &&other) noexcept
    : data(other.data)
    , size(other.size)
    , fd(other.fd)
    , writable(other.writable)
{
    other.data = nullptr;
    other.fd = -1;
    other.size = 0;
}

MmapFile &MmapFile::operator=(MmapFile &&other) noexcept
{
    if (this != &other)
    {
        if (data && data != MAP_FAILED)
            munmap(data, size);
        if (fd >= 0)
            close(fd);
        data = other.data;
        size = other.size;
        fd = other.fd;
        writable = other.writable;
        other.data = nullptr;
        other.fd = -1;
        other.size = 0;
    }
    return *this;
}

MmapFile mmapReadFile(const std::string &path)
{
    MmapFile result;
    result.fd = open(path.c_str(), O_RDONLY);
    if (result.fd < 0)
    {
        throw std::runtime_error("Cannot open file for mmap read: " + path + " (" +
                                 std::strerror(errno) + ")");
    }
    struct stat st;
    if (fstat(result.fd, &st) < 0)
    {
        close(result.fd);
        result.fd = -1;
        throw std::runtime_error("Cannot stat file: " + path + " (" + std::strerror(errno) + ")");
    }
    result.size = static_cast<Size>(st.st_size);
    if (result.size == 0)
    {
        close(result.fd);
        result.fd = -1;
        return result;
    }
    result.data = mmap(nullptr, result.size, PROT_READ, MAP_PRIVATE, result.fd, 0);
    if (result.data == MAP_FAILED)
    {
        close(result.fd);
        result.fd = -1;
        throw std::runtime_error("mmap failed for: " + path + " (" + std::strerror(errno) + ")");
    }
    madvise(result.data, result.size, MADV_SEQUENTIAL);
    madvise(result.data, result.size, MADV_WILLNEED);
    result.writable = false;
    return result;
}

MmapFile mmapWriteFile(const std::string &path, Size file_size)
{
    MmapFile result;
    result.fd =
        open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (result.fd < 0)
    {
        throw std::runtime_error("Cannot create file for mmap write: " + path + " (" +
                                 std::strerror(errno) + ")");
    }
    if (ftruncate(result.fd, static_cast<off_t>(file_size)) < 0)
    {
        close(result.fd);
        result.fd = -1;
        throw std::runtime_error("Cannot truncate file: " + path + " (" + std::strerror(errno) +
                                 ")");
    }
    result.data = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, result.fd, 0);
    if (result.data == MAP_FAILED)
    {
        close(result.fd);
        result.fd = -1;
        throw std::runtime_error("mmap write failed for: " + path + " (" + std::strerror(errno) +
                                 ")");
    }
    result.size = file_size;
    result.writable = true;
    return result;
}

persistence::Diagram loadDiagramMmap(const std::string &path, DiagramFormat format)
{
    auto file = mmapReadFile(path);
    if (!file.valid())
        return {};
    std::string data(reinterpret_cast<const char *>(file.bytes()), file.size);
    return deserializeDiagram(data, format);
}

void saveDiagramMmap(const std::string &path, const persistence::Diagram &diagram)
{
    auto data = serializeDiagramBinary(diagram);
    auto file = mmapWriteFile(path, data.size());
    std::memcpy(file.mutableBytes(), data.data(), data.size());
    msync(file.data, file.size, MS_SYNC);
}
#else
MmapFile::~MmapFile() {}
MmapFile::MmapFile(MmapFile &&) noexcept = default;
MmapFile &MmapFile::operator=(MmapFile &&) noexcept = default;

MmapFile mmapReadFile(const std::string &)
{
    throw std::runtime_error("mmap not supported on this platform");
}
MmapFile mmapWriteFile(const std::string &, Size)
{
    throw std::runtime_error("mmap not supported on this platform");
}
persistence::Diagram loadDiagramMmap(const std::string &, DiagramFormat)
{
    return {};
}
void saveDiagramMmap(const std::string &, const persistence::Diagram &) {}
#endif

} // namespace nerve::io
