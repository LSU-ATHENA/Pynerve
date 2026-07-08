#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "nerve/io/diagram_io.hpp"
#include "nerve/io/mmap_io.hpp"
#include "nerve/platform.hpp"

namespace nerve::io
{

#if defined(__linux__) || defined(__APPLE__)
MmapFile::~MmapFile()
{
    if (data && data != nerve::sys::kMapFailed)
    {
        nerve::sys::unmap(data, size);
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
        if (data && data != nerve::sys::kMapFailed)
            nerve::sys::unmap(data, size);
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
    result.data = nerve::sys::map(nullptr, result.size, nerve::sys::MAP_PROT_READ,
                                     nerve::sys::MAP_FLAG_PRIVATE, result.fd, 0);
    if (result.data == nerve::sys::kMapFailed)
    {
        close(result.fd);
        result.fd = -1;
        throw std::runtime_error("mmap failed for: " + path + " (" + std::strerror(errno) + ")");
    }
    nerve::sys::advise(result.data, result.size, nerve::sys::MAP_ADV_SEQUENTIAL);
    nerve::sys::advise(result.data, result.size, nerve::sys::MAP_ADV_WILLNEED);
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
    result.data = nerve::sys::map(nullptr, file_size, nerve::sys::MAP_PROT_RW,
                                     nerve::sys::MAP_FLAG_SHARED, result.fd, 0);
    if (result.data == nerve::sys::kMapFailed)
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
    nerve::sys::sync_map(file.data, file.size);
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
