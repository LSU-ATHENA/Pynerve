#pragma once
#include "nerve/core_types.hpp"
#include "nerve/io/diagram_io.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nerve::io
{

struct MmapFile
{
    void *data = nullptr;
    Size size = 0;
    int fd = -1;
    bool writable = false;

    MmapFile() = default;
    ~MmapFile();
    MmapFile(const MmapFile &) = delete;
    MmapFile &operator=(const MmapFile &) = delete;
    MmapFile(MmapFile &&other) noexcept;
    MmapFile &operator=(MmapFile &&other) noexcept;

    bool valid() const noexcept { return data != nullptr; }
    const uint8_t *bytes() const noexcept { return static_cast<const uint8_t *>(data); }
    uint8_t *mutableBytes() noexcept { return static_cast<uint8_t *>(data); }

    Size remaining(Size offset) const noexcept { return offset <= size ? size - offset : 0; }
};

MmapFile mmapReadFile(const std::string &path);
MmapFile mmapWriteFile(const std::string &path, Size file_size);

persistence::Diagram loadDiagramMmap(const std::string &path,
                                     DiagramFormat format = DiagramFormat::Binary);
void saveDiagramMmap(const std::string &path, const persistence::Diagram &diagram);

} // namespace nerve::io
