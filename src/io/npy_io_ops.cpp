#include "nerve/io/npy_io.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace nerve::io
{

static constexpr uint8_t kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
static constexpr Size kMagicSize = 6;
static constexpr Size kVersionMajor = 1;
static constexpr Size kVersionMinor = 0;
static constexpr Size kMaxHeaderSize = 65536;

std::string npyDtypeToString(NpyDataType dtype)
{
    switch (dtype)
    {
        case NpyDataType::Float32:
            return "<f4";
        case NpyDataType::Float64:
            return "<f8";
        case NpyDataType::Int32:
            return "<i4";
        case NpyDataType::Int64:
            return "<i8";
        case NpyDataType::Uint32:
            return "<u4";
        case NpyDataType::Uint64:
            return "<u8";
    }
    return "<f8";
}

NpyDataType npyDtypeFromString(const std::string &s)
{
    if (s == "<f4")
        return NpyDataType::Float32;
    if (s == "<f8")
        return NpyDataType::Float64;
    if (s == "<i4")
        return NpyDataType::Int32;
    if (s == "<i8")
        return NpyDataType::Int64;
    if (s == "<u4")
        return NpyDataType::Uint32;
    if (s == "<u8")
        return NpyDataType::Uint64;
    if (s == ">f4" || s == ">f8" || s == ">i4" || s == ">i8" || s == ">u4" || s == ">u8")
    {
        throw std::runtime_error("Big-endian NPY files are not supported");
    }
    throw std::runtime_error("Unknown NPY dtype: " + s);
}

Size npyDtypeSize(NpyDataType dtype)
{
    switch (dtype)
    {
        case NpyDataType::Float32:
        case NpyDataType::Int32:
        case NpyDataType::Uint32:
            return 4;
        case NpyDataType::Float64:
        case NpyDataType::Int64:
        case NpyDataType::Uint64:
            return 8;
    }
    return 8;
}

static std::string buildHeader(const NpyHeader &header)
{
    std::ostringstream os;
    os << "{'descr': '" << npyDtypeToString(header.dtype) << "', ";
    os << "'fortran_order': " << (header.fortran_order ? "True" : "False") << ", ";
    os << "'shape': (";
    for (Size i = 0; i < header.shape.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << header.shape[i];
    }
    os << "), }";
    std::string dict = os.str();
    while (dict.size() % 64 != 63)
        dict += ' ';
    dict += '\n';
    uint16_t header_len = static_cast<uint16_t>(dict.size());
    std::string header_str;
    header_str.resize(kMagicSize + 4 + dict.size());
    std::memcpy(header_str.data(), kMagic, kMagicSize);
    header_str[kMagicSize] = kVersionMajor;
    header_str[kMagicSize + 1] = kVersionMinor;
    std::memcpy(header_str.data() + kMagicSize + 2, &header_len, sizeof(uint16_t));
    std::memcpy(header_str.data() + kMagicSize + 4, dict.data(), dict.size());
    return header_str;
}

static NpyHeader parseHeader(const std::string &header_dict)
{
    NpyHeader header;
    static std::regex descr_re(R"('descr'\s*:\s*'([^']+)')");
    static std::regex fortran_re(R"('fortran_order'\s*:\s*(True|False))");
    static std::regex shape_re(R"('shape'\s*:\s*\(\s*(.*?)\s*\))");
    std::smatch m;
    if (std::regex_search(header_dict, m, descr_re))
    {
        header.dtype = npyDtypeFromString(m[1].str());
    }
    if (std::regex_search(header_dict, m, fortran_re))
    {
        header.fortran_order = (m[1].str() == "True");
    }
    if (std::regex_search(header_dict, m, shape_re))
    {
        std::string shape_str = m[1].str();
        std::istringstream is(shape_str);
        std::string token;
        while (std::getline(is, token, ','))
        {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            if (!token.empty())
            {
                header.shape.push_back(static_cast<Size>(std::stoull(token)));
            }
        }
    }
    return header;
}

NpyArray loadNpyFromMemory(const std::vector<uint8_t> &buffer)
{
    if (buffer.size() < kMagicSize + 4)
    {
        throw std::runtime_error("NPY buffer too small for header");
    }
    if (std::memcmp(buffer.data(), kMagic, kMagicSize) != 0)
    {
        throw std::runtime_error("Invalid NPY magic number");
    }
    uint16_t header_len;
    std::memcpy(&header_len, buffer.data() + kMagicSize + 2, sizeof(uint16_t));
    if (kMagicSize + 4 + header_len > buffer.size())
    {
        throw std::runtime_error("NPY header length exceeds buffer size");
    }
    std::string header_dict(reinterpret_cast<const char *>(buffer.data() + kMagicSize + 4),
                            header_len);
    header_dict.resize(header_dict.find('\n'));
    NpyHeader header = parseHeader(header_dict);
    Size total_elements = 1;
    for (Size s : header.shape)
        total_elements *= s;
    Size data_size = total_elements * npyDtypeSize(header.dtype);
    Size data_offset = kMagicSize + 4 + header_len;
    if (data_offset + data_size > buffer.size())
    {
        throw std::runtime_error("NPY data size exceeds buffer size");
    }
    NpyArray arr;
    arr.header = std::move(header);
    arr.data.assign(buffer.begin() + static_cast<std::ptrdiff_t>(data_offset),
                    buffer.begin() + static_cast<std::ptrdiff_t>(data_offset + data_size));
    return arr;
}

std::vector<uint8_t> saveNpyToMemory(const NpyArray &array)
{
    std::string hdr = buildHeader(array.header);
    std::vector<uint8_t> buf;
    buf.resize(hdr.size() + array.data.size());
    std::memcpy(buf.data(), hdr.data(), hdr.size());
    std::memcpy(buf.data() + hdr.size(), array.data.data(), array.data.size());
    return buf;
}

void saveNpy(const std::string &path, const NpyArray &array)
{
    auto buf = saveNpyToMemory(array);
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file for writing: " + path);
    file.write(reinterpret_cast<const char *>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
}

NpyArray loadNpy(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open file for reading: " + path);
    auto size = static_cast<Size>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
    return loadNpyFromMemory(buffer);
}

} // namespace nerve::io
