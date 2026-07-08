#include "nerve/io/diagram_io.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nerve::io
{

static std::string pairToText(const persistence::Pair &p)
{
    std::ostringstream os;
    os << p.dimension << ' ' << p.birth << ' ' << p.death;
    return os.str();
}

static persistence::Pair pairFromText(const std::string &line)
{
    std::istringstream is(line);
    int dim;
    double birth, death;
    is >> dim >> birth >> death;
    return persistence::Pair{birth, death, static_cast<Dimension>(dim)};
}

std::string serializeDiagram(const persistence::Diagram &diagram, DiagramFormat format)
{
    const auto &pairs = diagram.getPairs();
    switch (format)
    {
        case DiagramFormat::Text:
        {
            std::ostringstream os;
            os << pairs.size() << '\n';
            for (const auto &p : pairs)
                os << pairToText(p) << '\n';
            return os.str();
        }
        case DiagramFormat::Json:
            return diagramToJson(diagram);
        case DiagramFormat::Binary:
        {
            auto buf = serializeDiagramBinary(diagram);
            return std::string(reinterpret_cast<const char *>(buf.data()), buf.size());
        }
    }
    return {};
}

persistence::Diagram deserializeDiagram(const std::string &data, DiagramFormat format)
{
    switch (format)
    {
        case DiagramFormat::Text:
        {
            std::istringstream is(data);
            Size count;
            is >> count;
            std::vector<persistence::Pair> pairs;
            pairs.reserve(count);
            std::string line;
            std::getline(is, line);
            for (Size i = 0; i < count; ++i)
            {
                std::getline(is, line);
                if (line.empty())
                    continue;
                pairs.push_back(pairFromText(line));
            }
            return persistence::Diagram(std::move(pairs));
        }
        case DiagramFormat::Json:
            return diagramFromJson(data);
        case DiagramFormat::Binary:
        {
            std::vector<uint8_t> buf(data.begin(), data.end());
            return deserializeDiagramBinary(buf);
        }
    }
    return {};
}

std::vector<uint8_t> serializeDiagramBinary(const persistence::Diagram &diagram)
{
    const auto &pairs = diagram.getPairs();
    std::vector<uint8_t> buf;
    Size count = pairs.size();
    buf.resize(sizeof(Size) + count * (sizeof(Index) + 2 * sizeof(double)));
    std::memcpy(buf.data(), &count, sizeof(Size));
    uint8_t *ptr = buf.data() + sizeof(Size);
    for (const auto &p : pairs)
    {
        std::memcpy(ptr, &p.dimension, sizeof(Index));
        ptr += sizeof(Index);
        std::memcpy(ptr, &p.birth, sizeof(double));
        ptr += sizeof(double);
        std::memcpy(ptr, &p.death, sizeof(double));
        ptr += sizeof(double);
    }
    return buf;
}

persistence::Diagram deserializeDiagramBinary(const std::vector<uint8_t> &data)
{
    if (data.size() < sizeof(Size))
        return {};
    Size count;
    std::memcpy(&count, data.data(), sizeof(Size));
    std::vector<persistence::Pair> pairs;
    pairs.reserve(count);
    const uint8_t *ptr = data.data() + sizeof(Size);
    for (Size i = 0; i < count; ++i)
    {
        persistence::Pair p;
        std::memcpy(&p.dimension, ptr, sizeof(Index));
        ptr += sizeof(Index);
        std::memcpy(&p.birth, ptr, sizeof(double));
        ptr += sizeof(double);
        std::memcpy(&p.death, ptr, sizeof(double));
        ptr += sizeof(double);
        pairs.push_back(p);
    }
    return persistence::Diagram(std::move(pairs));
}

bool saveDiagramToFile(const persistence::Diagram &diagram, const std::string &path,
                       const DiagramIOConfig &config)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        return false;
    auto serialized = serializeDiagram(diagram, config.format);
    file.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    return file.good();
}

persistence::Diagram loadDiagramFromFile(const std::string &path, DiagramFormat format)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string data(static_cast<size_t>(size), '\0');
    file.read(data.data(), size);
    return deserializeDiagram(data, format);
}

std::string diagramToJson(const persistence::Diagram &diagram)
{
    std::ostringstream os;
    const auto &pairs = diagram.getPairs();
    os << "{\"version\":1,\"pairs\":[";
    for (Size i = 0; i < pairs.size(); ++i)
    {
        if (i > 0)
            os << ',';
        os << '[' << pairs[i].dimension << ',' << pairs[i].birth << ','
           << (std::isinf(pairs[i].death) ? "null" : std::to_string(pairs[i].death)) << ']';
    }
    os << "]}";
    return os.str();
}

persistence::Diagram diagramFromJson(const std::string &json)
{
    std::vector<persistence::Pair> pairs;
    const char *p = json.c_str();
    const char *end = p + json.size();
    auto skip_ws = [&]() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\t'))
            ++p;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (p < end && *p == c)
        {
            ++p;
            return true;
        }
        return false;
    };
    auto read_num = [&]() -> double {
        skip_ws();
        char *q;
        double v = std::strtod(p, &q);
        p = q;
        return v;
    };
    skip_ws();
    while (p < end && *p != '[')
        ++p;
    while (p < end)
    {
        if (expect('['))
        {
            Index dim = static_cast<Index>(read_num());
            expect(',');
            double birth = read_num();
            expect(',');
            skip_ws();
            double death;
            if (p + 4 <= end && p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l')
            {
                death = std::numeric_limits<double>::infinity();
                p += 4;
            }
            else
            {
                death = read_num();
            }
            pairs.push_back(persistence::Pair{birth, death, static_cast<Dimension>(dim)});
            expect(']');
            expect(',');
        }
        else
        {
            break;
        }
    }
    return persistence::Diagram(std::move(pairs));
}

} // namespace nerve::io
