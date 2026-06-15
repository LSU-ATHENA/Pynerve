#include "nerve/core_types.hpp"
#include "nerve/io/async_io.hpp"
#include "nerve/io/diagram_io.hpp"
#include "nerve/io/mmap_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using nerve::Size;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_test_diagram()
{
    Diagram d;
    d.addPair(Pair{0.0, 1.0, 0});
    d.addPair(Pair{0.2, 0.7, 0});
    d.addPair(Pair{0.5, 2.5, 1});
    return d;
}

bool check_diagram_serialize_text_roundtrip()
{
    Diagram d = make_test_diagram();
    std::string s = nerve::io::serializeDiagram(d, nerve::io::DiagramFormat::Text);
    if (s.empty())
    {
        std::cerr << "serialized text empty\n";
        return false;
    }
    Diagram loaded = nerve::io::deserializeDiagram(s, nerve::io::DiagramFormat::Text);
    if (loaded.count() != d.count())
    {
        std::cerr << "roundtrip count mismatch: " << loaded.count() << " vs " << d.count() << "\n";
        return false;
    }
    return true;
}

bool check_diagram_serialize_binary_roundtrip()
{
    Diagram d = make_test_diagram();
    auto buf = nerve::io::serializeDiagramBinary(d);
    if (buf.empty())
    {
        std::cerr << "serialized binary empty\n";
        return false;
    }
    Diagram loaded = nerve::io::deserializeDiagramBinary(buf);
    if (loaded.count() != d.count())
    {
        std::cerr << "binary roundtrip count mismatch\n";
        return false;
    }
    return true;
}

bool check_diagram_json_roundtrip()
{
    Diagram d = make_test_diagram();
    std::string json = nerve::io::diagramToJson(d);
    if (json.empty())
    {
        std::cerr << "json empty\n";
        return false;
    }
    Diagram loaded = nerve::io::diagramFromJson(json);
    if (loaded.count() != d.count())
    {
        std::cerr << "json roundtrip count mismatch\n";
        return false;
    }
    return true;
}

bool check_diagram_save_load_file()
{
    Diagram d = make_test_diagram();
    const char *tmp_path = "/tmp/pynerve_test_diagram.txt";
    bool saved = nerve::io::saveDiagramToFile(d, tmp_path);
    if (!saved)
    {
        std::cerr << "saveDiagramToFile failed\n";
        return false;
    }
    Diagram loaded = nerve::io::loadDiagramFromFile(tmp_path, nerve::io::DiagramFormat::Text);
    std::remove(tmp_path);
    if (loaded.count() != d.count())
    {
        std::cerr << "file roundtrip count mismatch\n";
        return false;
    }
    return true;
}

bool check_diagram_empty()
{
    Diagram empty;
    Diagram d = make_test_diagram();
    std::string s = nerve::io::serializeDiagram(empty);
    Diagram loaded = nerve::io::deserializeDiagram(s);
    if (loaded.count() != 0)
    {
        std::cerr << "empty diagram roundtrip should be empty\n";
        return false;
    }
    return true;
}

bool check_mmap_read_write()
{
    const char *tmp_path = "/tmp/pynerve_test_mmap.bin";
    Diagram d = make_test_diagram();
    nerve::io::saveDiagramMmap(tmp_path, d);
    Diagram loaded = nerve::io::loadDiagramMmap(tmp_path);
    std::remove(tmp_path);
    if (loaded.count() != d.count())
    {
        std::cerr << "mmap roundtrip count mismatch\n";
        return false;
    }
    return true;
}

bool check_async_io_engine_creation()
{
    auto engine = nerve::io::IoEngine::create(nerve::io::IoBackend::Mmap);
    if (!engine)
    {
        std::cerr << "IoEngine::create returned null\n";
        return false;
    }
    if (engine->backend() != nerve::io::IoBackend::Mmap)
    {
        std::cerr << "wrong backend type\n";
        return false;
    }
    return true;
}

bool check_mmap_io_engine()
{
    nerve::io::MmapIoEngine engine;
    auto backend = engine.backend();
    if (backend != nerve::io::IoBackend::Mmap)
    {
        std::cerr << "MmapIoEngine wrong backend\n";
        return false;
    }
    auto stats = engine.stats();
    (void)stats;
    return true;
}

bool check_pread_full_error()
{
    nerve::Size result = nerve::io::preadFull(-1, nullptr, 0, 0);
    if (result != 0)
    {
        std::cerr << "preadFull on invalid fd should return 0\n";
        return false;
    }
    return true;
}

bool check_pwrite_full_error()
{
    nerve::Size result = nerve::io::pwriteFull(-1, nullptr, 0, 0);
    if (result != 0)
    {
        std::cerr << "pwriteFull on invalid fd should return 0\n";
        return false;
    }
    return true;
}

bool check_mmap_file_construction()
{
    nerve::io::MmapFile mf;
    if (mf.valid())
    {
        std::cerr << "default MmapFile should not be valid\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_diagram_serialize_text_roundtrip())
    {
        std::cerr << "FAIL: text roundtrip\n";
        return 1;
    }
    if (!check_diagram_serialize_binary_roundtrip())
    {
        std::cerr << "FAIL: binary roundtrip\n";
        return 1;
    }
    if (!check_diagram_json_roundtrip())
    {
        std::cerr << "FAIL: json roundtrip\n";
        return 1;
    }
    if (!check_diagram_save_load_file())
    {
        std::cerr << "FAIL: file roundtrip\n";
        return 1;
    }
    if (!check_diagram_empty())
    {
        std::cerr << "FAIL: empty diagram\n";
        return 1;
    }
    if (!check_mmap_read_write())
    {
        std::cerr << "FAIL: mmap\n";
        return 1;
    }
    if (!check_async_io_engine_creation())
    {
        std::cerr << "FAIL: io engine\n";
        return 1;
    }
    if (!check_mmap_io_engine())
    {
        std::cerr << "FAIL: mmap engine\n";
        return 1;
    }
    if (!check_pread_full_error())
    {
        std::cerr << "FAIL: pread error\n";
        return 1;
    }
    if (!check_pwrite_full_error())
    {
        std::cerr << "FAIL: pwrite error\n";
        return 1;
    }
    if (!check_mmap_file_construction())
    {
        std::cerr << "FAIL: mmap file\n";
        return 1;
    }
    return 0;
}
