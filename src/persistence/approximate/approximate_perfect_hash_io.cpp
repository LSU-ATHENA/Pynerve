#include "nerve/persistence/approximate/perfect_hash.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nerve::persistence::perfect
{
namespace
{
constexpr uint32_t kPerfectPivotMapMagic = 0x54504831U; // "TPH1"
constexpr uint32_t kPerfectPivotMapVersion = 1;

bool remainingBytes(std::FILE *f, size_t &remaining)
{
    const long current = std::ftell(f);
    if (current < 0)
    {
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0)
    {
        return false;
    }
    const long end = std::ftell(f);
    if (end < current || end < 0)
    {
        return false;
    }
    if (std::fseek(f, current, SEEK_SET) != 0)
    {
        return false;
    }
    remaining = static_cast<size_t>(end - current);
    return true;
}
} // namespace

void PerfectPivotMap::save(const std::string &filename) const
{
    std::FILE *f = std::fopen(filename.c_str(), "wb");
    if (!f)
        return;
    const uint8_t direct = direct_mode_ ? 1 : 0;
    auto writeVector = [f]<typename T>(const std::vector<T> &data) {
        const size_t size = data.size();
        return std::fwrite(&size, sizeof(size), 1, f) == 1 &&
               (size == 0 || std::fwrite(data.data(), sizeof(T), size, f) == size);
    };
    bool ok = std::fwrite(&kPerfectPivotMapMagic, sizeof(kPerfectPivotMapMagic), 1, f) == 1;
    ok = ok && std::fwrite(&kPerfectPivotMapVersion, sizeof(kPerfectPivotMapVersion), 1, f) == 1;
    ok = ok && std::fwrite(&seed1_, sizeof(seed1_), 1, f) == 1;
    ok = ok && std::fwrite(&seed2_, sizeof(seed2_), 1, f) == 1;
    ok = ok && std::fwrite(&num_keys_, sizeof(num_keys_), 1, f) == 1;
    ok = ok && std::fwrite(&min_key_, sizeof(min_key_), 1, f) == 1;
    ok = ok && std::fwrite(&direct, sizeof(direct), 1, f) == 1;
    ok = ok && writeVector(pilots_) && writeVector(values_) && writeVector(slot_keys_);
    ok = ok && writeVector(slot_occupied_) && writeVector(keys_);
    std::fclose(f);
    if (!ok)
    {
        throw std::runtime_error("failed to save perfect pivot map");
    }
}

bool PerfectPivotMap::load(const std::string &filename)
{
    std::FILE *f = std::fopen(filename.c_str(), "rb");
    if (!f)
        return false;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t seed1 = 0;
    uint64_t seed2 = 0;
    size_t num_keys = 0;
    int min_key = 0;
    uint8_t direct = 0;
    std::vector<uint8_t> pilots;
    std::vector<int> values;
    std::vector<int> slot_keys;
    std::vector<uint8_t> slot_occupied;
    std::vector<int> keys;
    auto readVector = [f]<typename T>(std::vector<T> &data) {
        size_t size = 0;
        if (std::fread(&size, sizeof(size), 1, f) != 1)
            return false;
        if (size > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            return false;
        }
        const size_t byte_count = size * sizeof(T);
        size_t remaining = 0;
        if (!remainingBytes(f, remaining) || byte_count > remaining)
        {
            return false;
        }
        try
        {
            data.resize(size);
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "error: %s\n", e.what());
            return false;
        }
        return size == 0 || std::fread(data.data(), sizeof(T), size, f) == size;
    };
    bool ok = std::fread(&magic, sizeof(magic), 1, f) == 1;
    ok = ok && std::fread(&version, sizeof(version), 1, f) == 1;
    ok = ok && magic == kPerfectPivotMapMagic && version == kPerfectPivotMapVersion;
    ok = ok && std::fread(&seed1, sizeof(seed1), 1, f) == 1;
    ok = ok && std::fread(&seed2, sizeof(seed2), 1, f) == 1;
    ok = ok && std::fread(&num_keys, sizeof(num_keys), 1, f) == 1;
    ok = ok && std::fread(&min_key, sizeof(min_key), 1, f) == 1;
    ok = ok && std::fread(&direct, sizeof(direct), 1, f) == 1;
    ok = ok && readVector(pilots) && readVector(values) && readVector(slot_keys);
    ok = ok && readVector(slot_occupied) && readVector(keys);
    std::fclose(f);
    if (!ok || direct > 1 || values.size() != slot_keys.size() ||
        values.size() != slot_occupied.size() || keys.size() != num_keys ||
        (!direct && values.size() != num_keys))
    {
        return false;
    }
    seed1_ = seed1;
    seed2_ = seed2;
    num_keys_ = num_keys;
    min_key_ = min_key;
    direct_mode_ = direct != 0;
    pilots_ = std::move(pilots);
    values_ = std::move(values);
    slot_keys_ = std::move(slot_keys);
    slot_occupied_ = std::move(slot_occupied);
    keys_ = std::move(keys);
    return true;
}

} // namespace nerve::persistence::perfect
