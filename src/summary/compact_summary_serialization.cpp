
#include "nerve/summary/compact_summary.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nerve::summary
{

namespace
{

constexpr std::size_t kLifetimePaddingBytes = 3;
constexpr std::size_t kEigenvaluePaddingBytes = 2;

void appendUint8(uint8_t value, std::vector<uint8_t> *out)
{
    out->push_back(value);
}

void appendBool(bool value, std::vector<uint8_t> *out)
{
    appendUint8(value ? 1U : 0U, out);
}

void appendPadding(std::size_t count, std::vector<uint8_t> *out)
{
    out->insert(out->end(), count, 0U);
}

void appendUint16(uint16_t value, std::vector<uint8_t> *out)
{
    out->push_back(static_cast<uint8_t>(value & 0xFFU));
    out->push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void appendUint32(uint32_t value, std::vector<uint8_t> *out)
{
    out->push_back(static_cast<uint8_t>(value & 0xFFU));
    out->push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    out->push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    out->push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

void appendUint64(uint64_t value, std::vector<uint8_t> *out)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        out->push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

void appendInt64(int64_t value, std::vector<uint8_t> *out)
{
    appendUint64(std::bit_cast<uint64_t>(value), out);
}

void appendFloat(float value, std::vector<uint8_t> *out)
{
    static_assert(sizeof(float) == sizeof(uint32_t));
    appendUint32(std::bit_cast<uint32_t>(value), out);
}

bool canRead(const std::vector<uint8_t> &data, size_t offset, std::size_t count)
{
    return offset <= data.size() && data.size() - offset >= count;
}

bool skipBytes(const std::vector<uint8_t> &data, size_t *offset, std::size_t count)
{
    if (offset == nullptr || !canRead(data, *offset, count))
    {
        return false;
    }
    *offset += count;
    return true;
}

bool readUint8(const std::vector<uint8_t> &data, size_t *offset, uint8_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint8_t)))
    {
        return false;
    }
    *value = data[*offset];
    *offset += sizeof(uint8_t);
    return true;
}

bool readBool(const std::vector<uint8_t> &data, size_t *offset, bool *value)
{
    uint8_t raw = 0;
    if (!readUint8(data, offset, &raw))
    {
        return false;
    }
    *value = raw != 0U;
    return true;
}

bool readUint16(const std::vector<uint8_t> &data, size_t *offset, uint16_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint16_t)))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    *value = static_cast<uint16_t>(cursor[0]) |
             static_cast<uint16_t>(static_cast<uint16_t>(cursor[1]) << 8U);
    *offset += sizeof(uint16_t);
    return true;
}

bool readUint32(const std::vector<uint8_t> &data, size_t *offset, uint32_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint32_t)))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    *value = static_cast<uint32_t>(cursor[0]) | (static_cast<uint32_t>(cursor[1]) << 8U) |
             (static_cast<uint32_t>(cursor[2]) << 16U) | (static_cast<uint32_t>(cursor[3]) << 24U);
    *offset += sizeof(uint32_t);
    return true;
}

bool readUint64(const std::vector<uint8_t> &data, size_t *offset, uint64_t *value)
{
    if (offset == nullptr || value == nullptr || !canRead(data, *offset, sizeof(uint64_t)))
    {
        return false;
    }
    const auto *cursor = data.data() + *offset;
    uint64_t parsed = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        parsed |= static_cast<uint64_t>(*cursor++) << shift;
    }
    *value = parsed;
    *offset += sizeof(uint64_t);
    return true;
}

bool readInt64(const std::vector<uint8_t> &data, size_t *offset, int64_t *value)
{
    uint64_t raw = 0;
    if (!readUint64(data, offset, &raw))
    {
        return false;
    }
    *value = std::bit_cast<int64_t>(raw);
    return true;
}

bool readFloat(const std::vector<uint8_t> &data, size_t *offset, float *value)
{
    static_assert(sizeof(float) == sizeof(uint32_t));
    uint32_t raw = 0;
    if (!readUint32(data, offset, &raw))
    {
        return false;
    }
    *value = std::bit_cast<float>(raw);
    return true;
}

void appendLifetime(const CompactSummary::Lifetime &lifetime, std::vector<uint8_t> *data)
{
    appendFloat(lifetime.birth, data);
    appendFloat(lifetime.death, data);
    appendUint8(lifetime.dimension, data);
    appendPadding(kLifetimePaddingBytes, data);
    appendFloat(lifetime.persistence, data);
}

bool readLifetime(const std::vector<uint8_t> &data, size_t *offset,
                  CompactSummary::Lifetime *lifetime)
{
    return readFloat(data, offset, &lifetime->birth) && readFloat(data, offset, &lifetime->death) &&
           readUint8(data, offset, &lifetime->dimension) &&
           skipBytes(data, offset, kLifetimePaddingBytes) &&
           readFloat(data, offset, &lifetime->persistence);
}

void appendEigenvalue(const CompactSummary::Eigenvalue &eigenvalue, std::vector<uint8_t> *data)
{
    appendFloat(eigenvalue.value, data);
    appendUint16(eigenvalue.multiplicity, data);
    appendPadding(kEigenvaluePaddingBytes, data);
}

bool readEigenvalue(const std::vector<uint8_t> &data, size_t *offset,
                    CompactSummary::Eigenvalue *eigenvalue)
{
    return readFloat(data, offset, &eigenvalue->value) &&
           readUint16(data, offset, &eigenvalue->multiplicity) &&
           skipBytes(data, offset, kEigenvaluePaddingBytes);
}

} // namespace

std::vector<uint8_t> CompactSummary::HighDimExtension::serializeExtension() const
{
    std::vector<uint8_t> data;
    data.reserve(256);
    appendUint8(highdim_betti_count, &data);
    for (uint8_t i = 0; i < highdim_betti_count; ++i)
    {
        appendUint16(highdim_betti_top8[i], &data);
    }
    for (const auto &stats : highdim_lifetime_stats)
    {
        appendFloat(stats.mean_lifetime, &data);
        appendFloat(stats.std_deviation, &data);
        appendFloat(stats.max_lifetime, &data);
        appendUint32(stats.feature_count, &data);
    }
    for (size_t i = 0; i < dimension_complexity.size(); ++i)
    {
        appendFloat(dimension_complexity[i], &data);
        appendUint32(simplex_counts[i], &data);
    }
    appendBool(truncated_by_budget, &data);
    appendUint8(max_dimension_attempted, &data);
    appendUint32(num_boundary_ops, &data);
    appendFloat(budget_utilization, &data);
    appendFloat(compression_ratio, &data);
    appendFloat(memory_efficiency, &data);
    appendFloat(computational_efficiency, &data);
    return data;
}

bool CompactSummary::HighDimExtension::deserializeExtension(const std::vector<uint8_t> &data)
{
    size_t offset = 0;
    if (!readUint8(data, &offset, &highdim_betti_count) ||
        highdim_betti_count > highdim_betti_top8.size())
    {
        return false;
    }
    for (uint8_t i = 0; i < highdim_betti_count; ++i)
    {
        if (!readUint16(data, &offset, &highdim_betti_top8[i]))
        {
            return false;
        }
    }
    for (auto &stats : highdim_lifetime_stats)
    {
        if (!readFloat(data, &offset, &stats.mean_lifetime) ||
            !readFloat(data, &offset, &stats.std_deviation) ||
            !readFloat(data, &offset, &stats.max_lifetime) ||
            !readUint32(data, &offset, &stats.feature_count))
        {
            return false;
        }
    }
    for (size_t i = 0; i < dimension_complexity.size(); ++i)
    {
        if (!readFloat(data, &offset, &dimension_complexity[i]) ||
            !readUint32(data, &offset, &simplex_counts[i]))
        {
            return false;
        }
    }
    return readBool(data, &offset, &truncated_by_budget) &&
           readUint8(data, &offset, &max_dimension_attempted) &&
           readUint32(data, &offset, &num_boundary_ops) &&
           readFloat(data, &offset, &budget_utilization) &&
           readFloat(data, &offset, &compression_ratio) &&
           readFloat(data, &offset, &memory_efficiency) &&
           readFloat(data, &offset, &computational_efficiency);
}

size_t CompactSummary::HighDimExtension::extensionSizeBytes() const
{
    return serializeExtension().size();
}

std::vector<uint8_t> CompactSummary::serialize() const
{
    std::vector<uint8_t> data;
    data.reserve(512);
    appendUint8(lifetime_count, &data);
    for (uint8_t i = 0; i < lifetime_count; ++i)
    {
        appendLifetime(top_lifetimes[i], &data);
    }
    appendUint8(betti_dimension_count, &data);
    for (uint8_t i = 0; i < betti_dimension_count; ++i)
    {
        appendUint16(betti_counts[i], &data);
    }
    appendUint8(eigenvalue_count, &data);
    for (uint8_t i = 0; i < eigenvalue_count; ++i)
    {
        appendEigenvalue(top_eigenvalues[i], &data);
    }
    appendFloat(persistence_entropy, &data);
    appendFloat(betti_entropy, &data);
    appendFloat(spectral_entropy, &data);
    appendInt64(timestamp_ns, &data);
    appendInt64(symbol_id, &data);
    appendUint32(computation_time_us, &data);
    appendUint16(data_points_count, &data);
    appendFloat(noise_level, &data);
    appendBool(has_highdim_extension, &data);
    if (has_highdim_extension)
    {
        const auto extension = highdim_extension.serializeExtension();
        data.insert(data.end(), extension.begin(), extension.end());
    }
    return data;
}

bool CompactSummary::deserialize(const std::vector<uint8_t> &data)
{
    size_t offset = 0;
    if (!readUint8(data, &offset, &lifetime_count) || lifetime_count > top_lifetimes.size())
    {
        return false;
    }
    for (uint8_t i = 0; i < lifetime_count; ++i)
    {
        if (!readLifetime(data, &offset, &top_lifetimes[i]))
        {
            return false;
        }
    }
    if (!readUint8(data, &offset, &betti_dimension_count) ||
        betti_dimension_count > betti_counts.size())
    {
        return false;
    }
    for (uint8_t i = 0; i < betti_dimension_count; ++i)
    {
        if (!readUint16(data, &offset, &betti_counts[i]))
        {
            return false;
        }
    }
    if (!readUint8(data, &offset, &eigenvalue_count) || eigenvalue_count > top_eigenvalues.size())
    {
        return false;
    }
    for (uint8_t i = 0; i < eigenvalue_count; ++i)
    {
        if (!readEigenvalue(data, &offset, &top_eigenvalues[i]))
        {
            return false;
        }
    }
    if (!readFloat(data, &offset, &persistence_entropy) ||
        !readFloat(data, &offset, &betti_entropy) || !readFloat(data, &offset, &spectral_entropy) ||
        !readInt64(data, &offset, &timestamp_ns) || !readInt64(data, &offset, &symbol_id) ||
        !readUint32(data, &offset, &computation_time_us) ||
        !readUint16(data, &offset, &data_points_count) || !readFloat(data, &offset, &noise_level) ||
        !readBool(data, &offset, &has_highdim_extension))
    {
        return false;
    }
    if (has_highdim_extension)
    {
        std::vector<uint8_t> extension(data.begin() + static_cast<std::ptrdiff_t>(offset),
                                       data.end());
        return highdim_extension.deserializeExtension(extension);
    }
    return true;
}

size_t CompactSummary::sizeBytes() const
{
    return serialize().size();
}
bool CompactSummary::isValid() const
{
    if (lifetime_count > MAX_LIFETIMES || betti_dimension_count > MAX_BETTI_DIM ||
        eigenvalue_count > MAX_EIGENVALUES || !std::isfinite(persistence_entropy) ||
        !std::isfinite(betti_entropy) || !std::isfinite(spectral_entropy) ||
        !std::isfinite(noise_level))
    {
        return false;
    }
    for (uint8_t i = 0; i < lifetime_count; ++i)
    {
        const auto &lifetime = top_lifetimes[i];
        const bool finite_death = std::isfinite(lifetime.death);
        const bool infinite_death = std::isinf(lifetime.death) && lifetime.death > 0.0F;
        const bool finite_persistence = std::isfinite(lifetime.persistence);
        const bool infinite_persistence =
            std::isinf(lifetime.persistence) && lifetime.persistence > 0.0F;
        if (!std::isfinite(lifetime.birth) || (!finite_death && !infinite_death) ||
            (!finite_persistence && !infinite_persistence) ||
            (finite_death && lifetime.death < lifetime.birth) ||
            (finite_persistence && lifetime.persistence < 0.0F))
        {
            return false;
        }
    }
    for (uint8_t i = 0; i < eigenvalue_count; ++i)
    {
        if (!std::isfinite(top_eigenvalues[i].value))
        {
            return false;
        }
    }
    if (!has_highdim_extension)
    {
        return true;
    }
    const auto &extension = highdim_extension;
    if (extension.highdim_betti_count > extension.highdim_betti_top8.size() ||
        !std::isfinite(extension.budget_utilization) ||
        !std::isfinite(extension.compression_ratio) ||
        !std::isfinite(extension.memory_efficiency) ||
        !std::isfinite(extension.computational_efficiency) || extension.budget_utilization < 0.0F ||
        extension.compression_ratio < 0.0F || extension.memory_efficiency < 0.0F ||
        extension.computational_efficiency < 0.0F)
    {
        return false;
    }
    for (const auto &stats : extension.highdim_lifetime_stats)
    {
        if (!std::isfinite(stats.mean_lifetime) || !std::isfinite(stats.std_deviation) ||
            !std::isfinite(stats.max_lifetime) || stats.mean_lifetime < 0.0F ||
            stats.std_deviation < 0.0F || stats.max_lifetime < 0.0F)
        {
            return false;
        }
    }
    return std::all_of(extension.dimension_complexity.begin(), extension.dimension_complexity.end(),
                       [](float value) { return std::isfinite(value) && value >= 0.0F; });
}
bool CompactSummary::isUnderSizeLimit() const
{
    return sizeBytes() <= TARGET_SIZE_BYTES;
}

const CompactSummary::HighDimExtension &CompactSummary::getHighdimExtension() const
{
    return highdim_extension;
}
void CompactSummary::setHighdimExtension(const HighDimExtension &extension)
{
    highdim_extension = extension;
    has_highdim_extension = true;
}
void CompactSummary::clearHighdimExtension()
{
    highdim_extension = HighDimExtension{};
    has_highdim_extension = false;
}
uint16_t CompactSummary::getHighdimBetti(uint8_t dimension) const
{
    return dimension < highdim_extension.highdim_betti_count
               ? highdim_extension.highdim_betti_top8[dimension]
               : 0;
}
const CompactSummary::HighDimExtension::LifetimeStats &
CompactSummary::getLifetimeStats(uint8_t dimension) const
{
    static const CompactSummary::HighDimExtension::LifetimeStats empty;
    return dimension < highdim_extension.highdim_lifetime_stats.size()
               ? highdim_extension.highdim_lifetime_stats[dimension]
               : empty;
}
float CompactSummary::getDimensionComplexity(uint8_t dimension) const
{
    return dimension < highdim_extension.dimension_complexity.size()
               ? highdim_extension.dimension_complexity[dimension]
               : 0.0F;
}

} // namespace nerve::summary
