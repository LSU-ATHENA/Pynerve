#pragma once

#include "nerve/serialization/serialization_manager.hpp"

#include <string>
#include <vector>

namespace nerve::serialization
{

SchemaVersion parseVersion(const std::string &str);
std::string versionToString(const SchemaVersion &v);
bool isCompatible(const SchemaVersion &a, const SchemaVersion &b, bool allow_major_upgrade = false);
int compareVersions(const SchemaVersion &a, const SchemaVersion &b);

// Utils
uint8_t swapBytes(uint8_t val);
uint16_t swapBytes(uint16_t val);
uint32_t swapBytes(uint32_t val);
uint64_t swapBytes(uint64_t val);

bool isVersionSupported(const SchemaVersion &v, const std::vector<SchemaVersion> &supported);

} // namespace nerve::serialization
