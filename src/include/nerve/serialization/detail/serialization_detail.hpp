#pragma once
#include "nerve/serialization/serialization_manager.hpp"

#include <string>
#include <vector>

namespace nerve::serialization
{
struct SchemaVersion
{
    int major = 0;
    int minor = 0;
    int patch = 0;
};
SchemaVersion parseVersion(const std::string &str);
std::string versionToString(const SchemaVersion &v);
bool isCompatible(const SchemaVersion &a, const SchemaVersion &b, bool allow_major_upgrade = false);
int compareVersions(const SchemaVersion &a, const SchemaVersion &b);

class VersionNegotiator
{
public:
    void registerSchema(const std::string &name, const SchemaVersion &version);
    bool unregister(const std::string &name);
    SchemaVersion find(const std::string &name) const;
    SchemaVersion negotiate(const std::string &name, const SchemaVersion &required);
    std::vector<SchemaVersion> supportedVersions(const std::string &name) const;
};

class SerializationManager
{
public:
    static SerializationManager &instance();
    bool supportsFormat(const std::string &format) const;
};

// Utils
uint8_t swapBytes(uint8_t val);
uint16_t swapBytes(uint16_t val);
uint32_t swapBytes(uint32_t val);
uint64_t swapBytes(uint64_t val);

struct SchemaMetadata
{
    std::string name;
    SchemaVersion version;
    std::vector<SchemaVersion> compatible_versions;
};
bool isVersionSupported(const SchemaVersion &v, const std::vector<SchemaVersion> &supported);

// Result factory
template <typename T>
class Result
{
public:
    static Result success(const T &val);
    static Result failure(const std::string &error);
    bool isSuccess() const;
    T value() const;
    std::string error() const;
};
} // namespace nerve::serialization
