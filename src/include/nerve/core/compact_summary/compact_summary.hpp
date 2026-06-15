
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

namespace nerve::core
{

class CompactSummary
{
public:
    CompactSummary() = default;
    CompactSummary(const CompactSummary &) = default;
    CompactSummary &operator=(const CompactSummary &) = default;
    virtual ~CompactSummary() = default;

    virtual std::string toString() const { return "CompactSummary"; }

    virtual std::unordered_map<std::string, std::string> toMap() const { return {}; }

    virtual bool isValid() const { return true; }

    virtual std::size_t getMemoryFootprint() const { return sizeof(CompactSummary); }
};

} // namespace nerve::core
