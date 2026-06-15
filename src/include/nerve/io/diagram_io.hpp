#pragma once
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nerve::io
{

enum class DiagramFormat
{
    Text,
    Json,
    Binary,
};

struct DiagramIOConfig
{
    DiagramFormat format = DiagramFormat::Text;
    bool include_header = true;
    uint32_t version = 1;
};

std::string serializeDiagram(const persistence::Diagram &diagram,
                             DiagramFormat format = DiagramFormat::Text);
persistence::Diagram deserializeDiagram(const std::string &data,
                                        DiagramFormat format = DiagramFormat::Text);
std::vector<uint8_t> serializeDiagramBinary(const persistence::Diagram &diagram);
persistence::Diagram deserializeDiagramBinary(const std::vector<uint8_t> &data);
bool saveDiagramToFile(const persistence::Diagram &diagram, const std::string &path,
                       const DiagramIOConfig &config = {});
persistence::Diagram loadDiagramFromFile(const std::string &path,
                                         DiagramFormat format = DiagramFormat::Text);
std::string diagramToJson(const persistence::Diagram &diagram);
persistence::Diagram diagramFromJson(const std::string &json);

} // namespace nerve::io
