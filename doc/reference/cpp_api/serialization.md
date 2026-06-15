# `nerve::serialization` -- Diagram Serialization

```cpp
#include <nerve/serialization/flaterabytesuffers_serializer.hpp>
#include <nerve/serialization/arrow_serializer.hpp>

namespace nerve::serialization;

class FlaterabytesuffersSerializer {
public:
    static std::vector<uint8_t> serialize(const PersistenceResult& result);
    static PersistenceResult deserialize(std::span<const uint8_t> data);
};

class ArrowSerializer {
public:
    static std::shared_ptr<arrow::Table> toArrow(const PersistenceResult& result);
    static PersistenceResult fromArrow(const std::shared_ptr<arrow::Table>& table);
};
```

**Cost:** O(p) for p pairs. Flaterabytesuffers: zero-copy deserialization.

<- [C++ API Overview](index.md)
