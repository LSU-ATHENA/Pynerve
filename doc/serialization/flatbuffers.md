## FlatBuffers serialization

Schema-based binary serialization using FlatBuffers for compact size and
zero-copy access.

```python
from pynerve.serialization import FlatBuffersSerializer

fb = FlatBuffersSerializer()
bytes = fb.serialize(data, size, context)
data = fb.deserialize(bytes, context)
```

### FlatBuffers schema

```fbs
// persistence_diagram.fbs
table PersistencePair {
    birth: float;
    death: float;
    dimension: int;
}

table PersistenceDiagram {
    pairs: [PersistencePair];
    max_dimension: int;
    schema_version_major: short = 1;
    schema_version_minor: short = 1;
    metadata: [KeyValue];
}

table KeyValue {
    key: string;
    value: string;
}

root_type PersistenceDiagram;
```

### Zero-copy read

```cpp
#include <nerve/serialization/flatbuffers.h>
#include <flatbuffers/flatbuffers.h>

auto diagram = GetPersistenceDiagram(buffer);
auto pairs = diagram->pairs();
for (size_t i = 0; i < pairs->size(); i++) {
    auto pair = pairs->Get(i);
    float birth = pair->birth();
    float death = pair->death();
}
```

No deserialization step -- the buffer is read directly from memory.
Serialize is O(n), deserialize is O(1).

### Predefined schema types

```python
from pynerve.formats import PersistenceImage, BettiVector, LaplacianSpec

img = PersistenceImage()
img.width = 64
img.height = 64
img.image_data = pixel_values
buf = img.to_flatbuffer()
img.from_flatbuffer(buf.data(), buf.size())

bv = BettiVector()
bv.betti_numbers = [1, 3, 1]
bytes = bv.serialize()

ls = LaplacianSpec()
ls.eigenvalues = [0.0, 0.5, 1.0, 1.5]
bytes = ls.serialize()
```


## Custom FlatBuffers schemas

Define your own FlatBuffers schema for custom data:

```fbs
// my_custom_data.fbs
table CustomTopologicalData {
    persistence_pairs: [PersistencePair];
    eigenvalues: [float];
    metadata: [KeyValue];
    version: int = 1;
}

table PersistencePair {
    birth: float;
    death: float;
    dimension: int;
    gradient_pair: int = -1;  // optional DMT gradient pair index
}
```

Then register with the serialization manager:

```python
from pynerve.serialization import SerializationManager
from pynerve.formats import register_flatbuffer_schema

register_flatbuffer_schema("custom_topo", "my_custom_data.fbs")
mgr = SerializationManager.instance()
mgr.register_schema("custom_topo", serializer)
```

## Zero-copy in C++

```cpp
#include <flatbuffers/flatbuffers.h>
#include "persistence_diagram_generated.h"

void processBuffer(const uint8_t* buffer, size_t size) {
    // Verify buffer integrity
    auto verifier = flatbuffers::Verifier(buffer, size);
    assert(VerifyPersistenceDiagramBuffer(verifier));

    // Zero-copy access
    auto diagram = GetPersistenceDiagram(buffer);
    auto pairs = diagram->pairs();

    // Access without copying
    for (auto* pair : *pairs) {
        float birth = pair->birth();
        float death = pair->death();
        // process directly from buffer
    }
}
```

## Schema compatibility

```python
from pynerve.serialization import SchemaVersion, VersionNegotiator

v1 = SchemaVersion(1, 0, 0)
v2 = SchemaVersion(1, 1, 0)

negotiator = VersionNegotiator()
negotiator.register_schema_range(
    "persistence_diagram",
    min_version=v1,
    current_version=v2,
)

# Write with schema metadata
bytes = fb.serialize(data, size, {
    "schema_version": v2,
    "min_compatible": v1,
})

# Read: auto-negotiates
result = fb.deserialize(bytes, {
    "schema_version": v1,  # reader only knows v1
})
# Returns negotiated v1-compatible view
```

## Comparison: FlatBuffers vs other binary formats

- **Zero-copy read**: Supported by FlatBuffers, not by Protobuf or MessagePack
- **Schema evolution**: FlatBuffers supports additive changes; Protobuf supports full schema evolution; MessagePack has no schema
- **Code generation**: FlatBuffers and Protobuf both generate code; MessagePack does not
- **Random access**: FlatBuffers supports it; Protobuf and MessagePack do not
- **Size efficiency**: FlatBuffers is excellent; Protobuf and MessagePack are good
- **Language support**: FlatBuffers supports C++, Python, and others; Protobuf and MessagePack support many languages

FlatBuffers is the best choice for performance-critical serialization of fixed-schema persistence data.


## FAQ

**Q: When should I use FlatBuffers over Arrow?**
A: Use FlatBuffers when you need the fastest possible random access to individual pairs, zero-copy deserialization for latency-sensitive workloads, or compact binary storage for archival. Use Arrow when you need analytical queries (filtering, aggregation) or interop with Pandas/Polars.

**Q: Can I add new fields to the FlatBuffers schema?**
A: Yes. Adding fields is backward-compatible -- readers using an older schema version will ignore unrecognized fields. Removing fields is not backward-compatible, so always add new fields instead of removing old ones.

**Q: Does compression work with zero-copy access?**
A: No. Zstd or LZ4 compression reduces file size by 2-3x but destroys the zero-copy property. Use compression for archival storage and uncompressed FlatBuffers for real-time access.


### Cross-references

- `pynerve.serialization`: Serialization overview
- `pynerve.serialization.arrow`: Columnar alternative for interop
- `pynerve.serialization.manager`: Format selection and versioning
- `pynerve.io`: FlatBuffers I/O utilities
