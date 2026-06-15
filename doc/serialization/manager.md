## Serialization manager

Format selection, schema versioning, and version negotiation for serialization.

### Schema versioning

```python
from pynerve.serialization import SchemaVersion, SchemaMetadata, VersionNegotiator

version = SchemaVersion(major=1, minor=1, patch=0)
print(version.to_string())         # "1.1.0"
v2 = SchemaVersion.from_string("1.0.0")

version.is_compatible_with(v2)     # True (1.1.0 >= 1.0.0)

meta = SchemaMetadata()
meta.version = SchemaVersion(1, 1, 0)
meta.min_compatible_version = SchemaVersion(1, 0, 0)
meta.schema_name = "persistence_diagram"
meta.custom_fields = {"algorithm": "ph6"}

negotiator = VersionNegotiator()
negotiator.register_schema(meta)
result = negotiator.negotiate_version(
    schema_name="persistence_diagram",
    requested_version=SchemaVersion(1, 0, 0),
)
# result.success, result.negotiated_version
# result.requires_conversion, result.conversion_strategy
```

### SerializationManager (singleton)

```python
from pynerve.serialization import SerializationManager
mgr = SerializationManager.instance()

mgr.register_schema("persistence_diagram", my_serializer)
mgr.register_schema("laplacian_spec", laplacian_serializer)

ctx = mgr.create_context(format="flatbuffers", version=SchemaVersion(1, 1, 0))
bytes = mgr.serialize(data, schema_name="persistence_diagram", context=ctx)
decoded = mgr.deserialize(bytes, schema_name="persistence_diagram", context=ctx)

bytes = mgr.serialize_with_metadata(data, metadata, context=ctx)
data, meta = mgr.deserialize_with_metadata(bytes, context=ctx)
```

### PH5/PH6 schema serializer

```python
from pynerve.serialization import PH5PH6SchemaSerializer

ph = PH5PH6SchemaSerializer()
artifact_meta = PH5PH6ArtifactMetadata(
    schema_version=SchemaVersion(1, 1, 0),
    artifact_type="persistence_result",
    algorithm_variant="ph6",
    has_highdim_extension=True,
)
bytes = ph.serialize_ph6_artifact(data, size, artifact_meta, context)
```

### Format selection strategy

```python
from pynerve.serialization import SerializationManager

mgr = SerializationManager.instance()

context = mgr.select_format(
    data_size=len(pairs),
    use_case="archival",        # "archival" | "interop" | "debug"
    target_tool="pandas",
)
```


## Version negotiation strategy

```python
from pynerve.serialization import SchemaVersion, VersionNegotiator

negotiator = VersionNegotiator()

# Register supported schemas
negotiator.register_schema(SchemaMetadata(
    version=SchemaVersion(1, 1, 0),
    min_compatible_version=SchemaVersion(1, 0, 0),
    schema_name="persistence_diagram",
    custom_fields={"algorithm": "ph6"},
))

# Negotiate with reader
result = negotiator.negotiate_version(
    schema_name="persistence_diagram",
    requested_version=SchemaVersion(1, 0, 0),
)

if result.success:
    print(f"Negotiated version: {result.negotiated_version}")
    if result.requires_conversion:
        print(f"Conversion strategy: {result.conversion_strategy}")
else:
    print(f"Cannot negotiate: {result.error}")
```

## Format selection details

```python
from pynerve.serialization import SerializationManager

mgr = SerializationManager.instance()

# Automatic format selection based on use case
for use_case, data_size in [("archival", 1000), ("interop", 10000)]:
    ctx = mgr.select_format(
        data_size=data_size,
        use_case=use_case,
    )
    print(f"{use_case}: {ctx.format}")

# Results:
# archival (1000 pairs): flatbuffers (compact, durable)
# interop (10000 pairs): arrow (Pandas/Polars compatible)
# debug (any): json (human-readable)
```

## Custom schema serialization

```python
from pynerve.serialization import PH5PH6SchemaSerializer
from pynerve.serialization import SchemaVersion, PH5PH6ArtifactMetadata

# Serialize with PH6 metadata
serializer = PH5PH6SchemaSerializer()
artifact_meta = PH5PH6ArtifactMetadata(
    schema_version=SchemaVersion(1, 1, 0),
    artifact_type="persistence_result",
    algorithm_variant="ph6",
    has_highdim_extension=True,
)

# Include metadata in serialized output
bytes = serializer.serialize_ph6_artifact(
    data, size, artifact_meta,
    {"source": "experiment_42", "timestamp": "2026-01-15"}
)

# Deserialize with version negotiation
decoded, meta = serializer.deserialize_with_metadata(bytes)
print(f"Algorithm: {meta.algorithm_variant}")
print(f"Schema version: {meta.schema_version}")
```

## Manager singleton usage

```python
mgr = SerializationManager.instance()

# Register all schemas at startup
mgr.register_all_default_schemas()

# Serialize with automatic format selection
ctx = mgr.create_context(
    format="flatbuffers",
    version=SchemaVersion(1, 1, 0),
)

# Single object
data_bytes = mgr.serialize(diagram, "persistence_diagram", ctx)

# Batch serialization (more efficient for multiple objects)
batch_bytes = mgr.serialize_batch(
    [diagram1, diagram2, diagram3],
    "persistence_diagram",
    ctx,
)

# Deserialize
loaded = mgr.deserialize(data_bytes, "persistence_diagram", ctx)

# With metadata
data_with_meta = mgr.serialize_with_metadata(
    diagram, meta, "persistence_diagram", ctx
)
loaded_data, loaded_meta = mgr.deserialize_with_metadata(
    data_with_meta, "persistence_diagram", ctx
)
```

## Async serialization

```python
# Async serialization for non-blocking writes
future = mgr.serialize_async(
    diagram, "persistence_diagram", ctx,
    callback=lambda bytes: save_to_disk(bytes, "output.nvf"),
)

# ... do other work ...
result = future.result()  # blocks if not done
```

## Registry of known schemas

Five schemas are registered by default:
- `persistence_diagram`: flatbuffers, version 1.1.0 -- birth-death pairs
- `laplacian_spec`: flatbuffers, version 1.0.0 -- eigenvalue spectrum
- `compact_summary`: flatbuffers, version 1.0.0 -- around one kilobyte summary
- `persistence_image`: arrow, version 1.0.0 -- 2D image representation
- `betti_vector`: arrow, version 1.0.0 -- Betti number vector


## FAQ

**Q: How does the SerializationManager select a format automatically?**
A: The `select_format` method uses the use case and data size: `"archival"` selects FlatBuffers for compact durable storage, `"interop"` selects Arrow for Pandas/Polars compatibility, and `"debug"` selects JSON for human readability.

**Q: How does version negotiation work?**
A: The `VersionNegotiator` compares the reader's requested version against the writer's `min_compatible_version`. If the requested version falls within the supported range, the negotiator returns the highest compatible version. If conversion is needed, it provides a `conversion_strategy` callback.

**Q: Can I register my own schema?**
A: Yes. Call `mgr.register_schema(name, serializer)` with any custom serializer. For FlatBuffers schemas, use `register_flatbuffer_schema()` to load a `.fbs` file. For custom versioning logic, register a `SchemaMetadata` with the `VersionNegotiator`.


### Cross-references

- `pynerve.serialization`: Serialization overview
- `pynerve.serialization.flatbuffers`: FlatBuffers format
- `pynerve.serialization.arrow`: Arrow format
- `pynerve.io`: File I/O utilities
