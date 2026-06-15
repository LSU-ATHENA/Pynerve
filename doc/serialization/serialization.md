# Serialization

Schema-based binary serialization for persistence diagrams and topological
data structures. Supports FlatBuffers (compact binary), Arrow (columnar
interop with Pandas/Polars), and memory-mapped I/O for zero-copy loading.

```python
import pynerve
import numpy as np

# Compute and save
points = np.random.randn(500, 3)
result = pynerve.compute_persistence(points, max_dim=2)
pynerve.serialization.save("diagram.nvf", result, format="flatbuffers")

# Load later
loaded = pynerve.serialization.load("diagram.nvf")
print(loaded["pairs"][:3])
```


## API

### `pynerve.serialization.save`

```python
from pynerve.serialization import save

save(
    path="diagram.nvf",
    data=result,
    format="flatbuffers",        # "flatbuffers" | "arrow" | "json" | "binary"
    schema_version=(1, 1, 0),
    metadata={"dataset": "circles", "n_points": "500"},
    compression="none",          # "none" | "zstd" | "lz4"
)
```

**Supported formats:**

Four formats are available: `flatbuffers` (`.nvf`) for compact binary with fast random access, `arrow` (`.arrow`) for interop with Pandas, Polars, and PyArrow, `json` (`.json`) for debugging and human-readable output, and `binary` (`.bin`) for raw binary legacy compatibility.

### `pynerve.serialization.load`

Auto-detects format from magic bytes or file extension.

```python
from pynerve.serialization import load

data = load(
    path="diagram.nvf",
    format=None,                 # None = auto-detect from magic bytes
    schema_version=None,         # None = negotiate with file metadata
    validate=True,               # validate checksums on load
)
```


## Sub-modules

- [flatbuffers.md](flatbuffers.md): FlatBuffers schema-based binary serialization
- [arrow.md](arrow.md): Arrow columnar format and Pandas/Polars interop
- [manager.md](manager.md): Serialization manager, format selection, version negotiation


## MMAP I/O

Memory-mapped file save/load for large diagrams -- no deserialization cost.

```python
from pynerve.io import mmap_read_file, mmap_write_file
from pynerve.io import load_diagram_mmap, save_diagram_mmap

save_diagram_mmap("large_diagram.nvf", diagram)
mmap = mmap_read_file("large_diagram.nvf")
diagram = load_diagram_mmap("large_diagram.nvf")
```


## Error handling

```python
from pynerve.serialization import SerializationErrorCode

# Error codes by category
# 0x0001xxxx: INCOMPATIBLE_SCHEMA_VERSION
# 0x0002xxxx: UNSUPPORTED_SERIALIZATION_FORMAT
# 0x0003xxxx: DATA_CORRUPTED
# 0x0004xxxx: METADATA errors
# 0x0005xxxx: CONVERSION errors
```


## Complexity

- **FlatBuffers serialize**: O(n) where n = number of pairs
- **FlatBuffers deserialize**: O(1) -- zero-copy access
- **Arrow serialize**: O(n) -- columnar encoding
- **Arrow deserialize**: O(n) -- reconstruct columns
- **MMAP load**: O(1) -- virtual memory map
- **MMAP save**: O(n) -- page-cached write
- **JSON serialize**: O(n) -- text encoding overhead



## Practical guidance

### Choosing a format

Use cases for each format:
- **FlatBuffers**: Fastest random access -- zero-copy O(1) deserialize; also smallest size with no text overhead
- **Arrow**: Pandas/Polars interop and streaming -- native columnar format with zero-copy row-by-row IPC
- **JSON**: Debugging and human-readable output
- **Binary**: Legacy compatibility for old `.bin` files

### Common pitfalls

1. **FlatBuffers schema evolution**: Adding fields to the schema is backward-compatible, but removing fields is not. Always set `schema_version` and test compatibility with `VersionNegotiator`.
2. **Arrow dictionary encoding**: Repeated strings in metadata benefit from dictionary encoding. Enable with `arrow.enable_dictionary=True`.
3. **MMAP on network filesystems**: Memory-mapped files on NFS can cause SIGBUS if the file is truncated. Use `mmap_read_file` with SIGBUS signal handler, or pre-validate file size.
4. **Compression interaction**: Zstd compression reduces FlatBuffers size by 2-3x, but destroys zero-copy access. Use compression for archival, not for real-time access.

### MMAP best practices

```python
from pynerve.io import mmap_read_file, load_diagram_mmap, save_diagram_mmap

# Save diagram to mmap-compatible format
save_diagram_mmap("large.nvf", diagram)

# Zero-copy load (no deserialization)
mmap = mmap_read_file("large.nvf")
# Access data directly from virtual memory
print(f"Total pairs: {mmap.num_pairs}")

# For very large files (many gigabytes), load in sections
diagram = load_diagram_mmap("large.nvf", offset=0, size=1024*1024)
```

### Error handling strategies

```python
from pynerve.serialization import (
    save, load,
    SerializationErrorCode,
    SchemaNegotiationError,
    DataCorruptionError,
)

try:
    data = load("diagram.nvf", validate=True)
except SchemaNegotiationError as e:
    print(f"Schema version mismatch: {e}")
    data = load("diagram.nvf", schema_version=None)  # auto-negotiate
except DataCorruptionError as e:
    print(f"File corrupted: {e}")
    # Attempt recovery from backup
    data = load("diagram.nvf.backup", validate=True)
except Exception as e:
    # Catch-all for serialization failures
    print(f"Failed to load: {e}")
```

### Performance comparison

```python
from pynerve.validation import benchmark_serialization

bm = benchmark_serialization(
    num_pairs=100000,
    formats=["flatbuffers", "arrow", "json", "binary"],
)

for fmt, metrics in bm.items():
    print(f"{fmt}:")
    print(f"  Serialize: {metrics.serialize_ms:.1f}ms")
    print(f"  Deserialize: {metrics.deserialize_ms:.1f}ms")
    print(f"  Size: {metrics.size_bytes / 1024:.1f} KB")
```

Typical results for 100,000 pairs:
- **FlatBuffers**: 0.5 ms serialize, 0.001 ms deserialize, a few megabytes
- **Arrow**: 0.8 ms serialize, 0.5 ms deserialize, a few megabytes
- **JSON**: 15 ms serialize, 10 ms deserialize, a few megabytes
- **Binary**: 0.3 ms serialize, 0.2 ms deserialize, a few megabytes


## FAQ

**Q: Can I serialize a diagram with gradient information?**
A: Yes. The FlatBuffers schema supports an optional `gradient_pairs` field. Arrow serialization can add a gradient column. Use `PersistenceDiagram.withGradients()` before serializing.

**Q: How do I handle version migration?**
A: The `VersionNegotiator` selects the highest compatible version. For schema changes that require conversion, register a migration callback with `negotiator.register_migration(from_version, to_version, converter)`.

**Q: What happens if I load a file from a newer Pynerve version?**
A: The `VersionNegotiator` checks `min_compatible_version` in the file metadata. If the file requires a newer schema than available, it raises `SchemaNegotiationError` with a clear message about the required version.

**Q: Is MMAP thread-safe?**
A: Yes for reads (multiple threads can read the same mmap concurrently). Writes require external synchronization. For concurrent read/write, use the Arrow IPC streaming format which supports zero-copy multi-threaded reads.


### Cross-references

- `pynerve.io`: I/O module (mmap, diagram formats)
- `pynerve.persistence`: Diagram data structures
- `pynerve.validation`: Schema validation
- `pynerve.compression`: Compression of serialized data
