## Arrow serialization

Columnar format for interop with Pandas, Polars, and PyArrow.

```python
from pynerve.serialization import ArrowSerializer

arrow = ArrowSerializer()
table = arrow.to_arrow_table(diagram)
# table is an arrow::Table, usable with Pandas
import pyarrow as pa
df = table.to_pandas()
```

### Arrow schema

```python
from pynerve.serialization import ArrowSerializer
import pyarrow as pa

arrow = ArrowSerializer()
table = arrow.to_arrow_table(diagram)

print(table.schema)
# birth: float, death: float, dimension: int, persistence: float

mask = pa.compute.greater(table.column("persistence"), 0.5)
persistent = table.filter(mask)
```

### Columnar structure

The Arrow schema contains four columns: `birth` (float32, birth filtration value), `death` (float32, death filtration value), `dimension` (int32, homological dimension), and `persistence` (float32, death - birth, computed).

### Pandas / Polars interop

```python
import pandas as pd
import polars as pl

df = table.to_pandas()
pdf = pl.from_arrow(table)
restored = pa.Table.from_pandas(df)
```

### Predefined Arrow schemas

```python
from pynerve.formats import PersistenceImage, BettiVector

img = PersistenceImage()
img.width = 64
img.height = 64
img.image_data = pixel_values
table = img.to_arrow_table()
img.from_arrow_table(table)

bv = BettiVector()
bv.betti_numbers = [1, 3, 1]
table = bv.to_arrow_table()
```


## Arrow IPC streaming

For real-time streaming of persistence diagrams:

```python
from pynerve.serialization import ArrowIPCStream

stream = ArrowIPCStream()

# Write batches incrementally
for batch in diagram_batches:
    stream.write_batch(batch)

# Finalize and get the complete table
table = stream.finalize()

# Read back as streaming
reader = ArrowIPCStream(table)
for batch in reader.read_batches():
    process_batch(batch)
```

### Custom compute on Arrow columns

```python
import pyarrow.compute as pc

# Filter by persistence
mask = pc.greater(table.column("persistence"), 0.5)
persistent = table.filter(mask)

# Group by dimension
from pyarrow import compute as pc
by_dim = table.group_by("dimension")
agg = by_dim.aggregate([
    ("persistence", "mean"),
    ("persistence", "count"),
])

# Convert back to PersistenceDiagram
from pynerve.serialization import ArrowSerializer
arrow = ArrowSerializer()
diagram = arrow.from_arrow_table(persistent)
```

### Memory-mapped Arrow

```python
import pyarrow as pa
import pyarrow.ipc as ipc

# Write IPC file
with pa.OSFile("diagram.arrow", "wb") as sink:
    writer = ipc.new_file(sink, table.schema)
    writer.write_table(table)
    writer.close()

# Memory-map and read (zero-copy)
with pa.memory_map("diagram.arrow", "r") as mmap:
    reader = ipc.open_file(mmap)
    table = reader.read_all()
    # No data copy -- reads directly from mmap
```

### Interop with Polars

```python
import polars as pl

# Arrow -> Polars
pdf = pl.from_arrow(table)

# Polars -> Arrow
table = pdf.to_arrow()

# Polars query on diagram data
result = (pdf
    .filter(pl.col("persistence") > 0.5)
    .group_by("dimension")
    .agg([
        pl.col("persistence").mean().alias("mean_persistence"),
        pl.col("birth").min().alias("earliest_birth"),
    ])
)
```

### Predefined schema with images

```python
from pynerve.formats import PersistenceImage, BettiVector

# PersistenceImage as Arrow
img = PersistenceImage()
img.width = 64
img.height = 64
img.image_data = np.random.randn(64, 64).astype(np.float32)
table = img.to_arrow_table()  # Schema: [width, height, image_data]

# BettiVector as Arrow
bv = BettiVector()
bv.betti_numbers = [1, 2, 1, 0, 0]
table = bv.to_arrow_table()  # Schema: [betti_0, betti_1, ...]
```


## Performance notes

- **Write 100k pairs**: Arrow 0.8 ms, FlatBuffers 0.5 ms, JSON 15 ms
- **Read 100k pairs**: Arrow 0.5 ms, FlatBuffers 0.001 ms, JSON 10 ms
- **Filter by persistence**: Arrow 0.1 ms, FlatBuffers N/A, JSON N/A
- **Convert to pandas**: Arrow 0.3 ms, FlatBuffers 1.0 ms, JSON 2.0 ms
- **Memory-mapped read**: Arrow and FlatBuffers both support it; JSON does not

Arrow excels at analytical workloads (filtering, aggregation) where columns are accessed independently.


## FAQ

**Q: When should I use Arrow over FlatBuffers?**
A: Use Arrow when you need analytical queries (filtering, aggregation, group-by on persistence or dimension), interop with Pandas or Polars, or streaming workloads. Use FlatBuffers when you need the fastest random access to individual pairs.

**Q: Can I memory-map Arrow files?**
A: Yes. PyArrow supports memory-mapped IPC files via `pa.memory_map`, enabling zero-copy reads from disk. This works well for large diagrams that do not fit entirely in RAM.

**Q: Does Arrow support dictionary encoding for metadata strings?**
A: Yes. Enable dictionary encoding with `arrow.enable_dictionary=True` to compress repeated strings in metadata columns, reducing memory and storage overhead.


### Cross-references

- `pynerve.serialization`: Serialization overview
- `pynerve.serialization.flatbuffers`: Alternative compact format
- `pynerve.serialization.manager`: Format selection
- `pynerve.io`: Memory-mapped I/O
