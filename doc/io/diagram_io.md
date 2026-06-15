## Diagram I/O

```cpp
namespace nerve::io {

enum class DiagramFormat {
    Text,
    Json,
    Binary,
};

struct DiagramIOConfig {
    DiagramFormat format = DiagramFormat::Text;
    bool include_header = true;
    uint32_t version = 1;
};

std::string serializeDiagram(const persistence::Diagram& diagram,
                              DiagramFormat format = DiagramFormat::Text);
persistence::Diagram deserializeDiagram(const std::string& data,
                                        DiagramFormat format = DiagramFormat::Text);

std::vector<uint8_t> serializeDiagramBinary(const persistence::Diagram& diagram);
persistence::Diagram deserializeDiagramBinary(const std::vector<uint8_t>& data);

bool saveDiagramToFile(const persistence::Diagram& diagram,
                       const std::string& path,
                       const DiagramIOConfig& config = {});
persistence::Diagram loadDiagramFromFile(const std::string& path,
                                         DiagramFormat format = DiagramFormat::Text);

std::string diagramToJson(const persistence::Diagram& diagram);
persistence::Diagram diagramFromJson(const std::string& json);

}
```


### Text format

```text
3
0 0.5 1.2
1 0.8 inf
0 1.5 2.5
```

First line: pair count. Each line: `dimension birth death`. Uses `inf` for
infinite death (essential classes).

**Parsing rules:**
- Lines starting with `#` are ignored (comments)
- Empty lines are ignored
- Death values of `inf`, `Inf`, `INF`, or `infinity` are parsed as infinite

### JSON format

```json
{"version":1,"pairs":[
    {"dim":0,"birth":0.5,"death":1.2},
    {"dim":1,"birth":0.8,"death":null},
    {"dim":0,"birth":1.5,"death":2.5}
]}
```

`null` death = essential class.

**Schema:**
```
{
  "version": int,
  "pairs": [
    {
      "dim": int,
      "birth": float,
      "death": float | null
    }
  ]
}
```

### Binary format

```text
[4 bytes: count as Size]
[4 bytes: dim as Index][8 bytes: birth as double][8 bytes: death as double]
... repeated count times
```

Death = NaN for essential classes. Little-endian.

**Structure:**
```cpp
struct BinaryHeader {
    uint32_t num_pairs;
};

struct BinaryPair {
    uint32_t dimension;
    double birth;
    double death;  // NaN for essential classes
};
```


### Python

```python
import pynerve.io as nio

# Save in different formats
nio.save_diagram(diagram, "output.json", format="json")
nio.save_diagram(diagram, "output.txt", format="text")
nio.save_diagram(diagram, "output.bin", format="binary")

# Load (format auto-detected from extension)
diagram = nio.load_diagram("output.json")
```

### Round-trip guarantees

Text format suffers precision loss from text representation of floats and preserves no metadata. JSON also suffers precision loss from text representation but preserves version info. Binary format has no precision loss (IEEE 754) but preserves no metadata.

For exact round-trips, use binary format.


### Cross-references

- `pynerve.io.io`: I/O module overview
- `pynerve.serialization`: Schema-based serialization with versioning
- `pynerve.persistence`: Diagram data structures
