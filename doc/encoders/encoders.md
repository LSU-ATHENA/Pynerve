# Encoders

## Quick start

```python
from pynerve.encoders import encode, decode

points = np.random.randn(100, 3).astype(np.float32)
encoded = encode(points)
decoded = decode(encoded)

from pynerve.encoders import TopologicalEncoder
enc = TopologicalEncoder(feature_dim=64)
enc.enable_betti_numbers(True)
enc.enable_persistence_landscape(True)
features = enc.encode(diagram)
```

Autoencoder-based topology encoding and decoding. Provides CNN, MLP, graph,
topological, and hybrid encoders with SIMD-accelerated batch operations and
CUDA GPU kernels. Tensor core mixed-precision support for high-throughput
encoding pipelines.


## Pages

- [simd.md](simd.md) -- SIMD encode/decode batch ops
- [gpu.md](gpu.md) -- GPU encoder kernels, Tensor Core support


## API

```python
import pynerve.encoders as enc

def encode(data, encoder_type="auto", **kwargs) -> np.ndarray: ...
def decode(encoded, encoder_type="auto", **kwargs) -> np.ndarray: ...
```

```cpp
#include <nerve/encoders/encoders.hpp>
#include <nerve/encoders/simd_encoder.hpp>
#include <nerve/encoders/gpu_encoders.hpp>

namespace nerve::encoders {

class FeatureEncoder {
    virtual Tensor encode(const std::vector<std::vector<double>>&) const = 0;
    virtual Tensor encode(const SimplicialComplex&) const = 0;
    virtual Tensor encode(const Diagram&) const = 0;
    virtual std::vector<Tensor> encodeBatch(...) const = 0;
    virtual void setInputSize(Size) = 0;
    virtual void setOutputSize(Size) = 0;
    virtual Size getInputSize() const = 0;
    virtual Size getOutputSize() const = 0;
    virtual std::string getEncoderType() const = 0;
};

class CNNEncoder : public FeatureEncoder { /* conv layers */ };
class MLPEncoder : public FeatureEncoder { /* fc layers */ };
class TopologicalEncoder : public FeatureEncoder { /* Betti, landscape, image */ };
class PersistenceEncoder : public FeatureEncoder { /* landscape/image vectors */ };
class GraphEncoder : public FeatureEncoder { /* GCN/GAT layers */ };
class HybridEncoder : public FeatureEncoder { /* multi-modal fusion */ };

class EncoderFactory {
    static std::unique_ptr<FeatureEncoder> createCnnEncoder(...);
    static std::unique_ptr<FeatureEncoder> createMlpEncoder(...);
    static std::unique_ptr<FeatureEncoder> createTopologicalEncoder(Size dim);
    static std::unique_ptr<FeatureEncoder> createPersistenceEncoder(Size dim);
    static std::unique_ptr<FeatureEncoder> createGraphEncoder(...);
    static std::unique_ptr<FeatureEncoder> createHybridEncoder(...);
    static std::unique_ptr<FeatureEncoder> createDefaultPersistenceEncoder();
    static std::unique_ptr<FeatureEncoder> loadFromConfig(const std::string&);
};

}
```


## Encoder types

- **`CNNEncoder`**: Multi-channel grid -> Fixed vector, for image-like topological data.
- **`MLPEncoder`**: Flat vector -> Fixed vector, for general topology encoding.
- **`TopologicalEncoder`**: Complex/Diagram -> Topological feature vector, for Betti, landscape, image features.
- **`PersistenceEncoder`**: Diagram -> Landscape/image vectors, for persistence-specific encoding.
- **`GraphEncoder`**: Node+edge features -> Graph embedding, for GCN/GAT-based encoding.
- **`HybridEncoder`**: Multiple inputs -> Fused vector, for multi-modal fusion.


## Encoder fusion

```cpp
namespace nerve::encoders::fusion {

struct FusionConfig {
    bool fuse_persistence = true;
    bool fuse_normalization = true;
    bool fuse_activation = true;
    bool use_persistent_kernels = false;
};

class FusedEncoderPipeline {
    explicit FusedEncoderPipeline(const FusionConfig&);
    void encodeFused(const std::vector<std::pair<float, float>>&, std::vector<float>&);
};

class MemoryOptimizedEncoder {
    struct MemoryConfig {
        bool enable_checkpointing = true;
        bool activation_compression = true;
        int compression_bits = 8;
        size_t max_memory_mb = 1024;
    };
    std::vector<float> encodeWithCheckpointing(const std::vector<float>&);
    std::vector<float> backward(const std::vector<float>&);
    void clearCheckpoints();
};

}
```



## Practical guidance

### Which encoder to choose

- **Point clouds** -> `TopologicalEncoder`: extracts Betti, landscape, image features.
- **Diagrams (from PH)** -> `PersistenceEncoder`: domain-specific landscape/image vectors.
- **Grid/image data** -> `CNNEncoder`: spatial locality via convolution.
- **Flat vectors** -> `MLPEncoder`: dense feature extraction.
- **Graph data** -> `GraphEncoder`: message passing preserves structure.
- **Multi-modal** -> `HybridEncoder`: fuses topological + geometric features.

### Common pitfalls

1. **Input size mismatch**: Each encoder subclass has specific input expectations. `TopologicalEncoder` expects (batch, n_pairs, 3) for diagrams, while `MLPEncoder` expects (batch, flat_dim). Check `getInputSize()` before calling `encode()`.
2. **Memory in batched encoding**: `encodeBatch` pre-allocates output tensors. For very large batches (>100k items), use the streaming variant or reduce batch size.
3. **Serialization of trained encoders**: Use `EncoderFactory::saveToConfig()` to export encoder architecture + weights. The config string can be loaded later with `createFromConfig()`.
4. **Fused pipeline misconfiguration**: Fusion requires all layers in the pipeline to be compatible. `FusedEncoderPipeline::validate()` checks for incompatible activation sequences before encoding.

### Encoder pipeline example

```python
from pynerve.encoders import (
    Preprocessor, TopologicalEncoder,
    PersistenceEncoder, FeatureFusion,
)

# Build a multi-stage encoder pipeline
prep = Preprocessor()
prep.normalize().center().scale()

topo_enc = TopologicalEncoder(feature_dim=128)
topo_enc.enable_betti_numbers(True)
topo_enc.enable_persistence_landscape(True, resolution=100)

pers_enc = PersistenceEncoder(feature_dim=64)
pers_enc.set_method("persistence_image", resolution=32)

fusion = FeatureFusion()
fusion.add_encoder("topological", topo_enc)
fusion.add_encoder("persistence", pers_enc)
fusion.set_fusion_method("concat")  # "concat" | "attention" | "weighted"

# End-to-end
features = fusion.encode(prep.process(diagram))
```

### Memory optimization

```python
from pynerve.encoders.fusion import MemoryOptimizedEncoder

mem_enc = MemoryOptimizedEncoder(
    MemoryConfig(
        enable_checkpointing=True,
        activation_compression=True,
        compression_bits=8,
        max_memory_mb=512,
    )
)

# Checkpointing: stores intermediate activations compressed
# Backward pass recomputes from checkpoints
# Reduces memory by 2-4x at 10% compute overhead
```


## Advanced topics

### Encoder composition

```python
from pynerve.encoders import EncoderFactory

# Create a composite encoder
enc = EncoderFactory.createHybridEncoder(
    topology_config={"dim": 3, "landscape_resolution": 50},
    mlp_config={"hidden_dims": [256, 128]},
    fusion_config={"method": "attention"},
)

# Serialize entire pipeline
config = enc.toConfig()
restored = EncoderFactory.loadFromConfig(config)
assert np.allclose(enc.encode(data), restored.encode(data))
```

### Custom encoder subclass

```cpp
class MyCustomEncoder : public nerve::encoders::FeatureEncoder {
public:
    Tensor encode(const std::vector<std::vector<double>>& data) const override {
        // Custom logic
        Tensor result;
        // ...
        return result;
    }

    Tensor encode(const SimplicialComplex& complex) const override {
        // Custom logic for simplicial complexes
    }

    Tensor encode(const Diagram& diagram) const override {
        // Custom logic for persistence diagrams
    }

    std::string getEncoderType() const override { return "MyCustomEncoder"; }
};
```

Register with `EncoderFactory`:

```cpp
EncoderFactory::registerCustomEncoder("my_custom",
    [](const Config& cfg) { return std::make_unique<MyCustomEncoder>(cfg); });
```


## FAQ

**Q: Can I use the same encoder for point clouds and diagrams?**
A: The base `FeatureEncoder` supports both via overloaded `encode()`. However, the internal layers differ. `TopologicalEncoder` handles diagrams natively, while `MLPEncoder` treats everything as flat vectors. For best results, use the encoder matching your data type.

**Q: How do I export an encoder for deployment?**
A: Use `encode.toConfig()` which returns a JSON string containing architecture, weights, and parameters. Combine with `pynerve.serialization` to store in FlatBuffers for zero-copy loading on the target platform.

**Q: What is the overhead of encoder fusion?**
A: `HybridEncoder` fusion via concatenation adds O(d1 + d2) memory. Attention-based fusion adds O(d1 * d2) due to the attention matrix. For large feature dimensions, prefer concatenation or weighted fusion.

**Q: Does `TopologicalEncoder` support batch encoding?**
A: Yes. Call `encodeBatch()` with batched diagrams (shape `[batch, n_pairs, 3]`). The encoder automatically vectorizes across the batch dimension using SIMD or GPU.


### Cross-references

- `pynerve.ml`: ML pipeline using encoders
- `pynerve.nn`: Neural network layers
- `pynerve.compression`: Compression of encoder outputs
- `pynerve.serialization`: Persistence of encoder configs
