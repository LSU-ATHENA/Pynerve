# Advanced Usage

### Adaptive compression pipeline

```python
from pynerve.compression import CompressionManager

mgr = CompressionManager.instance()
mgr.setCompressionConfig(CompressionConfig(
    compression_method="adaptive",
    target_compression_ratio=20.0,
    quality_threshold=0.95,
))

# Analyze content and select best method
result = mgr.compressAdaptive(image)
# Automatically chooses PCA/PQ/Autoencoder based on content features

# Cross-validate accuracy impact
perf = mgr.testMlPerformance(images, labels)
print(f"Best method: {perf.best_method}, accuracy: {perf.best_accuracy:.3f}")

# Get compression stats
stats = mgr.getCompressionStats()
# stats.total_images, stats.average_ratio, stats.average_quality
```

### Mixed-precision compression

Combine INT8 quantization with float16 storage:

```python
from pynerve.compression import MixedPrecisionCompression

mp = MixedPrecisionCompression(
    sensitive_layers=["conv1", "fc"],  # keep at FP32
    compression_layers=["conv2", "conv3"],  # quantize to INT8
)

mp.analyze_sensitivity(model, calibration_data)
mp.compress()
mp.deploy()  # exports optimized model
```

### Streaming compression

For out-of-core compression of large datasets:

```python
from pynerve.compression import StreamingCompressor

comp = StreamingCompressor(
    method="pca",
    components=64,
    chunk_size=10000,
)

for chunk in data_loader:  # iterates over dataset
    compressed = comp.compress_chunk(chunk)
    comp.write_chunk(compressed, output_stream)

# Finalize and save model
comp.finalize()
comp.save_model("streaming_pca.bin")
```

### GPU-accelerated autoencoder

```python
from pynerve.compression import AutoencoderCompression

cfg = CompressionConfig(
    compression_method="autoencoder",
    encoder_layers=[128, 64, 32],
    decoder_layers=[32, 64, 128],
    enable_gpu_acceleration=True,
    batch_size=256,
    training_epochs=50,
)

ae = AutoencoderCompression(cfg)
ae.train(training_images, validation_images)

# Benchmark GPU vs CPU
from pynerve.compression import benchmarkAutoencoder
bm = benchmarkAutoencoder(batch_size=256, input_dim=784, code_dim=32)
print(f"CPU: {bm.cpu_time_ms:.1f}ms, GPU: {bm.gpu_time_ms:.1f}ms, "
      f"speedup: {bm.speedup:.1f}x")
```


[Back to Compression Index](index.md)
