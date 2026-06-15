# Compression Benchmarking

```python
from pynerve.compression import CompressionManager, benchmark_compression

# Compare all methods on a dataset
mgr = CompressionManager.instance()
results = mgr.benchmarkMethods(images, labels)

for method, metrics in results.items():
    print(f"{method}:")
    print(f"  Ratio: {metrics.compression_ratio:.1f}x")
    print(f"  Quality: {metrics.quality_score:.3f}")
    print(f"  Compress: {metrics.compress_time_ms:.1f}ms")
    print(f"  Decompress: {metrics.decompress_time_ms:.1f}ms")
    print(f"  ML accuracy: {metrics.ml_accuracy:.3f}")
```

### Quality metrics

- **PSNR** measured in dB (higher is better) -- peak signal-to-noise ratio.
- **SSIM** on a [0,1] scale (higher is better) -- structural similarity.
- **ML accuracy** on a [0,1] scale (higher is better) -- downstream task accuracy.
- **Compression ratio** > 1 (higher is better) -- original size divided by compressed size.

### Compression pipeline template

```python
from pynerve.compression import (
    CompressionConfig, CompressionManager,
    AutoencoderCompression,
)

cfg = CompressionConfig(
    target_compression_ratio=10.0,
    quality_threshold=0.95,
    enable_gpu_acceleration=True,
)

mgr = CompressionManager.instance()
mgr.setCompressionConfig(cfg)

def compress_and_deploy(model, calibration_data):
    result = mgr.compressImage(model)
    perf = mgr.testMlPerformance([model], labels)

    if perf.accuracy < 0.95:
        cfg.target_compression_ratio = 5.0
        result = mgr.compressImage(model)

    mgr.deploy()
```


[Back to Compression Index](index.md)
