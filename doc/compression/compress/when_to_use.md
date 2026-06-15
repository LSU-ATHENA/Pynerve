# When to Use Compression

- **Reduce model size before deployment:** use `AutoencoderCompression` or `PCACompression`.
- **Quantize FP32 weights to INT8:** use `simdQuantize` / `simdDequantize`.
- **GPU-accelerated encode/decode:** use `benchmarkAutoencoder` + CuDNN kernels.
- **Adaptive quality-vs-size trade-off:** use `AdaptiveCompressionSelector` + `CompressionManager`.
- **ML pipeline testing:** `MLPerformanceTester` cross-validates accuracy impact.


[Back to Compression Index](index.md)
