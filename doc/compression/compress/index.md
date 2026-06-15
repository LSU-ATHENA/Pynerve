# Compression

Model-aware compression reduces the memory footprint of neural network models by quantizing FP32 weights to INT8, or by using autoencoder-based compression on GPU. SIMD-accelerated pack/unpack routines convert between `uint8_t` and `double` at cache-line speed.

## Topics

- [When to Use Compression](when_to_use.md)  --  Guidance for selecting the right compression approach
- [Configuration](configuration.md)  --  CompressionConfig struct reference
- [Compression Methods](methods.md)  --  PCA, Product Quantization, Vector Quantization, Autoencoder
- [Result Type](result_type.md)  --  CompressionResult struct
- [CompressionManager](manager.md)  --  Singleton manager, AdaptiveCompressionSelector, MLPerformanceTester
- [SIMD Quantize / Dequantize](simd.md)  --  SIMD-accelerated quantization routines
- [Complexity](complexity.md)  --  Time and space complexity for each method
- [Practical Guidance](guidance.md)  --  Method selection, common pitfalls, quantization parameters
- [Advanced Usage](advanced.md)  --  Adaptive pipeline, mixed precision, streaming, GPU acceleration
- [Compression Benchmarking](benchmarking.md)  --  Quality metrics, pipeline template
- [FAQ](faq.md)  --  Frequently asked questions


[Back to doc/compression](../compression.md)
