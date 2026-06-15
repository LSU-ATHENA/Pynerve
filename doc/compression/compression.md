# Compression

Model-aware compression reduces the memory footprint of neural network models by quantizing FP32 weights to INT8, or by using autoencoder-based compression on GPU. SIMD-accelerated pack/unpack routines convert between `uint8_t` and `double` at cache-line speed.

## Sections

- [When to Use Compression](compress/when_to_use.md)  --  Guidance for selecting the right compression approach
- [Configuration](compress/configuration.md)  --  CompressionConfig struct reference
- [Compression Methods](compress/methods.md)  --  PCA, Product Quantization, Vector Quantization, Autoencoder
- [Result Type](compress/result_type.md)  --  CompressionResult struct
- [CompressionManager](compress/manager.md)  --  Singleton manager, AdaptiveCompressionSelector, MLPerformanceTester
- [SIMD Quantize / Dequantize](compress/simd.md)  --  SIMD-accelerated quantization routines
- [Complexity](compress/complexity.md)  --  Time and space complexity for each method
- [Practical Guidance](compress/guidance.md)  --  Method selection, common pitfalls, quantization parameters
- [Advanced Usage](compress/advanced.md)  --  Adaptive pipeline, mixed precision, streaming, GPU acceleration
- [Compression Benchmarking](compress/benchmarking.md)  --  Quality metrics, pipeline template
- [FAQ](compress/faq.md)  --  Frequently asked questions
