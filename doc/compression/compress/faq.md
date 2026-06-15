# FAQ

**Q: What compression ratio should I target?**
A: Start with 10x for lossy compression of ML weights. For diagrams, 5x is safer (diagrams are already sparse). Test quality degradation at each ratio and stop before accuracy drops below your threshold.

**Q: Does SIMD quantize work on GPU tensors?**
A: No -- SIMD operates on CPU memory. For GPU tensors, use the CUDA kernel `cuda_quantize` in `src/compression/gpu/quantize_kernel.cu` which launches one thread per 4 bytes.

**Q: How do I deploy a compressed model?**
A: Use `CompressionManager::deploy()` which exports the compressed model as a single FlatBuffers file containing: compressed weights, quantization parameters, layer metadata, and the decompression graph. Load with `pynerve.serialization.load()`.

**Q: Can I compress a model without retraining?**
A: Yes -- use post-training quantization via `simdQuantize` on the weights. This requires no training but typically achieves 5-8x compression with <1% accuracy loss on most models.

**Q: What hardware is required for Tensor Core compression?**
A: Tensor Core 16x16x16 MMA instructions require CUDA compute >= 7.0 (V100+). FP8 requires compute >= 8.9 (Ada/Hopper). Autoencoder training on GPU requires at least a few gigabytes of VRAM for moderate models.

**Q: Can I quantize only certain layers?**
A: Yes. The `AdaptiveCompressionSelector` analyzes each layer's sensitivity and applies different methods per layer. Sensitive layers (first conv, final fc) use less aggressive compression.

**Q: How do I know if my compressed model is safe to deploy?**
A: Run `mgr.testMlPerformance()` on a held-out validation set. Compare accuracy, precision, recall, and F1 against the uncompressed baseline. Deploy only if all metrics are within your tolerance threshold.

**Q: What is the power/energy impact of decompression?**
A: SIMD dequantize adds ~1uJ per kilobyte of data. Autoencoder decompression on GPU adds ~10W during inference. For battery-constrained devices, prefer SIMD quantize over neural compression.


### Cross-references

- `pynerve.encoders`: Encoder architectures
- `pynerve.nn`: Neural network layers
- `pynerve.core.simd_ops`: SIMD operations
- `pynerve.cuda`: GPU acceleration
- `pynerve.serialization.flatbuffers`: Storage format for compressed models
- `pynerve.validation.benchmarks`: Compression benchmarking


[Back to Compression Index](index.md)
