# Practical Guidance

### Choosing a compression method

- **Fastest compression:** SIMD quantize.
- **Best ratio (linear):** PCA.
- **Best ratio (non-linear):** Autoencoder.
- **No training needed:** SIMD quantize or PCA.
- **Adaptive quality:** use `AdaptiveCompressionSelector`.
- **ML pipeline impact test:** use `MLPerformanceTester`.

### Common pitfalls

1. **Training-test mismatch**: Autoencoder and PCA methods assume the test data distribution matches training. If the deployment data differs significantly, retrain or use a non-trained method (SIMD quantize).
2. **Compression artifacts**: INT8 quantization can introduce errors that cascade through downstream ML. Always validate accuracy impact with `MLPerformanceTester` before deploying.
3. **Codebook size**: In Product Quantization, too few centroids (e.g., <16) causes high distortion. Too many centroids (>4096) causes overfitting. Rule of thumb: start with 256 centroids and 8 sub-vectors.
4. **Memory overhead during training**: Autoencoder training allocates intermediate activations. For large models (>100M parameters), enable gradient checkpointing or reduce batch size.
5. **SIMD alignment**: `simdQuantize` and `simdDequantize` require 64-byte aligned input for maximum throughput. Use `ALIGN_SIMD` or `std::assume_aligned`.

### Quantization parameters

The `simdQuantize` function computes: `output[i] = (uint8_t)clamp((input[i] - offset) / scale, 0, 255)`

```cpp
void simdQuantize(const double* input, uint8_t* output, size_t n,
                   double scale, double offset);
void simdDequantize(const uint8_t* input, double* output, size_t n,
                     double scale, double offset);
```

Choose scale and offset based on min/max:

```cpp
double min_val = *std::min_element(input, input + n);
double max_val = *std::max_element(input, input + n);
double scale = (max_val - min_val) / 255.0;
double offset = min_val;
```


[Back to Compression Index](index.md)
