# Complexity

- **PCA:** compress in O(n * d * k), decompress in O(n * k * d), achieving a 5-20x ratio.
- **Product Quantization:** compress in O(n * d * k), decompress in O(n * d), achieving a 10-50x ratio.
- **Vector Quantization:** compress in O(n * d * k), decompress in O(n * d), achieving a 10-100x ratio.
- **Autoencoder:** compress and decompress in O(n * d * h), achieving a 10-100x ratio.
- **SIMD quantize:** compress and decompress in O(n/8), achieving an 8x ratio.


[Back to Compression Index](index.md)
