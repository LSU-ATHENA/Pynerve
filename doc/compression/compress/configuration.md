# Configuration

```cpp
struct CompressionConfig {
    std::string compression_method = "pca";   // pca, pq, vq, autoencoder
    float target_compression_ratio = 10.0f;
    float quality_threshold = 0.95f;

    // PCA
    size_t pca_components = 50;
    float pca_variance_retained = 0.95f;

    // Product quantization
    size_t pq_subvectors = 8;
    size_t pq_centroids = 256;
    size_t pq_iterations = 20;

    // Vector quantization
    size_t vq_codebook_size = 256;
    size_t vq_iterations = 50;

    // Autoencoder
    std::vector<size_t> encoder_layers = {128, 64, 32};
    std::vector<size_t> decoder_layers = {32, 64, 128};
    std::string activation = "relu";
    float learning_rate = 0.001f;
    size_t training_epochs = 100;
    bool enable_gpu_acceleration = false;
    size_t batch_size = 32;
    bool enable_adaptive_compression = true;
};
```


[Back to Compression Index](index.md)
