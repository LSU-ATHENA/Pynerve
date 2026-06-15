# Compression Methods

### PCACompression

Principal component analysis for linear dimensionality reduction.

```cpp
auto pca = std::make_shared<nerve::compression::PCACompression>(cfg);
pca->train(training_images);
auto result = pca->compress(image);          // single
auto batch = pca->compressBatch(images);     // batched
auto recon = pca->decompress(result.compressed_data);
pca->saveModel("pca_model.bin");
```

### ProductQuantization

Splits vectors into sub-vectors and quantizes each with a separate codebook.

```cpp
auto pq = std::make_shared<nerve::compression::ProductQuantization>(cfg);
pq->train(training_images);
auto result = pq->compress(image);
pq->saveCodebooks("pq_codebooks.bin");
```

### VectorQuantization

Standard k-means vector quantization with a single codebook.

```cpp
auto vq = std::make_shared<nerve::compression::VectorQuantization>(cfg);
vq->train(training_images);
auto result = vq->compress(image);
vq->saveCodebook("vq_codebook.bin");
```

### AutoencoderCompression

Neural autoencoder with configurable encoder/decoder layer sizes.

```cpp
auto ae = std::make_shared<nerve::compression::AutoencoderCompression>(cfg);
ae->train(training_images, validation_images);
auto result = ae->compress(image);
auto reconstructed = ae->decompress(result.compressed_data);
ae->saveModel("ae_model.bin");
auto stats = ae->getTrainingStats();
```


[Back to Compression Index](index.md)
