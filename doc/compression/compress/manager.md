# CompressionManager (Singleton)

```cpp
auto& mgr = nerve::compression::CompressionManager::instance();
mgr.setCompressionConfig(cfg);
auto result = mgr.compressImage(image);
auto results = mgr.compressImages(images);
auto adaptive = mgr.compressAdaptive(image);
auto perf = mgr.testMlPerformance(images, labels);
auto stats = mgr.getCompressionStats();
```

### AdaptiveCompressionSelector

Selects compression method per image based on content analysis.

```cpp
auto sel = mgr.getAdaptiveSelector();
auto analysis = sel->analyzeContent(image);
sel->updatePerformanceProfile("pca", 12.5f, 0.97f, 3.2);
auto profiles = sel->getMethodProfiles();
```

### MLPerformanceTester

Cross-validates accuracy impact of each compression method.

```cpp
auto tester = mgr.getPerformanceTester();
auto results = tester->testCompressionImpact(images, labels, {"pca", "pq", "autoencoder"});
auto best = tester->selectOptimalCompression(results, 0.05f);
```


[Back to Compression Index](index.md)
