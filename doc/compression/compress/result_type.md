# Result Type

```cpp
struct CompressionResult {
    std::vector<uint8_t> compressed_data;
    size_t original_size;
    size_t compressed_size;
    float compression_ratio;
    float quality_score;
    double compression_time_ms;
    std::string compression_method;
    std::vector<float> quality_metrics;
};
```


[Back to Compression Index](index.md)
