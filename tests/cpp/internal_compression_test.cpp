#include "nerve/compression/model_aware_compression.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{

// PCA compression tests
bool check_pca_compress_decompress_roundtrip()
{
    nerve::compression::CompressionConfig cfg;
    cfg.compression_method = "pca";
    cfg.pca_components = 4;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    std::vector<float> data(100, 0.5f);
    auto result = pca.compress(data);
    auto decompressed = pca.decompress(result.compressed_data);

    if (result.original_size == 0)
    {
        std::cerr << "original_size should not be zero\n";
        return false;
    }
    if (result.compressed_size == 0)
    {
        std::cerr << "compressed_size should not be zero\n";
        return false;
    }
    if (decompressed.size() != data.size())
    {
        std::cerr << "decompressed size mismatch: " << decompressed.size() << " vs " << data.size()
                  << "\n";
        return false;
    }
    return true;
}

bool check_pca_compress_large_data()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 8;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    std::vector<float> data(1000, 0.25f);
    auto result = pca.compress(data);

    if (result.original_size != data.size() * sizeof(float))
    {
        std::cerr << "original_size mismatch\n";
        return false;
    }
    if (result.compressed_data.empty())
    {
        std::cerr << "compressed data should not be empty\n";
        return false;
    }
    return result.compression_ratio >= 1.0f;
}

bool check_pca_train_does_not_throw()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 3;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    // Train on some data
    std::vector<std::vector<float>> train_data;
    for (int i = 0; i < 5; ++i)
    {
        std::vector<float> row(10, static_cast<float>(i));
        train_data.push_back(row);
    }
    pca.train(train_data);

    // Compress after training
    std::vector<float> test(10, 1.0f);
    auto result = pca.compress(test);
    if (result.compressed_data.empty())
    {
        std::cerr << "compressed data should not be empty after training\n";
        return false;
    }
    return true;
}

bool check_pca_decompress_preserves_size()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 2;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    std::vector<float> original(50, 0.75f);
    auto result = pca.compress(original);
    auto decompressed = pca.decompress(result.compressed_data);

    // Decompressed should match original size
    if (decompressed.size() != original.size())
    {
        std::cerr << "size mismatch: " << decompressed.size() << " vs " << original.size() << "\n";
        return false;
    }
    // Decompressed should contain float values
    for (float v : decompressed)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "decompressed value is not finite: " << v << "\n";
            return false;
        }
    }
    return true;
}

bool check_pca_quality_score_range()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 2;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    // Same data: quality score should be high
    std::vector<float> original(20, 1.0f);
    std::vector<float> decompressed(20, 1.0f);

    float qs = pca.computeQualityScore(original, decompressed);
    if (qs < 0.0f || qs > 1.0f + 1e-6f)
    {
        std::cerr << "quality score out of range: " << qs << "\n";
        return false;
    }
    return true;
}

bool check_pca_quality_score_different_data()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 2;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    std::vector<float> original(20, 0.0f);
    std::vector<float> decompressed(20, 1.0f);

    float qs = pca.computeQualityScore(original, decompressed);
    // Score should be valid regardless of input
    return qs >= 0.0f && qs <= 1.0f;
}

bool check_pca_empty_data()
{
    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 3;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);

    std::vector<float> empty;
    auto result = pca.compress(empty);

    if (result.original_size != 0)
    {
        std::cerr << "empty data should have original_size 0\n";
        return false;
    }
    return true;
}

// CompressionConfig tests
bool check_config_defaults()
{
    nerve::compression::CompressionConfig cfg;
    if (cfg.compression_method.empty())
    {
        std::cerr << "default compression method should not be empty\n";
        return false;
    }
    if (cfg.target_compression_ratio <= 0.0f)
    {
        std::cerr << "target compression ratio should be positive\n";
        return false;
    }
    if (cfg.quality_threshold <= 0.0f || cfg.quality_threshold > 1.0f)
    {
        std::cerr << "quality threshold out of [0,1]: " << cfg.quality_threshold << "\n";
        return false;
    }
    return true;
}

bool check_config_pca_defaults()
{
    nerve::compression::CompressionConfig cfg;
    if (cfg.pca_components <= 0)
    {
        std::cerr << "pca_components should be positive\n";
        return false;
    }
    if (cfg.pca_variance_retained <= 0.0f || cfg.pca_variance_retained > 1.0f)
    {
        std::cerr << "pca_variance_retained out of range\n";
        return false;
    }
    return true;
}

bool check_config_custom_method()
{
    nerve::compression::CompressionConfig cfg;
    cfg.compression_method = "custom_codec";
    cfg.target_compression_ratio = 5.0f;
    cfg.quality_threshold = 0.8f;

    if (cfg.compression_method != "custom_codec")
        return false;
    if (cfg.target_compression_ratio != 5.0f)
        return false;
    if (cfg.quality_threshold != 0.8f)
        return false;
    return true;
}

// CompressionResult tests
bool check_compression_result_default()
{
    nerve::compression::CompressionResult res;
    if (res.original_size != 0)
        return false;
    if (res.compressed_size != 0)
        return false;
    if (res.compression_ratio != 1.0f)
        return false;
    if (res.quality_score != 1.0f)
        return false;
    if (res.compression_time_ms != 0.0f)
        return false;
    return res.compressed_data.empty();
}

bool check_compression_result_populated()
{
    nerve::compression::CompressionResult res;
    res.original_size = 1024;
    res.compressed_size = 256;
    res.compression_ratio = 4.0f;
    res.quality_score = 0.98f;
    res.compression_time_ms = 5.2f;
    res.compressed_data = {0.1f, 0.2f, 0.3f};

    if (res.compression_ratio != 4.0f)
    {
        std::cerr << "compression_ratio mismatch\n";
        return false;
    }
    if (res.compressed_data.size() != 3)
    {
        std::cerr << "compressed_data size mismatch\n";
        return false;
    }
    if (res.quality_score != 0.98f)
    {
        std::cerr << "quality_score mismatch\n";
        return false;
    }
    return true;
}

bool check_compression_result_bounds()
{
    nerve::compression::CompressionResult res;
    res.original_size = 100;
    res.compressed_size = 100;
    res.compression_ratio = 1.0f;
    res.quality_score = 0.0f;
    res.compression_time_ms = 100.0f;

    // Edge cases: compression_ratio of 1 (no compression)
    if (res.compression_ratio < 1.0f)
        return false;
    // Quality score can be 0
    if (res.quality_score < 0.0f || res.quality_score > 1.0f)
        return false;
    return true;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    // PCA compression
    run("pca_compress_decompress_roundtrip", check_pca_compress_decompress_roundtrip());
    run("pca_compress_large_data", check_pca_compress_large_data());
    run("pca_train_does_not_throw", check_pca_train_does_not_throw());
    run("pca_decompress_preserves_size", check_pca_decompress_preserves_size());
    run("pca_quality_score_range", check_pca_quality_score_range());
    run("pca_quality_score_different_data", check_pca_quality_score_different_data());
    run("pca_empty_data", check_pca_empty_data());

    // Config
    run("config_defaults", check_config_defaults());
    run("config_pca_defaults", check_config_pca_defaults());
    run("config_custom_method", check_config_custom_method());

    // CompressionResult
    run("compression_result_default", check_compression_result_default());
    run("compression_result_populated", check_compression_result_populated());
    run("compression_result_bounds", check_compression_result_bounds());

    return failures > 0 ? 1 : 0;
}
