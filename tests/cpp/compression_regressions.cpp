#include "nerve/common/accelerated_types.hpp"
#include "nerve/compression/gpu_autoencoder.hpp"
#include "nerve/compression/model_aware_compression.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_pca_compress_roundtrip()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(200);
    for (auto &v : data)
        v = dist(rng);

    nerve::compression::CompressionConfig cfg;
    cfg.compression_method = "pca";
    cfg.pca_components = 4;
    cfg.pca_variance_retained = 0.95f;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);
    std::vector<std::vector<float>> train_data(10, std::vector<float>(20));
    for (auto &row : train_data)
        for (auto &v : row)
            v = dist(rng);
    pca.train(train_data);

    auto result = pca.compress(data);
    auto decompressed = pca.decompress(result.compressed_data);

    if (result.original_size != data.size() * sizeof(float))
    {
        std::cerr << "compressed original_size mismatch\n";
        return false;
    }
    if (decompressed.size() != data.size())
    {
        std::cerr << "decompressed size mismatch\n";
        return false;
    }

    return true;
}

bool check_pca_quality_score()
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> original(50);
    for (auto &v : original)
        v = dist(rng);

    nerve::compression::CompressionConfig cfg;
    cfg.pca_components = 5;
    cfg.enable_gpu_acceleration = false;

    nerve::compression::PCACompression pca(cfg);
    std::vector<std::vector<float>> train_data(10, std::vector<float>(50));
    for (auto &row : train_data)
        for (auto &v : row)
            v = dist(rng);
    pca.train(train_data);

    auto result = pca.compress(original);
    auto decompressed = pca.decompress(result.compressed_data);

    float qs = pca.computeQualityScore(original, decompressed);
    if (qs < 0.0f || qs > 1.0f + 1e-6f)
    {
        std::cerr << "quality score out of range: " << qs << "\n";
        return false;
    }

    return true;
}

bool check_compression_result_valid()
{
    nerve::compression::CompressionResult res;
    res.original_size = 100;
    res.compressed_size = 50;
    res.compression_ratio = 2.0f;
    res.quality_score = 0.95f;
    res.compression_time_ms = 1.0;

    if (res.compression_ratio <= 0.0f)
    {
        std::cerr << "invalid compression ratio\n";
        return false;
    }
    if (res.quality_score < 0.0f || res.quality_score > 1.0f)
    {
        std::cerr << "quality score not in [0,1]\n";
        return false;
    }

    return true;
}

bool check_config_defaults_valid()
{
    nerve::compression::CompressionConfig cfg;
    if (cfg.compression_method.empty())
    {
        std::cerr << "empty compression method\n";
        return false;
    }
    if (cfg.target_compression_ratio <= 0.0f)
    {
        std::cerr << "invalid target compression ratio\n";
        return false;
    }
    if (cfg.quality_threshold <= 0.0f || cfg.quality_threshold > 1.0f)
    {
        std::cerr << "quality threshold out of range\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_pca_compress_roundtrip())
    {
        std::cerr << "FAIL: pca compress roundtrip\n";
        return 1;
    }
    if (!check_pca_quality_score())
    {
        std::cerr << "FAIL: pca quality score\n";
        return 1;
    }
    if (!check_compression_result_valid())
    {
        std::cerr << "FAIL: compression result valid\n";
        return 1;
    }
    if (!check_config_defaults_valid())
    {
        std::cerr << "FAIL: config defaults valid\n";
        return 1;
    }
    return 0;
}
