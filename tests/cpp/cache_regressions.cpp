#include "nerve/cache/feature_cache.hpp"
#include "nerve/common/accelerated_types.hpp"
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

bool check_cache_put_and_get()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    cfg.max_entries = 1000;
    cfg.feature_dim = 4;
    cfg.enable_lru = true;
    cfg.enable_shared_memory = false;

    nerve::cache::FeatureCache cache(cfg);

    std::vector<float> features = {0.5f, 1.0f, 1.5f, 2.0f};
    bool ok = cache.putFeature(42, 1000, features);
    if (!ok)
    {
        std::cerr << "putFeature failed\n";
        return false;
    }

    std::vector<float> retrieved;
    bool found = cache.getFeature(42, 1000, retrieved);
    if (!found)
    {
        std::cerr << "getFeature failed to find entry\n";
        return false;
    }

    if (retrieved.size() != features.size())
    {
        std::cerr << "feature size mismatch\n";
        return false;
    }
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (std::abs(retrieved[i] - features[i]) > 1e-7f)
        {
            std::cerr << "feature value mismatch at " << i << "\n";
            return false;
        }
    }

    return true;
}

bool cache_get_feature_ptr()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    cfg.max_entries = 500;
    cfg.feature_dim = 3;
    cfg.enable_shared_memory = false;

    nerve::cache::FeatureCache cache(cfg);

    std::vector<float> feats = {1.0f, 2.0f, 3.0f};
    cache.putFeature(1, 100, feats);

    const auto *entry = cache.getFeaturePtr(1, 100);
    if (!entry)
    {
        std::cerr << "getFeaturePtr returned null\n";
        return false;
    }
    if (!entry->isValid)
    {
        std::cerr << "entry should be valid\n";
        return false;
    }
    if (entry->symbol_id != 1)
    {
        std::cerr << "symbol_id mismatch\n";
        return false;
    }

    return true;
}

bool check_cache_latest_feature()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    cfg.max_entries = 100;
    cfg.feature_dim = 2;
    cfg.enable_shared_memory = false;

    nerve::cache::FeatureCache cache(cfg);

    cache.putFeature(10, 100, {1.0f, 2.0f});
    cache.putFeature(10, 200, {3.0f, 4.0f});

    std::vector<float> latest;
    int64_t ts = 0;
    bool found = cache.getLatestFeature(10, latest, ts);
    if (!found)
    {
        std::cerr << "getLatestFeature failed\n";
        return false;
    }
    if (ts != 200)
    {
        std::cerr << "expected timestamp 200, got " << ts << "\n";
        return false;
    }

    return true;
}

bool check_cache_clear()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    cfg.max_entries = 100;
    cfg.feature_dim = 2;
    cfg.enable_shared_memory = false;

    nerve::cache::FeatureCache cache(cfg);

    cache.putFeature(1, 10, {1.0f, 2.0f});
    cache.putFeature(2, 20, {3.0f, 4.0f});

    cache.clearSymbol(1);
    std::vector<float> out;
    bool found = cache.getFeature(1, 10, out);
    if (found)
    {
        std::cerr << "entry should be cleared after clearSymbol\n";
        return false;
    }

    cache.clearAll();
    found = cache.getFeature(2, 20, out);
    if (found)
    {
        std::cerr << "entry should be cleared after clearAll\n";
        return false;
    }

    return true;
}

bool check_cache_stats()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    cfg.max_entries = 100;
    cfg.feature_dim = 2;
    cfg.enable_shared_memory = false;

    nerve::cache::FeatureCache cache(cfg);

    cache.putFeature(1, 10, {1.0f, 2.0f});
    std::vector<float> out;
    cache.getFeature(1, 10, out);
    cache.getFeature(2, 20, out);

    auto stats = cache.getStats();
    if (stats.hit_rate < 0.0 || stats.hit_rate > 1.0)
    {
        std::cerr << "hit rate out of range: " << stats.hit_rate << "\n";
        return false;
    }
    if (stats.miss_rate < 0.0 || stats.miss_rate > 1.0)
    {
        std::cerr << "miss rate out of range\n";
        return false;
    }
    if (stats.accesses == 0)
    {
        std::cerr << "expected some cache accesses\n";
        return false;
    }

    return true;
}

bool check_cache_config_defaults()
{
    nerve::cache::FeatureCache::CacheConfig cfg;
    if (cfg.max_entries == 0)
    {
        std::cerr << "max_entries should be > 0\n";
        return false;
    }
    if (cfg.feature_dim == 0)
    {
        std::cerr << "feature_dim should be > 0\n";
        return false;
    }
    if (cfg.ttl_ns <= 0)
    {
        std::cerr << "ttl_ns should be > 0\n";
        return false;
    }
    if (cfg.eviction_threshold <= 0.0f || cfg.eviction_threshold > 1.0f)
    {
        std::cerr << "eviction_threshold out of range\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_cache_put_and_get())
    {
        std::cerr << "FAIL: cache put and get\n";
        return 1;
    }
    if (!cache_get_feature_ptr())
    {
        std::cerr << "FAIL: cache get feature ptr\n";
        return 1;
    }
    if (!check_cache_latest_feature())
    {
        std::cerr << "FAIL: cache latest feature\n";
        return 1;
    }
    if (!check_cache_clear())
    {
        std::cerr << "FAIL: cache clear\n";
        return 1;
    }
    if (!check_cache_stats())
    {
        std::cerr << "FAIL: cache stats\n";
        return 1;
    }
    if (!check_cache_config_defaults())
    {
        std::cerr << "FAIL: cache config defaults\n";
        return 1;
    }
    return 0;
}
