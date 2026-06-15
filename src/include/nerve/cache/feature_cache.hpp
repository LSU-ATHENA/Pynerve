
#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
namespace nerve
{
namespace cache
{
class FeatureCache
{
public:
    struct FeatureEntry
    {
        int64_t timestamp_ns;
        int64_t symbol_id;
        std::vector<float> features;
        uint32_t access_count;
        int64_t last_access_ns;
        bool isValid;
    };
    struct CacheConfig
    {
        size_t max_entries = 100000;
        size_t feature_dim = 128;
        int64_t ttl_ns = 60'000'000'000;
        float eviction_threshold = 0.8;
        bool enable_lru = true;
        bool enable_shared_memory = true;
        std::string shm_name = "nerve_cache";
    };
    explicit FeatureCache(const CacheConfig &config);
    ~FeatureCache();
    bool putFeature(int64_t symbol_id, int64_t timestamp_ns, const std::vector<float> &features);
    bool getFeature(int64_t symbol_id, int64_t timestamp_ns, std::vector<float> &features);
    bool getLatestFeature(int64_t symbol_id, std::vector<float> &features, int64_t &timestamp_ns);
    void putFeaturesBatch(const std::vector<int64_t> &symbol_ids,
                          const std::vector<int64_t> &timestamps,
                          const std::vector<std::vector<float>> &features);
    void getFeaturesBatch(const std::vector<int64_t> &symbol_ids,
                          const std::vector<int64_t> &timestamps,
                          std::vector<std::vector<float>> &features, std::vector<bool> &found);
    void evictExpired();
    void evictLru(size_t target_size);
    void clearSymbol(int64_t symbol_id);
    void clearAll();
    struct CacheStats
    {
        size_t total_entries;
        size_t memory_usage_bytes;
        double hit_rate;
        double miss_rate;
        size_t evictions;
        size_t accesses;
    };
    CacheStats getStats() const;
    void resetStats();
    const FeatureEntry *getFeaturePtr(int64_t symbol_id, int64_t timestamp_ns) const;

private:
    struct SymbolCache
    {
        std::vector<FeatureEntry> entries;
        size_t head = 0;
        size_t size = 0;
        mutable std::shared_mutex mutex;
    };
    CacheConfig config_;
    std::unordered_map<int64_t, std::unique_ptr<SymbolCache>> symbol_caches_;
    mutable std::shared_mutex global_mutex_;
    void *shm_ptr_ = nullptr;
    int shm_fd_ = -1;
    size_t shm_size_ = 0;
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> evictions_{0};
    void initializeSharedMemory();
    void cleanupSharedMemory();
    SymbolCache *getOrCreateSymbolCache(int64_t symbol_id);
    bool isExpired(const FeatureEntry &entry) const;
    void updateLru(SymbolCache &cache, size_t index);
};
class RingBuffer
{
public:
    explicit RingBuffer(size_t capacity);
    bool push(const FeatureCache::FeatureEntry &entry);
    bool getByTimestamp(int64_t timestamp_ns, FeatureCache::FeatureEntry &entry);
    bool getLatest(FeatureCache::FeatureEntry &entry);
    size_t size() const;
    size_t capacity() const;
    bool empty() const;
    bool full() const;
    void clear();

private:
    std::vector<FeatureCache::FeatureEntry> buffer_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t size_ = 0;
    mutable std::shared_mutex mutex_;
};
class LRUEvictionPolicy
{
public:
    explicit LRUEvictionPolicy(int64_t ttl_ns);
    bool shouldEvict(const FeatureCache::FeatureEntry &entry) const;
    void updateAccess(FeatureCache::FeatureEntry &entry);
    std::vector<size_t>
    getEvictionCandidates(const std::vector<FeatureCache::FeatureEntry> &entries,
                          size_t target_count) const;

private:
    int64_t ttl_ns_;
};
class CacheManager
{
public:
    static CacheManager &instance();
    void registerCache(const std::string &name, std::shared_ptr<FeatureCache> cache);
    std::shared_ptr<FeatureCache> getCache(const std::string &name);
    void cleanupAllCaches();
    std::vector<std::string> getCacheNames() const;

private:
    CacheManager() = default;
    std::unordered_map<std::string, std::shared_ptr<FeatureCache>> caches_;
    mutable std::shared_mutex mutex_;
};
} // namespace cache
} // namespace nerve
