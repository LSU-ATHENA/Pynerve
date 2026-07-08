#include "nerve/cache/feature_cache.hpp"
#include "nerve/core_types.hpp"
#include "nerve/platform.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include <sstream>

namespace nerve::cache
{

FeatureCache::FeatureCache(const CacheConfig &config)
    : config_(config)
{}

FeatureCache::~FeatureCache()
{
    cleanupSharedMemory();
}

bool FeatureCache::putFeature(int64_t symbol_id, int64_t timestamp_ns,
                              const std::vector<float> &features)
{
    auto *cache = getOrCreateSymbolCache(symbol_id);
    if (!cache)
        return false;
    std::unique_lock lock(cache->mutex);
    FeatureEntry entry{timestamp_ns, symbol_id, features, 0, 0, true};
    if (cache->size < config_.max_entries)
    {
        if (cache->entries.size() <= cache->size)
            cache->entries.emplace_back(std::move(entry));
        else
            cache->entries[cache->size] = std::move(entry);
        ++cache->size;
    }
    else
    {
        cache->entries[cache->head] = std::move(entry);
        cache->head = (cache->head + 1) % config_.max_entries;
    }
    return true;
}

bool FeatureCache::getFeature(int64_t symbol_id, int64_t timestamp_ns, std::vector<float> &features)
{
    auto *cache = getOrCreateSymbolCache(symbol_id);
    if (!cache)
        return false;
    std::shared_lock lock(cache->mutex);
    for (size_t i = 0; i < cache->size; ++i)
    {
        const auto &entry = cache->entries[i];
        if (entry.symbol_id == symbol_id &&
            std::llabs(entry.timestamp_ns - timestamp_ns) < config_.ttl_ns)
        {
            features = entry.features;
            ++hits_;
            return true;
        }
    }
    ++misses_;
    return false;
}

bool FeatureCache::getLatestFeature(int64_t symbol_id, std::vector<float> &features,
                                    int64_t &timestamp_ns)
{
    auto *cache = getOrCreateSymbolCache(symbol_id);
    if (!cache)
        return false;
    std::shared_lock lock(cache->mutex);
    int64_t latest = 0;
    const FeatureEntry *found = nullptr;
    for (size_t i = 0; i < cache->size; ++i)
    {
        const auto &entry = cache->entries[i];
        if (entry.symbol_id == symbol_id && entry.timestamp_ns > latest)
        {
            latest = entry.timestamp_ns;
            found = &entry;
        }
    }
    if (found)
    {
        features = found->features;
        timestamp_ns = found->timestamp_ns;
        ++hits_;
        return true;
    }
    ++misses_;
    return false;
}

void FeatureCache::putFeaturesBatch(const std::vector<int64_t> &symbol_ids,
                                    const std::vector<int64_t> &timestamps,
                                    const std::vector<std::vector<float>> &features)
{
    for (size_t i = 0; i < symbol_ids.size(); ++i)
        putFeature(symbol_ids[i], timestamps[i], features[i]);
}

void FeatureCache::getFeaturesBatch(const std::vector<int64_t> &symbol_ids,
                                    const std::vector<int64_t> &timestamps,
                                    std::vector<std::vector<float>> &features,
                                    std::vector<bool> &found)
{
    features.resize(symbol_ids.size());
    found.resize(symbol_ids.size());
    for (size_t i = 0; i < symbol_ids.size(); ++i)
        found[i] = getFeature(symbol_ids[i], timestamps[i], features[i]);
}

void FeatureCache::evictExpired()
{
    std::shared_lock global_lock(global_mutex_);
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    for (auto &[id, cache] : symbol_caches_)
    {
        std::unique_lock lock(cache->mutex);
        size_t write = 0;
        for (size_t read = 0; read < cache->size; ++read)
        {
            if (std::llabs(now - cache->entries[read].timestamp_ns) < config_.ttl_ns)
            {
                if (write != read)
                    cache->entries[write] = std::move(cache->entries[read]);
                ++write;
            }
        }
        evictions_ += cache->size - write;
        cache->size = write;
    }
}

void FeatureCache::evictLru(size_t target_size)
{
    // simple capacity-based eviction
    std::shared_lock global_lock(global_mutex_);
    for (auto &[id, cache] : symbol_caches_)
    {
        std::unique_lock lock(cache->mutex);
        if (cache->size > target_size)
        {
            evictions_ += cache->size - target_size;
            cache->size = target_size;
        }
    }
}

void FeatureCache::clearSymbol(int64_t symbol_id)
{
    std::shared_lock global_lock(global_mutex_);
    auto it = symbol_caches_.find(symbol_id);
    if (it != symbol_caches_.end())
    {
        std::unique_lock lock(it->second->mutex);
        it->second->size = 0;
        it->second->head = 0;
    }
}

void FeatureCache::clearAll()
{
    std::unique_lock lock(global_mutex_);
    symbol_caches_.clear();
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
}

FeatureCache::CacheStats FeatureCache::getStats() const
{
    CacheStats stats{};
    stats.total_entries = 0;
    for (const auto &[id, cache] : symbol_caches_)
    {
        std::shared_lock lock(cache->mutex);
        stats.total_entries += cache->size;
    }
    auto h = hits_.load();
    auto m = misses_.load();
    auto total = h + m;
    stats.hit_rate = total > 0 ? static_cast<double>(h) / static_cast<double>(total) : 0.0;
    stats.miss_rate = total > 0 ? static_cast<double>(m) / static_cast<double>(total) : 0.0;
    stats.evictions = evictions_.load();
    stats.accesses = total;
    stats.memory_usage_bytes = stats.total_entries * sizeof(FeatureEntry);
    return stats;
}

void FeatureCache::resetStats()
{
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
}

const FeatureCache::FeatureEntry *FeatureCache::getFeaturePtr(int64_t symbol_id,
                                                              int64_t timestamp_ns) const
{
    auto it = symbol_caches_.find(symbol_id);
    if (it == symbol_caches_.end())
        return nullptr;
    std::shared_lock lock(it->second->mutex);
    for (size_t i = 0; i < it->second->size; ++i)
    {
        const auto &entry = it->second->entries[i];
        if (std::llabs(entry.timestamp_ns - timestamp_ns) < config_.ttl_ns)
            return &entry;
    }
    return nullptr;
}

void FeatureCache::initializeSharedMemory()
{
    if (!config_.enable_shared_memory || config_.shm_name.empty())
        return;
    shm_size_ = config_.max_entries * sizeof(FeatureEntry);
    shm_fd_ = ::shm_open(config_.shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (shm_fd_ < 0)
        return;
    if (::ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0)
    {
        ::close(shm_fd_);
        shm_fd_ = -1;
        shm_size_ = 0;
        return;
    }
    shm_ptr_ = nerve::sys::map(nullptr, shm_size_, nerve::sys::MAP_PROT_RW,
                                 nerve::sys::MAP_FLAG_SHARED, shm_fd_, 0);
    if (shm_ptr_ == nerve::sys::kMapFailed)
    {
        shm_ptr_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        shm_size_ = 0;
    }
}

void FeatureCache::cleanupSharedMemory()
{
    if (shm_ptr_)
    {
        nerve::sys::unmap(shm_ptr_, shm_size_);
        shm_ptr_ = nullptr;
    }
    if (shm_fd_ >= 0)
    {
        ::close(shm_fd_);
        if (!config_.shm_name.empty())
            ::shm_unlink(config_.shm_name.c_str());
        shm_fd_ = -1;
    }
    shm_size_ = 0;
}

FeatureCache::SymbolCache *FeatureCache::getOrCreateSymbolCache(int64_t symbol_id)
{
    std::shared_lock read_lock(global_mutex_);
    auto it = symbol_caches_.find(symbol_id);
    if (it != symbol_caches_.end())
        return it->second.get();
    read_lock.unlock();
    std::unique_lock write_lock(global_mutex_);
    auto &cache = symbol_caches_[symbol_id];
    if (!cache)
    {
        cache = std::make_unique<SymbolCache>();
        cache->entries.resize(config_.max_entries);
    }
    return cache.get();
}

bool FeatureCache::isExpired(const FeatureEntry &entry) const
{
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::llabs(now - entry.timestamp_ns) >= config_.ttl_ns;
}

RingBuffer::RingBuffer(size_t capacity)
    : buffer_(capacity)
{}

bool RingBuffer::push(const FeatureCache::FeatureEntry &entry)
{
    std::unique_lock lock(mutex_);
    if (size_ >= buffer_.size())
        return false;
    buffer_[tail_] = entry;
    tail_ = (tail_ + 1) % buffer_.size();
    ++size_;
    return true;
}

bool RingBuffer::getByTimestamp(int64_t timestamp_ns, FeatureCache::FeatureEntry &entry)
{
    std::shared_lock lock(mutex_);
    for (size_t i = 0; i < size_; ++i)
    {
        size_t idx = (head_ + i) % buffer_.size();
        if (buffer_[idx].timestamp_ns == timestamp_ns)
        {
            entry = buffer_[idx];
            return true;
        }
    }
    return false;
}

bool RingBuffer::getLatest(FeatureCache::FeatureEntry &entry)
{
    std::shared_lock lock(mutex_);
    if (size_ == 0)
        return false;
    size_t idx = tail_ == 0 ? buffer_.size() - 1 : tail_ - 1;
    entry = buffer_[idx];
    return true;
}

size_t RingBuffer::size() const
{
    return size_;
}
size_t RingBuffer::capacity() const
{
    return buffer_.size();
}
bool RingBuffer::empty() const
{
    return size_ == 0;
}
bool RingBuffer::full() const
{
    return size_ >= buffer_.size();
}

void RingBuffer::clear()
{
    std::unique_lock lock(mutex_);
    head_ = 0;
    tail_ = 0;
    size_ = 0;
}

LRUEvictionPolicy::LRUEvictionPolicy(int64_t ttl_ns)
    : ttl_ns_(ttl_ns)
{}

bool LRUEvictionPolicy::shouldEvict(const FeatureCache::FeatureEntry &entry) const
{
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return (now - entry.timestamp_ns) > ttl_ns_;
}

void LRUEvictionPolicy::updateAccess(FeatureCache::FeatureEntry &entry)
{
    entry.access_count++;
    entry.last_access_ns = std::chrono::steady_clock::now().time_since_epoch().count();
}

std::vector<size_t>
LRUEvictionPolicy::getEvictionCandidates(const std::vector<FeatureCache::FeatureEntry> &entries,
                                         size_t target_count) const
{
    std::vector<std::pair<int64_t, size_t>> timestamps;
    for (size_t i = 0; i < entries.size(); ++i)
        timestamps.emplace_back(entries[i].last_access_ns, i);
    std::sort(timestamps.begin(), timestamps.end());
    std::vector<size_t> candidates;
    for (size_t i = 0; i < target_count && i < timestamps.size(); ++i)
        candidates.push_back(timestamps[i].second);
    return candidates;
}

CacheManager &CacheManager::instance()
{
    static CacheManager mgr;
    return mgr;
}

void CacheManager::registerCache(const std::string &name, std::shared_ptr<FeatureCache> cache)
{
    std::unique_lock lock(mutex_);
    caches_[name] = std::move(cache);
}

std::shared_ptr<FeatureCache> CacheManager::getCache(const std::string &name)
{
    std::shared_lock lock(mutex_);
    auto it = caches_.find(name);
    return it != caches_.end() ? it->second : nullptr;
}

void CacheManager::cleanupAllCaches()
{
    std::unique_lock lock(mutex_);
    for (auto &[name, cache] : caches_)
        cache->clearAll();
}

std::vector<std::string> CacheManager::getCacheNames() const
{
    std::shared_lock lock(mutex_);
    std::vector<std::string> names;
    names.reserve(caches_.size());
    for (const auto &[name, _] : caches_)
        names.push_back(name);
    return names;
}

} // namespace nerve::cache
