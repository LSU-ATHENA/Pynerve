// PTHash-style for O(1) collision-free lookup
#include "nerve/persistence/approximate/perfect_hash.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <ranges>
#include <utility>

namespace nerve::persistence::perfect
{
// Hash function (FNV-1a inspired)
inline uint64_t fnv1aHash(int key, uint64_t seed)
{
    uint64_t hash = seed;
    for (int i = 0; i < 4; ++i)
    {
        hash ^= static_cast<uint8_t>(key >> (i * 8));
        hash *= 1099511628211ULL;
    }
    return hash;
}
PerfectPivotMap::PerfectPivotMap()
    : seed1_(0x9E3779B97F4A7C15ULL)
    , seed2_(0xBB67AE8584CAA73BULL)
    , min_key_(0)
    , direct_mode_(false)
    , num_keys_(0)
{}
uint64_t PerfectPivotMap::hash1(int key) const
{
    return fnv1aHash(key, seed1_);
}
uint64_t PerfectPivotMap::hash2(int key) const
{
    return fnv1aHash(key, seed2_);
}
uint64_t PerfectPivotMap::hashCombined(int key, uint8_t pilot) const
{
    // Use pilot to perturb the hash
    return hash1(key) ^ (hash2(key) * (pilot + 1));
}
bool PerfectPivotMap::build(const std::vector<int> &keys, const std::vector<int> &vals)
{
    auto reset = [this]() {
        pilots_.clear();
        values_.clear();
        slot_keys_.clear();
        slot_occupied_.clear();
        keys_.clear();
        min_key_ = 0;
        direct_mode_ = false;
        num_keys_ = 0;
    };
    reset();
    if (keys.size() != vals.size() || keys.empty())
    {
        return false;
    }
    num_keys_ = keys.size();
    std::vector<std::pair<int, int>> sorted_pairs;
    sorted_pairs.reserve(num_keys_);
    for (size_t i = 0; i < keys.size(); ++i)
    {
        sorted_pairs.emplace_back(keys[i], vals[i]);
    }
    std::ranges::sort(sorted_pairs);
    for (size_t i = 1; i < sorted_pairs.size(); ++i)
    {
        if (sorted_pairs[i - 1].first == sorted_pairs[i].first)
        {
            reset();
            return false;
        }
    }
    keys_.reserve(num_keys_);
    for (const auto &entry : sorted_pairs)
    {
        keys_.push_back(entry.first);
    }
    auto buildSortedLayout = [&]() {
        pilots_.clear();
        values_.clear();
        slot_keys_.clear();
        values_.reserve(num_keys_);
        slot_keys_.reserve(num_keys_);
        for (const auto &[key, value] : sorted_pairs)
        {
            slot_keys_.push_back(key);
            values_.push_back(value);
        }
        slot_occupied_.assign(num_keys_, 1);
        min_key_ = 0;
        direct_mode_ = false;
        return true;
    };
    if (std::adjacent_find(keys_.begin(), keys_.end()) != keys_.end())
    {
        reset();
        return false;
    }
    const auto [min_it, max_it] = std::minmax_element(keys.begin(), keys.end());
    const int min_key = *min_it;
    const int max_key = *max_it;
    const auto range_i64 = static_cast<int64_t>(max_key) - static_cast<int64_t>(min_key) + 1;
    const size_t range = range_i64 > 0 ? static_cast<size_t>(range_i64) : 0;
    const bool dense_range =
        num_keys_ <= std::numeric_limits<size_t>::max() / 2 && range <= num_keys_ * 2;
    if (range > 0 && dense_range && range < 1000000)
    {
        direct_mode_ = true;
        min_key_ = min_key;
        values_.assign(range, -1);
        slot_keys_.assign(range, 0);
        slot_occupied_.assign(range, 0);
        for (size_t i = 0; i < keys.size(); ++i)
        {
            const size_t idx =
                static_cast<size_t>(static_cast<int64_t>(keys[i]) - static_cast<int64_t>(min_key_));
            values_[idx] = vals[i];
            slot_keys_[idx] = keys[i];
            slot_occupied_[idx] = 1;
        }
        return true;
    }
    const size_t num_buckets = std::max<size_t>(1, num_keys_);
    std::vector<std::vector<size_t>> buckets(num_buckets);
    for (size_t i = 0; i < keys.size(); ++i)
    {
        size_t bucket_idx = hash1(keys[i]) % num_buckets;
        buckets[bucket_idx].push_back(i);
    }
    std::vector<size_t> bucket_order(num_buckets);
    std::iota(bucket_order.begin(), bucket_order.end(), 0);
    std::ranges::sort(bucket_order, std::greater{},
                      [&buckets](size_t i) { return buckets[i].size(); });
    pilots_.assign(num_buckets, 0);
    values_.assign(num_keys_, -1);
    slot_keys_.assign(num_keys_, 0);
    slot_occupied_.assign(num_keys_, 0);
    std::vector<uint8_t> occupied(num_keys_, 0);

    // Larger buckets are placed first. A pilot is accepted only when all keys
    // in the bucket land in currently free slots and do not collide internally.
    for (size_t bucket_idx : bucket_order)
    {
        const auto &bucket = buckets[bucket_idx];
        if (bucket.empty())
            continue;
        bool found = false;
        for (int pilot_value = 0; pilot_value <= 255 && !found; ++pilot_value)
        {
            const uint8_t pilot = static_cast<uint8_t>(pilot_value);
            std::vector<size_t> positions;
            positions.reserve(bucket.size());
            bool collision = false;
            for (size_t key_idx : bucket)
            {
                size_t pos = hashCombined(keys[key_idx], pilot) % num_keys_;
                if (occupied[pos])
                {
                    collision = true;
                    break;
                }
                // Check for internal collisions within bucket
                if (std::ranges::find(positions, pos) != positions.end())
                {
                    collision = true;
                    break;
                }
                positions.push_back(pos);
            }
            if (!collision)
            {
                pilots_[bucket_idx] = pilot;
                for (size_t i = 0; i < bucket.size(); ++i)
                {
                    size_t key_idx = bucket[i];
                    size_t pos = positions[i];
                    occupied[pos] = true;
                    values_[pos] = vals[key_idx];
                    slot_keys_[pos] = keys[key_idx];
                    slot_occupied_[pos] = 1;
                }
                found = true;
            }
        }
        if (!found)
        {
            return buildSortedLayout();
        }
    }
    return true;
}
int PerfectPivotMap::lookup(int key) const
{
    if (values_.empty())
        return -1;
    if (direct_mode_)
    {
        const auto offset_i64 = static_cast<int64_t>(key) - static_cast<int64_t>(min_key_);
        if (offset_i64 < 0 || static_cast<size_t>(offset_i64) >= values_.size())
        {
            return -1;
        }
        const size_t pos = static_cast<size_t>(offset_i64);
        if (slot_occupied_[pos] && slot_keys_[pos] == key)
        {
            return values_[pos];
        }
        return -1;
    }
    if (pilots_.empty() || num_keys_ == 0)
    {
        auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (it == keys_.end() || *it != key)
        {
            return -1;
        }
        const size_t pos = static_cast<size_t>(it - keys_.begin());
        if (pos < values_.size() && pos < slot_occupied_.size() && slot_occupied_[pos])
        {
            return values_[pos];
        }
        return -1;
    }
    size_t num_buckets = pilots_.size();
    size_t bucket_idx = hash1(key) % num_buckets;
    uint8_t pilot = pilots_[bucket_idx];
    size_t pos = hashCombined(key, pilot) % num_keys_;
    if (pos < slot_occupied_.size() && slot_occupied_[pos] && slot_keys_[pos] == key)
    {
        return values_[pos];
    }
    return -1;
}
bool PerfectPivotMap::contains(int key) const
{
    if (values_.empty())
        return false;
    if (direct_mode_)
    {
        const auto offset_i64 = static_cast<int64_t>(key) - static_cast<int64_t>(min_key_);
        if (offset_i64 < 0 || static_cast<size_t>(offset_i64) >= slot_occupied_.size())
        {
            return false;
        }
        const size_t pos = static_cast<size_t>(offset_i64);
        return slot_occupied_[pos] && slot_keys_[pos] == key;
    }
    auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
    return it != keys_.end() && *it == key;
}
size_t PerfectPivotMap::memoryUsage() const
{
    return pilots_.size() * sizeof(uint8_t) + values_.size() * sizeof(int) +
           slot_keys_.size() * sizeof(int) + slot_occupied_.size() * sizeof(uint8_t) +
           keys_.size() * sizeof(int) + sizeof(*this);
}
double PerfectPivotMap::bitsPerKey() const
{
    if (num_keys_ == 0)
        return 0.0;
    return static_cast<double>(memoryUsage() * 8) / static_cast<double>(num_keys_);
}
// FastStaticMap implementation
bool FastStaticMap::build(const std::vector<int> &keys, const std::vector<int> &vals)
{
    if (keys.size() != vals.size())
        return false;
    if (keys.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        return false;
    keys_.clear();
    values_.clear();
    bucket_offsets_.clear();
    num_buckets_ = 0;
    bucket_size_ = 0;

    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
    {
        pairs.emplace_back(keys[i], vals[i]);
    }
    std::ranges::sort(pairs);
    for (size_t i = 1; i < pairs.size(); ++i)
    {
        if (pairs[i - 1].first == pairs[i].first)
        {
            return false;
        }
    }
    for (const auto &p : pairs)
    {
        keys_.push_back(p.first);
        values_.push_back(p.second);
    }
    if (keys_.empty())
    {
        return true;
    }
    num_buckets_ = std::max(size_t(1), keys_.size() / 16);
    bucket_size_ = (keys_.size() + num_buckets_ - 1) / num_buckets_;
    bucket_offsets_.resize(num_buckets_ + 1);
    bucket_offsets_[0] = 0;
    size_t current_bucket = 0;
    for (size_t i = 0; i < keys_.size(); ++i)
    {
        size_t bucket = i / bucket_size_;
        while (current_bucket < bucket)
        {
            bucket_offsets_[++current_bucket] = static_cast<uint32_t>(i);
        }
    }
    while (current_bucket < num_buckets_)
    {
        bucket_offsets_[++current_bucket] = static_cast<uint32_t>(keys_.size());
    }
    return true;
}
std::optional<int> FastStaticMap::find(int key) const
{
    if (keys_.empty())
        return std::nullopt;
    auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
    if (it != keys_.end() && *it == key)
    {
        size_t idx = it - keys_.begin();
        return values_[idx];
    }
    return std::nullopt;
}
size_t FastStaticMap::memoryUsage() const
{
    return keys_.size() * sizeof(int) + values_.size() * sizeof(int) +
           bucket_offsets_.size() * sizeof(uint32_t);
}
} // namespace nerve::persistence::perfect
