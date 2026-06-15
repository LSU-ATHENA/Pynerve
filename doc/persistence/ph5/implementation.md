# Internal Implementation Details

### PH5PH6Engine Architecture

```cpp
class PH5PH6Engine {
public:
    // Configuration
    PH5PH6Config config;

    // State
    std::vector<Column> columns;
    std::vector<int> pivot_of;
    DeterminismContract determinism;
    ChecksumCalculator checksum;

    // Metrics collection
    PH5PH6Metrics metrics;
    ErrorLog error_log;

    // Core methods
    void runReduction();
    void validateResults();
    Checksum computeChecksum();

    // Advanced clearing
    void cascadingClear(size_t birth_col);

    // Stability
    bool runStabilityTest(const Points& points, int max_dim, int num_runs);

    // Metrics
    PH5PH6Metrics getComputationMetrics() const;
};
```

### Checksum Implementation

```cpp
class ChecksumCalculator {
    SHA256_CTX ctx;

public:
    void begin() { SHA256_Init(&ctx); }

    void addPairs(const std::vector<PersistentPair>& pairs) {
        for (const auto& p : pairs) {
            uint8_t buf[20];  // 8 + 8 + 4 bytes
            std::memcpy(buf, &p.birth, 8);
            std::memcpy(buf + 8, &p.death, 8);
            std::memcpy(buf + 16, &p.dim, 4);
            SHA256_Update(&ctx, buf, 20);
        }
    }

    void addMetadata(const std::string& version, const std::string& config) {
        SHA256_Update(&ctx, version.data(), version.size());
        SHA256_Update(&ctx, config.data(), config.size());
    }

    Checksum finalize() {
        Checksum result;
        SHA256_Final(result.bytes, &ctx);
        return result;
    }
};
```

### DeterminismContract Implementation

```cpp
class DeterminismContract {
    DeterminismLevel level;
    uint64_t seed;
    std::mt19937_64 rng;  // seeded deterministically

public:
    // Generate a deterministic thread schedule
    std::vector<size_t> getThreadSchedule(size_t num_columns, int num_threads) {
        // Use a fixed permutation based on the seed
        std::vector<size_t> columns(num_columns);
        std::iota(columns.begin(), columns.end(), 0);
        if (level >= DeterminismLevel::STRICT) {
            // Deterministic shuffle using seeded RNG
            std::shuffle(columns.begin(), columns.end(), rng);
        }
        return columns;
    }

    // Verify determinism after computation
    bool verify(const std::vector<PersistentPair>& result) {
        if (level < DeterminismLevel::PARANOID) return true;

        // Run a second computation and compare
        auto result2 = recompute();
        return compareChecksums(checksumResult(result),
                               checksumResult(result2));
    }
};
```

Back to [PH5 Engine Overview](index.md)
