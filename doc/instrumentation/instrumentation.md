# Instrumentation

## Quick start

```python
from pynerve.instrumentation import metrics

# Emit performance counters
metrics.counter("persistence.computations", 1)
metrics.gauge("memory.usage_mb", 256.0)
metrics.histogram("batch.processing_time_ms", 12.5)

# Timer context
with metrics.timer("persistence.compute"):
    result = compute_persistence(points)

# Stability certificate
cert = metrics.create_persistence_certificate(
    bottleneck_bound=0.01,
    wasserstein_bound=0.05,
    numerical_residual=1e-8,
    confidence=200,
)
# -> cert.isHighQuality() == True
```

Performance counters, histogram timers, high-dimensional error tracking, and
stability certificate generation. Cache-line-aligned metric events with
configurable sinks for integration with external monitoring.


## API

```cpp
#include <nerve/instrumentation/metrics.hpp>
#include <nerve/instrumentation/stability_certificates.hpp>
#include <nerve/instrumentation/instrumentation.hpp>

namespace nerve::instrumentation {

// Metrics
struct alignas(64) MetricEvent {
    const char* name;
    MetricKind kind;     // COUNTER, GAUGE, HISTOGRAM
    double value;
    uint64_t timestamp_ns;
    uint32_t tags_hash;
    uint32_t component_id;
    uint16_t operation_id;
};

class MetricsRegistry {
    static MetricsRegistry& instance();
    void emitCounter(const char* name, double value, ...);
    void emitGauge(const char* name, double value, ...);
    void emitHistogram(const char* name, double value, ...);
    uint32_t registerComponent(const char* name);
    uint32_t registerOperation(const char* name);
};

class MetricTimer {
    MetricTimer(const char* name, MetricKind kind = HISTOGRAM, ...);
    void stop();
    void resume();
    uint64_t elapsedNs() const;
};

// Stability certificates (64 bytes)
struct StabilityCertificate {
    float bottleneck_upper_bound;
    float wasserstein_upper_bound;
    float numerical_residual;
    uint8_t confidence_bucket;
    uint8_t stability_level;
    uint32_t computation_hash;
    uint64_t timestamp_ns;
    float highdim_condition_estimate;
    float effective_rank_estimate;
    float compression_ratio;
    float memory_efficiency_score;
    bool isValid() const;
    bool isHighQuality() const;
};

class CertificateFactory {
    static StabilityCertificate createPersistenceCertificate(
        float bottleneck_bound, float wasserstein_bound,
        float numerical_residual, uint8_t confidence_bucket,
        uint32_t computation_hash, uint64_t computation_time_ns);
    static StabilityCertificate createPh5Ph6Certificate(
        float bottleneck_bound, float wasserstein_bound,
        float numerical_residual, uint8_t confidence_bucket,
        uint32_t computation_hash, uint64_t computation_time_ns,
        float condition_estimate, float effective_rank, ...);
};

class CertificateValidator {
    ValidationResult validateForMlTraining(const StabilityCertificate& cert);
    ValidationResult validateForResearch(const StabilityCertificate& cert);
};

// Top-level manager
class InstrumentationManager {
    static InstrumentationManager& instance();
    void initialize(const ManagerConfig& config);
    void shutdown();
    void flush();
    InstrumentationStats getStats() const;
};

}
```

```python
from pynerve.instrumentation import metrics

metrics.counter(name, value)
metrics.gauge(name, value)
metrics.histogram(name, value)
metrics.timer(name)              # context manager
metrics.create_persistence_certificate(...)
metrics.validate_for_ml(cert)
```

### Macros

```cpp
EMIT_COUNTER(name, value)
EMIT_GAUGE(name, value)
EMIT_HISTOGRAM(name, value)
METRIC_TIMER(name)
CREATE_PERSISTENCE_CERTIFICATE(...)
```

Instrumentation is compile-time toggleable via `NERVE_ENABLE_METRICS`,
`NERVE_ENABLE_STABILITY_CERTIFICATES` flags -- disabled code paths become
no-ops with zero overhead.
