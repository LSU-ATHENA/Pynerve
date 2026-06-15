
#pragma once
#include "error_events.hpp"
#include "metrics.hpp"
#include "stability_certificates.hpp"
#include "topological_delta.hpp"
namespace nerve
{
namespace instrumentation
{
namespace config
{
#ifndef NERVE_ENABLE_METRICS
#define NERVE_ENABLE_METRICS 1
#endif
#ifndef NERVE_ENABLE_ERROR_EVENTS
#define NERVE_ENABLE_ERROR_EVENTS 1
#endif
#ifndef NERVE_ENABLE_STABILITY_CERTIFICATES
#define NERVE_ENABLE_STABILITY_CERTIFICATES 1
#endif
#ifndef NERVE_ENABLE_TOPOLOGICAL_DELTA
#define NERVE_ENABLE_TOPOLOGICAL_DELTA 1
#endif
#ifndef NERVE_MAX_METRIC_SINKS
#define NERVE_MAX_METRIC_SINKS 8
#endif
#ifndef NERVE_MAX_ERROR_SINKS
#define NERVE_MAX_ERROR_SINKS 4
#endif
#ifndef NERVE_CACHE_LINE_SIZE
#define NERVE_CACHE_LINE_SIZE 64
#endif
#if NERVE_ENABLE_METRICS
#define EMIT_METRIC_IF_ENABLED(name, kind, value)                                                  \
    nerve::instrumentation::MetricsRegistry::instance().emit({name, kind, value, 0, 0, 0, 0})
#else
#define EMIT_METRIC_IF_ENABLED(name, kind, value)                                                  \
    do                                                                                             \
    {                                                                                              \
    } while (0)
#endif
#if NERVE_ENABLE_ERROR_EVENTS
#define EMIT_ERROR_IF_ENABLED(code, component, operation)                                          \
    nerve::instrumentation::ErrorEventsRegistry::instance().emitError(code, component, operation)
#else
#define EMIT_ERROR_IF_ENABLED(code, component, operation)                                          \
    do                                                                                             \
    {                                                                                              \
    } while (0)
#endif
} // namespace config
class InstrumentationManager
{
public:
    struct ManagerConfig
    {
        bool enable_metrics = NERVE_ENABLE_METRICS;
        bool enable_error_events = NERVE_ENABLE_ERROR_EVENTS;
        bool enable_stability_certificates = NERVE_ENABLE_STABILITY_CERTIFICATES;
        bool enable_topological_delta = NERVE_ENABLE_TOPOLOGICAL_DELTA;
        size_t max_buffer_size = 1000;
        uint64_t flush_interval_ns = 100000000;
    };
    static InstrumentationManager &instance();
    void initialize();
    void initialize(const ManagerConfig &config);
    void shutdown();
    bool isEnabled() const { return enabled_; }
    const ManagerConfig &getConfig() const { return config_; }
    void flush();
    struct InstrumentationStats
    {
        uint64_t metrics_emitted;
        uint64_t errors_emitted;
        uint64_t certificates_created;
        uint64_t deltas_detected;
        uint64_t buffer_overflows;
        uint64_t flush_count;
        double average_flush_time_ns;
    };
    InstrumentationStats getStats() const;
    void resetStats();

private:
    InstrumentationManager() = default;
    ManagerConfig config_;
    bool enabled_;
    bool initialized_;
    mutable InstrumentationStats stats_;
    void flushMetrics();
    void flushErrors();
    void updateFlushStats(uint64_t flush_time_ns);
};
class InstrumentationGuard
{
public:
    InstrumentationGuard();
    explicit InstrumentationGuard(const InstrumentationManager::ManagerConfig &config);
    ~InstrumentationGuard();
    InstrumentationGuard(const InstrumentationGuard &) = delete;
    InstrumentationGuard &operator=(const InstrumentationGuard &) = delete;
    InstrumentationGuard(InstrumentationGuard &&other) noexcept;
    InstrumentationGuard &operator=(InstrumentationGuard &&other) noexcept;

private:
    bool initialized_;
};
#ifdef EMIT_COUNTER
#undef EMIT_COUNTER
#endif
#define EMIT_COUNTER(name, value)                                                                  \
    EMIT_METRIC_IF_ENABLED(name, nerve::instrumentation::MetricKind::COUNTER, value)
#ifdef EMIT_GAUGE
#undef EMIT_GAUGE
#endif
#define EMIT_GAUGE(name, value)                                                                    \
    EMIT_METRIC_IF_ENABLED(name, nerve::instrumentation::MetricKind::GAUGE, value)
#ifdef EMIT_HISTOGRAM
#undef EMIT_HISTOGRAM
#endif
#define EMIT_HISTOGRAM(name, value)                                                                \
    EMIT_METRIC_IF_ENABLED(name, nerve::instrumentation::MetricKind::HISTOGRAM, value)
#ifdef EMIT_ERROR
#undef EMIT_ERROR
#endif
#define EMIT_ERROR(code, component, operation) EMIT_ERROR_IF_ENABLED(code, component, operation)
#if NERVE_ENABLE_METRICS
#define INSTRUMENT_TIMER(name)                                                                     \
    nerve::instrumentation::MetricTimer _timer(name, nerve::instrumentation::MetricKind::HISTOGRAM)
#else
#define INSTRUMENT_TIMER(name)                                                                     \
    do                                                                                             \
    {                                                                                              \
    } while (0)
#endif
#if NERVE_ENABLE_ERROR_EVENTS
#define INSTRUMENT_ERROR_CONTEXT(code, component, operation)                                       \
    nerve::instrumentation::ErrorContext _error_context(code, component, operation)
#else
#define INSTRUMENT_ERROR_CONTEXT(code, component, operation)                                       \
    do                                                                                             \
    {                                                                                              \
    } while (0)
#endif
#if NERVE_ENABLE_STABILITY_CERTIFICATES
#define CREATE_CERTIFICATE(factoryMethod, ...) factoryMethod(__VA_ARGS__)
#else
#define CREATE_CERTIFICATE(factoryMethod, ...)                                                     \
    nerve::instrumentation::StabilityCertificate {}
#endif
#if NERVE_ENABLE_TOPOLOGICAL_DELTA
#define DETECT_CHANGES(old_pd, new_pd, contract)                                                   \
    nerve::instrumentation::TopologicalChangeDetector().detectChanges(old_pd, new_pd, contract)
#else
#define DETECT_CHANGES(old_pd, new_pd, contract)                                                   \
    std::vector<nerve::instrumentation::TopologicalDelta> {}
#endif
#define REGISTER_INSTRUMENTATION_COMPONENT(name)                                                   \
    static const uint32_t INSTRUMENT_COMPONENT_##name =                                            \
        nerve::instrumentation::MetricsRegistry::instance().registerComponent(#name)
#define REGISTER_INSTRUMENTATION_OPERATION(name)                                                   \
    static const uint32_t INSTRUMENT_OPERATION_##name =                                            \
        nerve::instrumentation::MetricsRegistry::instance().registerOperation(#name)
class PerformanceSection
{
public:
    explicit PerformanceSection(const char *name, uint32_t component_id = 0);
    ~PerformanceSection();
    void markCheckpoint(const char *checkpoint_name);

private:
    const char *name_;
    uint32_t component_id_;
    uint64_t start_time_ns_;
    std::vector<std::pair<const char *, uint64_t>> checkpoints_;
};
#define PERFORMANCE_SECTION(name) nerve::instrumentation::PerformanceSection _perf_section(name)
#define PERFORMANCE_CHECKPOINT(name) _perf_section.markCheckpoint(name)
namespace features
{
constexpr bool hasMetrics()
{
    return NERVE_ENABLE_METRICS != 0;
}
constexpr bool hasErrorEvents()
{
    return NERVE_ENABLE_ERROR_EVENTS != 0;
}
constexpr bool hasStabilityCertificates()
{
    return NERVE_ENABLE_STABILITY_CERTIFICATES != 0;
}
constexpr bool hasTopologicalDelta()
{
    return NERVE_ENABLE_TOPOLOGICAL_DELTA != 0;
}
} // namespace features
struct BuildInfo
{
    bool metrics_enabled;
    bool error_events_enabled;
    bool stability_certificates_enabled;
    bool topological_delta_enabled;
    size_t cache_line_size;
    size_t max_metric_sinks;
    size_t max_error_sinks;
    const char *build_timestamp;
    const char *compiler_version;
    static BuildInfo get();
    std::string toString() const;
};
} // namespace instrumentation
} // namespace nerve
