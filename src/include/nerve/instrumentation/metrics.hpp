
#pragma once
#include <cmath>
#include <cstdint>
#include <functional>

namespace nerve::instrumentation
{
enum class MetricKind : uint8_t
{
    COUNTER = 0,
    GAUGE = 1,
    HISTOGRAM = 2
};
struct alignas(64) MetricEvent
{
    const char *name;
    MetricKind kind;
    double value;
    uint64_t timestamp_ns;
    uint32_t tags_hash;
    uint32_t component_id;
    uint16_t operation_id;
    uint8_t reserved[10];
    bool isValid() const
    {
        return name != nullptr && timestamp_ns > 0 && std::isfinite(value) &&
               (kind == MetricKind::COUNTER || kind == MetricKind::GAUGE ||
                kind == MetricKind::HISTOGRAM);
    }
};
static_assert(sizeof(MetricEvent) == 64, "MetricEvent must be 64 bytes for cache efficiency");
using MetricSinkFn = std::function<void(const MetricEvent &)>;
class MetricsRegistry
{
public:
    static MetricsRegistry &instance();
    void registerSink(MetricSinkFn sink);
    void unregisterSinks();
    bool hasSinks() const;
    void emit(const MetricEvent &event);
    void emitCounter(const char *name, double value, uint32_t component_id = 0,
                     uint32_t tags_hash = 0);
    void emitGauge(const char *name, double value, uint32_t component_id = 0,
                   uint32_t tags_hash = 0);
    void emitHistogram(const char *name, double value, uint32_t component_id = 0,
                       uint32_t tags_hash = 0);
    uint32_t registerComponent(const char *name);
    uint32_t registerOperation(const char *name);
    const char *getComponentName(uint32_t component_id) const;
    const char *getOperationName(uint32_t operation_id) const;

private:
    MetricsRegistry() = default;
    static constexpr size_t MAX_SINKS = 8;
    static constexpr size_t MAX_COMPONENTS = 256;
    static constexpr size_t MAX_OPERATIONS = 512;
    MetricSinkFn sinks_[MAX_SINKS];
    size_t sink_count_;
    const char *component_names_[MAX_COMPONENTS];
    const char *operation_names_[MAX_OPERATIONS];
    uint32_t next_component_id_;
    uint32_t next_operation_id_;
    mutable bool sinks_enabled_;
};
class MetricTimer
{
public:
    MetricTimer(const char *name, MetricKind kind = MetricKind::HISTOGRAM,
                uint32_t component_id = 0, uint32_t tags_hash = 0);
    ~MetricTimer();
    void stop();
    void resume();
    bool isStopped() const;
    uint64_t elapsedNs() const;

private:
    const char *name_;
    MetricKind kind_;
    uint32_t component_id_;
    uint32_t tags_hash_;
    uint64_t start_time_ns_;
    uint64_t elapsed_time_ns_;
    bool stopped_;
    uint64_t getTimestampNs() const;
};
#define EMIT_COUNTER(name, value)                                                                  \
    nerve::instrumentation::MetricsRegistry::instance().emitCounter(name, value)
#define EMIT_GAUGE(name, value)                                                                    \
    nerve::instrumentation::MetricsRegistry::instance().emitGauge(name, value)
#define EMIT_HISTOGRAM(name, value)                                                                \
    nerve::instrumentation::MetricsRegistry::instance().emitHistogram(name, value)
#define METRIC_TIMER(name)                                                                         \
    nerve::instrumentation::MetricTimer _timer(name, nerve::instrumentation::MetricKind::HISTOGRAM)
#define METRIC_TIMER_COMPONENT(name, component_id)                                                 \
    nerve::instrumentation::MetricTimer _timer(                                                    \
        name, nerve::instrumentation::MetricKind::HISTOGRAM, component_id)
#define REGISTER_COMPONENT(name)                                                                   \
    static const uint32_t COMPONENT_##name =                                                       \
        nerve::instrumentation::MetricsRegistry::instance().registerComponent(#name)
#define REGISTER_OPERATION(name)                                                                   \
    static const uint32_t OPERATION_##name =                                                       \
        nerve::instrumentation::MetricsRegistry::instance().registerOperation(#name)
} // namespace nerve::instrumentation
