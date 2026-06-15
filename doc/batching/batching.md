# Batching

## Quick start

```python
import pynerve.batching as batch

# Create a micro-batch processor
proc = batch.MicroBatchingEngine(
    max_batch_size=32,
    max_wait_time_ms=10,
    num_batch_threads=4,
)

# Register batch handler
proc.setBatchProcessor(lambda batch: process_batch(batch))
proc.start()

# Enqueue items (thread-safe)
for item in data_stream:
    seq_id = proc.enqueueItem(
        item, timestamp_ns=now(), symbol_id=symbol
    )

# Force flush pending items
proc.flush()

stats = proc.getStats()
# -> stats.average_batch_size = 28.3, stats.total_items_processed = 1048576
```

Process data in small batches for memory efficiency and throughput. Configurable
batch size, auto-flush by timeout, priority queuing, and symbol-level batching.
Ideal for streaming data ingestion and GPU batch processing.


## API

```cpp
#include <nerve/batching/micro_batching.hpp>

namespace nerve::batching {

struct BatchConfig {
    size_t max_batch_size = 32;
    size_t max_wait_time_ms = 10;
    size_t min_batch_size = 1;
    bool enable_dynamic_batching = true;
    bool enable_priority_batching = false;
    size_t num_batch_threads = 4;
    size_t max_queue_size = 1000;
    bool enable_zero_copy = true;
};

template <Batchable T>
struct BatchItem {
    T data;
    int64_t timestamp_ns;
    int64_t symbol_id;
    uint32_t priority;
    uint64_t sequence_id;
    function<void(const vector<T>&)> callback;
    bool isValid() const;
};

template <Batchable T>
class MicroBatchingEngine {
    explicit MicroBatchingEngine(const BatchConfig& config);
    ~MicroBatchingEngine();

    uint64_t enqueueItem(T data, int64_t timestamp_ns, int64_t symbol_id,
                         uint32_t priority = 0,
                         function<void(const vector<T>&)> callback = nullptr);
    void setBatchProcessor(BatchProcessor processor);
    void start();
    void stop();
    void flush();

    struct BatchStats {
        uint64_t total_items_processed;
        uint64_t total_batches_processed;
        double average_batch_size;
        double average_wait_time_ms;
        double average_processing_time_ms;
        uint64_t queue_overflows;
        uint64_t timeouts;
    };
    BatchStats getStats() const;
};

// Symbol-level batching with cross-symbol merging
template <typename T>
class SymbolBatchingEngine {
    explicit SymbolBatchingEngine(const SymbolBatchConfig& config);
    uint64_t enqueueSymbolItem(T data, int64_t symbol_id, int64_t timestamp_ns);
    vector<T> getBatchForSymbols(const vector<int64_t>& symbol_ids);
};

}
```

Python wrapping `MicroBatchingEngine` is available as `pynerve.batching.MicroBatchProcessor`
with the same config interface.
