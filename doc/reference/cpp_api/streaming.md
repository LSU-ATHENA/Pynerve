# `nerve::streaming` -- Streaming Persistence

```cpp
#include <nerve/persistence/adaptive_acceleration/streaming/streaming_processor.hpp>

namespace nerve::streaming;

struct StreamingConfig {
    size_t chunk_size = 1000;
    size_t max_buffered_chunks = 3;
    bool use_gpu = true;
    int num_workers = 4;
};

class StreamingProcessor {
    StreamingResult processStreaming(DataStream& stream);
    StreamingResult processStreamingProgressive(DataStream& stream);
};

struct StreamingResult {
    std::vector<Pair> pairs;
    std::vector<double> betti_curve;
    size_t chunks_processed;
    double total_time_ms;
};
```

**Cost (processStreaming):** O(n_chunks * persistence_cost_per_chunk). Memory: O(chunk_size^2).

<- [C++ API Overview](index.md)
