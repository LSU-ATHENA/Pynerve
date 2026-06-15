# When NOT to Use

- **Experimental or research code** where PH6's latest innovations are preferred
- **Maximum throughput** on very large datasets where validation overhead is not justified
- **Quick exploration** where PH4's simpler API and lower overhead suffice
- **Resource-constrained environments** where the 10-20% additional memory for checksumming and logging matters

### PH5 vs PH4: Decision Guide

For speed on exact sparse data, PH4 is the baseline while PH5 is an estimated 10-20% faster, making PH5 the recommendation. For exact dense data, both are similar and either is fine. Memory use is baseline for PH4 and roughly 10-20% higher for PH5, so PH4 is recommended when memory is constrained. Reproducibility is basic on PH4 and PARANOID-level on PH5; PH5 is recommended when strong reproducibility is needed. Checksum validation and differentiability are PH5-only features, making PH5 the clear choice when those are required. For simplicity, PH4 is simpler with a smaller API surface, so PH4 is recommended when simplicity is the priority.

Back to [PH5 Engine Overview](index.md)
