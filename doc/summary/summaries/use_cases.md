# Use cases

- **Storage-limited (edge devices):** use the EXACT strategy -- a fixed size of around one kilobyte.
- **Real-time monitoring:** use the APPROXIMATE strategy -- under five milliseconds per summary.
- **Streaming data:** use the INCREMENTAL strategy via `updateSummary()`.
- **Large-scale comparison:** use the EXACT strategy for fast binary comparison.
- **Logging + debugging:** use the EXACT strategy to retain full metadata.


[Back to index](index.md)
