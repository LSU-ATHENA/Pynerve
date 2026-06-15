# Streaming Architecture

The streaming pipeline flows through six stages. (1) **Data source**  --  a file, network stream, or async iterator feeds points. (2) **Chunk reader** reads fixed-size chunks from HDF5, NPY, or memory-mapped storage. (3) **Lock-free chunk queue** (MPMC, atomic indices, no mutex) buffers chunks between producer and consumer. (4) **Worker threads (or GPU streams)** each process one chunk independently. (5) **Persistence computation per chunk** runs distance + filtration + reduction on each chunk. (6) **Result merger** handles overlap and cross-window matching, feeding an **async iterator** that yields merged results in concat, mean, or max mode.


[Back to Architecture Index](index.md)
