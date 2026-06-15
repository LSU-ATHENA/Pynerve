# CPU dispatch flowchart

```
                   nerve.compute_persistence(points)
                               |
                        ┌──────┴──────┐
                        │ CUDA tensor?│
                        └──────┬──────┘
                      yes/         \no
                        v           v
              CUDA_HYBRID     CPU tensor/ndarray
              backend                |
                              ┌──────┴──────┐
                              │  CPUID ->    │
                              │  AVX-512?   │
                              └──────┬──────┘
                            yes/         \no
                              v           v
                  512-bit dist+    ┌──────┴──────┐
                  reduction        │  CPUID ->    │
                                   │  AVX2?      │
                                   └──────┬──────┘
                                 yes/         \no
                                   v           v
                       256-bit dist+    ┌──────┴──────┐
                       reduction        │  CPUID ->    │
                                        │  SSE4.1?    │
                                        └──────┬──────┘
                                       yes/         \no
                                         v           v
                             128-bit dist+    scalar fallback
                             reduction
                                   |
                              ┌────┴────┐
                              │ n<1000? │
                              └────┬────┘
                            yes/       \no
                              v         v
                      single-thread  ┌──┴──┐
                      stack alloc    │     │
                                     │n<   │
                                     │10000│
                                     └──┬──┘
                                   yes/   \no
                                     v     v
                              thread pool  GPU dispatch
                              work-stealing (if GPU avail)
                                           or Distributed
                                           (if MPI)
```

[Back to index](index.md)
