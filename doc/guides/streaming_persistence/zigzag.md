# Zigzag persistence

Zigzag persistence tracks feature births and deaths **across windows** -- a feature that dies in window k may be reborn in window k+1:

```
 Windows:  [ Window 0 ] ──-> [ Window 1 ] ──-> [ Window 2 ]
                 │                │                │
 H₁ features:    │                │                │
  ┌────┐         │                │                │
  │ a  │──dies──->│   ────│        │   ────│        │
  └────┘         │                │                │
  ┌────┐         │                │                │
  │ b  │────────->│   b   │────────│   ──   │      │
  └────┘         │                │                │
                 │  ┌────┐        │                │
                 │  │ c  │────────────────────->│ c  │
                 │  └────┘        │         persists│
  ┌────┐         │                │                │
  │ d  │──dies──->│   ──   │       │                │
  └────┘         │                │                │
                 │                │                │
 Legend: [Green] = Birth  [Blue] = Persist  [Gray] = Dead
```

Zigzag tracks these lineage changes across the full stream:

```python
from pynerve._streaming_persistence import StreamingPersistence

sp = StreamingPersistence(
    chunk_size=500,
    max_dim=2,
    use_gpu=True,
    # zigzag=True  # enable cross-window tracking
)
```

The C++ implementation in `src/streaming/windowed/` implements zigzag with:
- Per-window birth/death event emission
- Cross-window pair matching via persistence pairing
- Multi-dimensional event tracking (H0, H1, ... simultaneously)

### Zigzag matching algorithm

```
Input: windows W_0, W_1, ..., W_{k-1}
Output: tracked features across all windows

1. For each window W_i:
   a. Compute persistence pairs for W_i
   b. Emit (birth_event, death_event) for each pair

2. Cross-window matching:
   For consecutive windows W_i and W_{i+1}:
       a. Filter pairs to those with persistence > noise_threshold
       b. Compute IoU (intersection over union) of point indices:
          IoU = |indices(pair_i) AND indices(pair_{i+1})|
                / |indices(pair_i) OR indices(pair_{i+1})|
       c. If IoU > match_threshold (default 0.5):
          Feature continues from W_i to W_{i+1}
       d. If feature in W_i has no match in W_{i+1}: feature died
       e. If feature in W_{i+1} has no match in W_i: new feature born

3. Emit tracked features with birth_window, death_window, and persistence_history
```

[Back to index](index.md)
