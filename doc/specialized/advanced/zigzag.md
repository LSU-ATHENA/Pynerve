# Advanced zigzag persistence

### Interval matching across time

```python
from pynerve.specialized import ZigzagMatcher

matcher = ZigzagMatcher(distance_threshold=0.1)
matches = matcher.match_intervals(zigzag_result)

# Track how a feature evolves across time
for interval, time_range in matches:
    print(f"Feature: birth={interval.birth:.2f}, "
          f"active from t={time_range.start} to t={time_range.end}")
```

### Streaming zigzag

For very long time series:

```python
from pynerve.specialized import StreamingZigzag

stream = StreamingZigzag(
    max_dim=2,
    max_radius=1.0,
    window_size=10,       # number of slices in memory
    stride=2,             # slide window by 2 slices
)

for slice_idx, points_slice in enumerate(time_slices):
    stream.add_slice(points_slice, slice_idx)
    if stream.is_window_ready():
        result = stream.compute_window()
        print(f"Window {stream.window_id}: {len(result.pairs)} pairs")
```


[Back to index](index.md)
