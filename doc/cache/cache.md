# Cache

## Quick start

```python
from pynerve.cache import DiagramCache, SmartCache

# LRU cache for persistence diagrams
cache = DiagramCache(maxsize=128, ttl=3600)
result = cache.get(points, max_dim=2)
if result is None:
    result = compute_persistence(points, max_dim=2)
    cache.set(points, result, max_dim=2)

# Smart cache: small items in memory, large items on disk
smart = SmartCache(memory_maxsize=128, disk_size_limit=10 * 1024**3)
smart.set("diagram_key", result, data=points)
cached = smart.get("diagram_key")

# Decorator for automatic caching
from pynerve.cache import cached_persistence

@cached_persistence(maxsize=256, ttl=3600)
def compute_diagram(points, max_dim=2):
    return compute_persistence(points, max_dim=max_dim)
```

LRU cache for computed features (persistence diagrams, distance matrices).
Thread-safe with hit/miss tracking and optional disk-backed persistence via
`diskcache`. Use to avoid redundant computation in iterative algorithms.


## API

```python
class DiagramCache:
    def __init__(self, maxsize=128, ttl=None, use_disk=False,
                 disk_path=None): ...
    def get(self, data: np.ndarray, **params) -> Optional[Any]: ...
    def set(self, data: np.ndarray, result: Any, **params): ...
    def clear(self): ...
    def close(self): ...

class PersistentDiagramCache(DiagramCache):
    def __init__(self, cache_dir="~/.nerve_cache",
                 size_limit=10 * 1024**3, ttl=7 * 86400): ...

class SmartCache:
    def __init__(self, memory_maxsize=128,
                 disk_size_limit=10 * 1024**3,
                 small_threshold=1024 * 1024): ...
    def get(self, key: str, data: np.ndarray = None) -> Optional[Any]: ...
    def set(self, key: str, result: Any, data: np.ndarray = None): ...

def cached_persistence(maxsize=128, ttl=None, use_disk=False,
                       key_fn=None): ...
def memoize_persistent(cache_dir="~/.nerve_memo", ttl=30 * 86400,
                       ignore_args=None): ...
def get_cache_stats(cache: DiagramCache) -> dict: ...
```

### Configuration

The `maxsize` parameter controls the LRU memory capacity in number of entries, defaulting to 128. `ttl` sets the time-to-live in seconds (no expiry when set to None). `use_disk` enables a disk-backed cache via diskcache. `size_limit` sets the disk cache size limit, defaulting to a few gigabytes.

Use when the same persistence computation repeats with identical inputs --
batch processing pipelines, grid searches, or iterative optimization loops.


## FAQ

**Q: When should I use the disk-backed cache?**
A: Use disk-backed caching when the cached diagrams are large (hundreds of megabytes or more) or when you want to persist results across process restarts. The `SmartCache` automatically moves items to disk when they exceed the `small_threshold`.

**Q: How is the cache key computed?**
A: The cache uses a hash of the input data array and all computation parameters. The default key function uses `xxhash` for fast hashing. You can provide a custom `key_fn` to the `cached_persistence` decorator.

**Q: Is the cache thread-safe?**
A: Yes. `DiagramCache` and `SmartCache` use read-write locks for concurrent access. Multiple threads can read simultaneously; writes are exclusive.
