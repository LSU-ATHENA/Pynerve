## RNG

```cpp
SeededRNG rng(42);           // deterministic, seed must be provided
uint64_t v1 = rng();         // sample from mt19937_64
uint64_t v2 = rng();

double uniform(rng, 0.0, 1.0);       // [0, 1) uniform
int uniform_int(rng, 0, 100);          // [0, 100] uniform
double normal(rng, 0.0, 1.0);          // standard normal
double exponential(rng, 1.0);          // exponential with rate 1
```


### Guarantees

- **Always deterministic** -- no default seed, no `std::random_device` fallback
- **Same seed -> same sequence** across platforms and compiler versions
- **Thread-local instances** for concurrent use (each thread gets its own state)
- Used by all randomized algorithms (HNSW, witness sampling, Mapper cover)

```python
import pynerve

# Deterministic RNG -- always seeded, bitwise reproducible
rng = pynerve.random.seed(42)
value = rng.random()  # deterministic across platforms
```


### Internal implementation

The RNG wraps `std::mt19937_64` (64-bit Mersenne Twister). This provides:

The Mersenne Twister provides a period of 2^19937 - 1 with a state size of 19968 bits (a few kilobytes). Its word size is 64 bits, it passes the BigCrush TestU01 suite, and achieves roughly 200 million values per second.

The seed is used to initialize a `std::seed_seq` with 8 rounds of
mixing, ensuring that similar seeds produce different sequences.

```cpp
class SeededRNG {
    std::mt19937_64 engine;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    // ... other distributions
};
```


### Distribution functions

```cpp
namespace nerve::core {

// Uniform [0, 1)
double uniform(SeededRNG& rng, double lo = 0.0, double hi = 1.0);

// Uniform integer [lo, hi]
int uniform_int(SeededRNG& rng, int lo, int hi);

// Normal (mean, stddev)
double normal(SeededRNG& rng, double mean = 0.0, double stddev = 1.0);

// Exponential (rate)
double exponential(SeededRNG& rng, double rate = 1.0);

// Bernoulli (probability p of true)
bool bernoulli(SeededRNG& rng, double p = 0.5);

// Categorical over weights
int categorical(SeededRNG& rng, const std::vector<double>& weights);

// Shuffle in-place
template <typename Iter>
void shuffle(SeededRNG& rng, Iter begin, Iter end);

// Sample k elements without replacement
template <typename Iter>
std::vector<typename Iter::value_type> sample(
    SeededRNG& rng, Iter begin, Iter end, size_t k);

}
```


### Thread-local RNG

Each thread gets its own `thread_local` RNG instance, seeded from the
master seed plus a thread-specific offset:

```cpp
thread_local SeededRNG tls_rng(master_seed ^ (thread_id * 0x9e3779b97f4a7c15));

uint64_t thread_safe_random() {
    return tls_rng();
}
```

This avoids synchronization overhead in parallel algorithms.


### Determinism across platforms

The Mersenne Twister implementation produces identical output across:
- x86 (GCC, Clang, MSVC)
- ARM (GCC, Clang)
- Different C++ standard library implementations

This is enforced in CI by running randomized tests on multiple platforms
and comparing outputs for the same seed.


### Python API

```python
from pynerve.determinism import seed

# Create a deterministic RNG
rng = seed(42)
print(rng.random())        # float64 in [0, 1)
print(rng.randint(0, 10))  # int in [0, 10]
print(rng.randn())         # standard normal

# Shuffle
arr = list(range(100))
rng.shuffle(arr)

# Sample without replacement
subset = rng.sample(arr, 10)
```


### Common pitfalls

1. **Default seed**: There is NO default seed. Always provide one for
   reproducible results.

2. **Thread safety**: The RNG is NOT thread-safe. Use thread-local
   instances in parallel code.

3. **State size**: Each RNG instance stores a few kilobytes of state.
   Creating thousands of instances is wasteful.

4. **Seeding with time**: Do NOT use `time(nullptr)` as a seed -- it
   changes infrequently and leads to correlated sequences.

5. **Distribution state**: Distribution objects (`uniform_dist`, etc.)
   store internal state. Do not share across RNG instances.


### Cross-references

- `pynerve.validation.determinism`: Determinism validation across runs
- `pynerve.core.thread_pool`: Thread pool using thread-local RNG
