# `nerve::determinism` -- Determinism System

### Seed Management

```cpp
#include <nerve/determinism.hpp>

namespace nerve::determinism;

void seed(uint64_t s) noexcept;         // set thread-local seed
uint64_t get_seed() noexcept;           // get thread-local seed
uint64_t next_seed() noexcept;          // mt19937-64 from seed (seeds RNG on first call)
```

**Cost:** O(1) for all operations.

### GPU Reduction (Device Code)

```cpp
#if defined(__CUDACC__)

template <int kBlockSize>
__device__ double warpReduceSum(double val);    // shuffle-based warp reduction

template <int kBlockSize>
__device__ double blockReduceSum(double val);   // shared-memory tree reduction
```

**Cost:** O(log warpSize) for warp reduce, O(log blockSize) for block reduce. Deterministic.

**Example:**

```cpp
__global__ void myReductionKernel(const double* input, double* output) {
    extern __shared__ double shared[];
    double thread_sum = computeSomething(input);
    double block_sum = nerve::determinism::blockReduceSum<256>(thread_sum);
    if (threadIdx.x == 0) output[blockIdx.x] = block_sum;
}
```

### MPI Reduction

```cpp
void deterministic_reduce(const double* send, double* recv, int n,
                          int root, MPI_Comm comm);
void deterministic_allreduce(const double* send, double* recv, int n,
                             MPI_Comm comm);
void deterministic_allreduce_binned(const double* send, double* recv, int n,
                                    MPI_Comm comm);
```

**Cost:** O(n) communication, O(n) computation.

### DeterminismContract

```cpp
#include <nerve/core/determinism_contract.hpp>

namespace nerve::core;

enum class DeterminismLevel : uint8_t { NONE, BASIC, STRICT, AUDIT };

struct DeterminismContract {
    DeterminismLevel level;
    bool enable_checksum_validation;
    bool enable_deterministic_threading;
    bool enable_deterministic_random;
    uint64_t rng_seed[16];
    uint8_t params_hash[32];

    bool isValid() const;
    std::vector<std::string> validationErrors() const;
    void setRngSeed(uint64_t seed);
    template <typename T> void addParameterToHash(const T& param);
    void finalizeParamsHash();
    std::string serialize() const;
    static DeterminismContract deserialize(const std::string&);
};

class DeterminismEnforcer {
public:
    static DeterminismContract createContract(DeterminismLevel level, std::string name);
    static DeterminismContract create(uint64_t seed, uint64_t params_hash,
                                       bool bitwise_reproducible,
                                       uint8_t precision_level, uint8_t algorithm_version);
    static bool canSatisfyContract(const DeterminismContract& contract);
    static bool supportsDeterministicThreading();
    static bool supportsDeterministicGpu();
    static bool validateComputationResult(const DeterminismContract& contract,
                                           const void* result_data, uint64_t expected_checksum);
};

class DeterminismContext {
public:
    explicit DeterminismContext(const DeterminismContract& contract);
    ~DeterminismContext();
    const DeterminismContract& contract() const;
    bool isActive() const;
    void setFailOnNonDeterministic(bool fail);
};
```

**Cost:** O(1) construction. Checksum validation is O(pairs) for results.

<- [C++ API Overview](index.md)
