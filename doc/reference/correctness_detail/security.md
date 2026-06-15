# Security Model

Pynerve assumes the following threat model:

1. **Input data is untrusted**: All inputs are validated before processing. Malformed inputs produce errors, not undefined behavior.
2. **Memory safety**: C++ code uses RAII and `ErrorResult<T>` for all operations that can fail. No raw `new`/`delete` in production code.
3. **GPU safety**: CUDA kernel launches are followed by error checking. GPU memory allocation failures are caught and propagated.
4. **MPI safety**: MPI calls check return codes. Communicator validity is verified before use.
5. **Concurrency safety**: Lock-free data structures use appropriate memory ordering. Thread pools guarantee absence of data races.

### Non-goals

- **Cryptographic security**: Pynerve is not designed for adversarial inputs. A determined attacker can cause incorrect results.
- **Side-channel resistance**: Computation time may leak information about input structure.
- **Sandboxing**: GPU and MPI operations execute with the user's privileges.


[Back to Correctness Index](index.md)
