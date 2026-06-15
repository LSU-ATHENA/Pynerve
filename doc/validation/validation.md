# Validation and Benchmarks

Pynerve provides per-kernel microbenchmarks (GPU + CPU), automated regression
suites (PH5/PH6 benchmark CI), API signature validation between C++ and
Python, bitwise determinism checks, and CUDA kernel launch audit.

```python
import pynerve

result = pynerve.validation.benchmark(n_points=5000, max_dim=2, backend="cuda")  # requires pynerve.validation submodule
print(result)
```


## Modules

- **[benchmarks.md](benchmarks.md)** -- Microbenchmarks, kernel perf, PH5/PH6 CI
- **[determinism.md](determinism.md)** -- Cross-run bitwise reproducibility checks
- **[launch_audit.md](launch_audit.md)** -- CUDA kernel error checking and validation
- **[contracts.md](contracts.md)** -- API signature validation between C++ and Python


## Complexity

The validation suite has the following complexity characteristics: determinism checks run in O(p) for p pairs, CUDA kernel audits in O(k) for k launches, individual microbenchmarks in O(trials x work) including warmup, and the full CI benchmark suite across roughly 10 configurations completes in approximately 2--5 minutes.



## Quick API reference

```python
import pynerve.validation as val

# Full pipeline benchmark
result = val.benchmark(n_points=5000, max_dim=2, backend="cuda")

# Determinism check
ok = val.check_determinism(result1, result2, level=val.DeterminismLevel.STRICT)

# CUDA launch audit
report = val.audit_cuda_launches()
if not report.passed:
    for entry in report.unverified_launches:
        print(f"Missing error check: {entry.file}:{entry.line}")

# API contract validation
report = val.validate_all_bindings()
print(f"Passed: {report.passed}/{report.total_functions}")
```


## Integration with developer workflow

### Pre-commit hooks

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: validate-bindings
      name: Validate C++/Python API bindings
      entry: python -m pynerve.validation.contracts.validate_all_bindings
      language: system
      files: \.(py|cpp|hpp)$
    - id: audit-cuda
      name: Audit CUDA kernel launches
      entry: python -m pynerve.validation.audit_cuda --fail-on-error
      language: system
      files: \.cu$
```

### CI regression thresholds

```yaml
# .github/workflows/benchmarks.yml
- name: Run benchmarks
  run: |
    python -m pynerve.validation.benchmark --suite=ph6 \
      --expected-file=benchmark_expected.json \
      --tolerance=0.05  # fail if >5% regression
```


## Validation hierarchy

The validation system is organized into layers ordered by scope and frequency. Unit tests validate individual function correctness on every PR. API contracts check C++/Python binding consistency also on every PR. The CUDA audit verifies kernel launch correctness on every PR. Determinism checks ensure bitwise reproducibility during nightly runs. Benchmarks guard against performance regression during nightly runs. The full CI suite encompassing all of the above plus integration tests runs at release time.


## Common failure patterns

1. **Determinism failure on GPU**: Atomic operations (e.g., in cohomology reduction) produce non-deterministic results. Switch to `DeterminismLevel::AUDIT` which forces fixed-tree reductions, or use CPU backend for reproducible results.
2. **Benchmark regression on new hardware**: GPU benchmarks tuned for A100 may show different scaling on H100. Rebaseline expected times with `benchmark --rebuild-expected`.
3. **CUDA launch audit fails after refactor**: Moving kernel launches to new files breaks the file:line tracking. Re-run the audit and update `CHECK_CUDA` placements.


## FAQ

**Q: How do I run only one validation check instead of the full suite?**
A: Each validation tool can be invoked independently. Use `pynerve.validation.benchmark` for benchmarks, `check_determinism` for determinism, `audit_cuda_launches` for launch checks, and `validate_all_bindings` for API contracts.

**Q: What happens when a validation check fails in CI?**
A: Each tool returns a structured report with pass/fail status and detailed diagnostics. CI jobs are configured to exit non-zero on failure, and the failure output includes the specific violations or regressions found.

**Q: Can I add custom validation checks?**
A: Yes. Each subsystem supports extension: benchmarks accept custom kernel registration, determinism accepts user-defined contracts, and the binding validator supports custom validation rules.


### Cross-references

- `pynerve.cuda`: CUDA infrastructure validated by launch_audit
- `pynerve.core.rng`: RNG determinism
- `pynerve.cuda.determinism`: GPU determinism
- `pynerve.algorithms`: Benchmarked algorithm implementations
